from __future__ import annotations

import os
import timeit
import warnings

warnings.filterwarnings(
    "ignore",
)

import functools
from contextlib import nullcontext
from typing import Optional

import math
import numpy as np
import einops
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group

from tqdm import tqdm
import wandb

# from mcpfn import TabICL
from tabimpute.model.mcpfn import MCPFN
from tabimpute.prior.genload import LoadPriorDataset
from tabimpute.train.optim import get_scheduler, SharedGradNormWeighterEMA
from tabimpute.train.train_config import build_parser
from tabimpute.model.bar_distribution import FullSupportBarDistribution
from tabimpute.model.encoders import torch_nanmean
from tabimpute.prior.training_set_generation import MissingnessPrior

# # Set CUDA environment variables for debugging
# os.environ["TORCH_USE_CUDA_DSA"] = "1"
# os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

warnings.filterwarnings(
    "ignore",
    message=".*The PyTorch API of nested tensors is in prototype stage.*",
    category=UserWarning,
)

class Timer:
    """Context manager for timing code execution."""

    def __enter__(self):
        self.start_time = timeit.default_timer()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.elapsed = timeit.default_timer() - self.start_time
        return False  # Don't suppress exceptions


def ddp_cleanup(func):
    """Decorator to clean up DDP process group after method execution.

    Ensures that destroy_process_group() is called if DDP is enabled,
    even if an exception occurs during method execution.
    """

    @functools.wraps(func)
    def wrapper(self, *args, **kwargs):
        try:
            return func(self, *args, **kwargs)
        finally:
            if self.ddp:
                destroy_process_group()

    return wrapper


class Trainer:
    """This class handles the complete training lifecycle for TabICL, including:

    - Environment setup and distributed training configuration
    - Model building and initialization
    - Optimizer, scheduler, and dataloader configuration
    - Checkpoint management and recovery
    - Training loop execution with gradient accumulation
    - Metrics tracking and logging using wandb

    Parameters
    ----------
    config : argparse.Namespace
        Training configuration parameters containing all settings for model,
        optimizer, distributed training, and data generation.
    """

    def __init__(self, config, step_progress: Optional[tqdm] = None):
        self.config = config
        self.configure_ddp()
        self.configure_wandb()
        self.build_model()
        self.configure_prior(self.config.prior_dir)
        self.configure_optimizer()
        self.configure_amp()
        
        borders = torch.load(self.config.borders_path).to(self.config.device)
        self.bar_distribution = FullSupportBarDistribution(borders=borders)

        self.step_progress = step_progress

        self.len_train = 8
        self.len_val = 2
        
        # Initialize running statistics trackers for task loss normalization
        # Uses exponentially weighted mean and variance (similar to EMA but handles negative values)
        # Default to 3 tasks: mcar, mar, mnar
        # num_tasks = len(self.train_dataset.missingness_inducer.patterns) if hasattr(self.train_dataset, 'missingness_inducer') else 3
        # self.task_loss_ema_decay = getattr(self.config, 'task_loss_ema_decay', 0.99)
        # self.task_loss_means = torch.zeros(num_tasks, device=self.config.device, requires_grad=False)
        # self.task_loss_vars = torch.ones(num_tasks, device=self.config.device, requires_grad=False)  # Initialize to 1 to avoid division issues
        # self.task_loss_initialized = torch.zeros(num_tasks, dtype=torch.bool, device=self.config.device)
        
        self.load_checkpoint()

    def configure_ddp(self):
        """Set up distributed training and system configuration.

        This method:
        1. Configures distributed data parallel (DDP) if enabled
        2. Sets up device and process information
        3. Adjusts batch size for multi-GPU training
        4. Sets random seeds for reproducibility
        """
        # Setup distributed training
        print("Configuring DDP")
        self.ddp = int(os.environ.get("RANK", -1)) != -1

        if self.ddp:
            init_process_group(backend="nccl")
            self.ddp_rank = int(os.environ["RANK"])
            self.ddp_local_rank = int(os.environ["LOCAL_RANK"])
            self.ddp_world_size = int(os.environ["WORLD_SIZE"])
            self.master_process = self.ddp_rank == 0
            self.config.device = f"cuda:{self.ddp_local_rank}"
            torch.cuda.set_device(self.config.device)

            # Adjust batch size for distributed training
            original_batch_size = self.config.batch_size
            self.config.batch_size = math.ceil(
                original_batch_size / self.ddp_world_size
            )

            if self.master_process:
                print(f"DDP training with {self.ddp_world_size} processes")
                if original_batch_size % self.ddp_world_size == 0:
                    print(f"Per-GPU batch size: {self.config.batch_size}")
                else:
                    print(
                        f"Original batch size ({original_batch_size}) cannot be divided by world size ({self.ddp_world_size}).\n"
                        f"Use ceiling division for equal per-GPU batch size: {self.config.batch_size}.\n"
                        f"Effective batch size is {self.config.batch_size * self.ddp_world_size}.\n"
                    )
        else:
            self.master_process = True
            self.ddp_rank = 0
            self.ddp_world_size = 1
            self.ddp_local_rank = 0

        self.curr_step = 0  # Initialize current step for training

        # Set random seeds
        seed_offset = self.ddp_rank if self.ddp else 0
        np.random.seed(self.config.np_seed + seed_offset)
        torch.manual_seed(self.config.torch_seed + seed_offset)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    def configure_wandb(self):
        """Set up Weights & Biases logging."""
        if self.config.wandb_log and self.master_process:
            print("Configuring wandb")
            id_path = os.path.join(self.config.checkpoint_dir, "wand_id.txt")
            if self.config.wandb_id is None:
                if os.path.exists(id_path):
                    with open(id_path, "r") as f:
                        self.config.wandb_id = f.read().strip()

            self.wandb_run = wandb.init(
                dir=self.config.wandb_dir,
                project=self.config.wandb_project,
                name=self.config.wandb_name,
                id=self.config.wandb_id,
                config=self.config,
                resume="allow",
                mode=self.config.wandb_mode,
            )

            with open(id_path, "w") as f:
                f.write(self.wandb_run.id)
        else:
            self.wandb_run = None

    def build_model(self):
        """Build and initialize the TabICL model."""
        print("Building model")
        self.model_config = {
            "max_classes": self.config.max_classes,
            "embed_dim": self.config.embed_dim,
            "col_num_blocks": self.config.col_num_blocks,
            "col_nhead": self.config.col_nhead,
            "col_num_inds": self.config.col_num_inds,
            "row_num_blocks": self.config.row_num_blocks,
            "row_nhead": self.config.row_nhead,
            "row_num_cls": self.config.row_num_cls,
            "row_rope_base": self.config.row_rope_base,
            "icl_num_blocks": self.config.icl_num_blocks,
            "icl_nhead": self.config.icl_nhead,
            "ff_factor": self.config.ff_factor,
            "dropout": self.config.dropout,
            "activation": self.config.activation,
            "norm_first": self.config.norm_first,
        }

        model = MCPFN(nhead=2)
        model.to(device=self.config.device)

        # print(f"Loading tabpfn model weights from {self.config.tabpfn_path}")
        # model.model.load_state_dict(
        #     torch.load(self.config.tabpfn_path, weights_only=True)
        # )

        # self.tabpfn_model = TabPFNModel(device=self.config.device)

        if self.master_process:
            num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            print(f"Model has {num_params} parameters.")

        # # Freeze model components if requested
        # if self.config.freeze_col:
        #     model.col_embedder.eval()
        #     for param in model.col_embedder.parameters():
        #         param.requires_grad = False

        # if self.config.freeze_row:
        #     model.row_interactor.eval()
        #     for param in model.row_interactor.parameters():
        #         param.requires_grad = False

        # if self.config.freeze_icl:
        #     model.icl_predictor.eval()
        #     for param in model.icl_predictor.parameters():
        #         param.requires_grad = False

        # Compile model if requested
        if self.config.model_compile:
            model = torch.compile(model, dynamic=True)
            if self.master_process:
                print("Model compiled successfully.")

        # Wrap model into DDP container if using distributed training
        if self.ddp:
            self.model = DDP(
                model, device_ids=[self.ddp_local_rank], broadcast_buffers=False
            )
            self.raw_model = self.model.module
        else:
            self.model = model
            self.raw_model = model

    def configure_prior(self, prior_dir: Optional[str] = None):
        """
        Sets up a tabular dataset generator that creates synthetic datasets
        during training with controllable properties and data distributions.
        """
        if prior_dir is not None:
            # Load pre-generated prior data from disk
            self.train_dataset = LoadPriorDataset(
                data_dir=prior_dir + "/train",
                batch_size=self.config.batch_size,
                ddp_world_size=self.ddp_world_size,
                ddp_rank=self.ddp_rank,
                start_from=self.config.load_prior_start,
                delete_after_load=self.config.delete_after_load,
                device=self.config.prior_device,
            )

            val_dataset = LoadPriorDataset(
                data_dir=prior_dir + "/val",
                batch_size=self.config.batch_size,
                ddp_world_size=self.ddp_world_size,
                ddp_rank=self.ddp_rank,
                start_from=self.config.load_prior_start,
                delete_after_load=self.config.delete_after_load,
                device=self.config.prior_device,
            )
            self.val_dataloader = DataLoader(
                val_dataset,
                batch_size=None,  # No additional batching since PriorDataset handles batching internally
                shuffle=False,
                num_workers=1,
                prefetch_factor=4,
                pin_memory=True if self.config.prior_device == "cpu" else False,
                pin_memory_device=(
                    self.config.device if self.config.prior_device == "cpu" else ""
                ),
            )
        else:
            gen_config = {
                "num_rows_low": self.config.min_seq_len,
                "num_rows_high": self.config.max_seq_len,
                "num_cols_low": self.config.min_features,
                "num_cols_high": self.config.max_features,
                "p_missing": 0.4,
                # MNAR configs
                "threshold_quantile": 0.25,
                "n_core_items": 5,
                "n_genres": 3,
                "n_policies": 4,
                # Latent Factor configs
                "latent_rank_low": 1,
                "latent_rank_high": 15,
                "latent_spike_p": 0.3,
                "latent_slab_sigma": 2.0,
                "apply_feature_warping_prob": 0.0,
                "apply_quantization_prob": 0.0,
                # Non-linear Factor configs
                "spline_knot_k": [3, 5, 7],
                "gp_length_scale_low": 0.3,
                "gp_length_scale_high": 2.0,
                "fourier_dim_low": 100,
                "fourier_dim_high": 501,
                # Robust-PCA configs
                "rpca_beta_a": 2,
                "rpca_beta_b": 30,
                # Soft Polarization configs
                "soft_polarization_alpha": 2.5,
                "soft_polarization_epsilon": 0.05,
                # User Cascade configs
                "cascade_n_genres": 5,
                "cascade_delta": 1.5,
                # Cluster Level configs
                "cluster_level_n_row_clusters": 8,
                "cluster_level_n_col_clusters": 8,
                "cluster_level_tau_r_std": 1.0,
                # Spatial Block configs
                "spatial_block_n_blocks": 5,
                "spatial_block_p_geom": 0.2,
                # Last few ones
                "censor_quantile": 0.1,
                "two_phase_cheap_fraction": 0.4,
                "two_phase_beta": 2.5,
                "skip_logic_p_noise": 0.9,
                "cold_start_fraction": 0.3,
                "cold_start_gamma": 0.15,
                # MAR configs
                'mar_config': {
                    "num_layers_upper": 3,
                    "hidden_lower": 1,
                    "hidden_upper": 100,
                    "activation": "relu",
                    "N": 100, # Row size of X (reduced for testing)
                    "T": 50, # Column size of X (reduced for testing)
                    "row_neighbor_upper": 5, # Upper bound of row neighbor (reduced for testing)
                    "col_neighbor_upper": 5, # Upper bound of column neighbor (reduced for testing)
                    "seed": 42,
                    "neighbor_type": "random"
                },
                # MAR Block configs
                'mar_block_config': {
                    "N": 100,
                    "T": 50,
                    "row_blocks": 10,
                    "col_blocks": 10,
                    "convolution_type": "mean"
                },
                # MAR Sequential Bandit configs
                'mar_sequential_bandit_config': {
                    'algorithm': 'epsilon_greedy',
                    'pooling': False, 
                    'epsilon': 0.4, 
                    'epsilon_decay': 0.99, 
                    'random_seed': 42
                }
            }
            # Create data on the fly
            self.train_dataset = MissingnessPrior(
                generator_type="latent_factor",
                missingness_type=self.config.missingness_type,
                config=gen_config,
                batch_size=self.config.batch_size,
                verbose=False,
            )
            self.val_dataloader = None
            # self.probs = [1/len(self.train_dataset.missingness_inducer.patterns)] * len(self.train_dataset.missingness_inducer.patterns)
            # self.prob_names = self.train_dataset.missingness_inducer.pattern_names
        # Create dataloader for efficient loading and prefetching
        self.train_dataloader = DataLoader(
            self.train_dataset,
            batch_size=None,  # No additional batching since PriorDataset handles batching internally
            shuffle=False,
            num_workers=1,
            prefetch_factor=4,
            pin_memory=True,
            pin_memory_device=(
                self.config.device if self.config.prior_device == "cpu" else ""
            ),
        )

    def configure_optimizer(self):
        """Configure optimizer and scheduler."""
        print("Configuring optimizer")
        self.optimizer = optim.AdamW(
            params=self.raw_model.parameters(),
            lr=self.config.lr,
            weight_decay=self.config.weight_decay,
        )
        self.scheduler = get_scheduler(config=self.config, optimizer=self.optimizer)
        
        # # initialize once
        # self.weighter = SharedGradNormWeighterEMA(
        #     num_tasks=len(self.train_dataset.missingness_inducer.patterns),
        #     alpha=0.5,     # balancing strength
        #     lr=0.025,      # step size for weight updates
        #     beta=0.9,      # EMA smoothing
        #     device=self.config.device
        # )

    def configure_amp(self):
        """Configure automatic mixed precision (AMP) for training."""
        print("Configuring AMP")
        self.amp = self.config.amp and "cuda" in self.config.device
        self.scaler = torch.GradScaler("cuda", enabled=self.amp)
        if self.amp:
            if self.master_process:
                print(f"Automatic Mixed Precision is enabled.")
            self.amp_ctx = torch.autocast(
                device_type="cuda",
                dtype=(
                    torch.float16 if self.config.dtype == "float16" else torch.float32
                ),
            )
        else:
            self.amp_ctx = nullcontext()

    def get_latest_checkpoint(self):
        """Returns the latest checkpoint from `checkpoint_dir`

        Only considers files with the .ckpt extension (PyTorch checkpoint files).
        """
        ckpt_dir = self.config.checkpoint_dir

        if not os.path.isdir(ckpt_dir):
            return None

        # Filter for files with "ckpt" extension matching the pattern "step-*.ckpt"
        checkpoints = [
            f
            for f in os.listdir(ckpt_dir)
            if f.startswith("step-") and f.endswith(".ckpt")
        ]

        if not checkpoints:
            return None

        # Sort the checkpoint files by step number and get the latest
        try:
            latest_checkpoint = sorted(
                checkpoints, key=lambda x: int(x.split("-")[1].split(".")[0])
            )[-1]
            checkpoint_path = os.path.join(ckpt_dir, latest_checkpoint)
            return checkpoint_path
        except Exception as e:
            print(f"Error parsing checkpoint filenames: {e}")
            return None

    def load_checkpoint(self):
        """Load model and training state from checkpoint.

        First checks if `checkpoint_path` is directly specified. If not, attempts to find
        the latest checkpoint in the checkpoint directory.
        """

        checkpoint_path = None
        if hasattr(self.config, "checkpoint_path") and self.config.checkpoint_path:
            checkpoint_path = self.config.checkpoint_path
        # elif hasattr(self.config, "checkpoint_dir") and self.config.checkpoint_dir:
        #     checkpoint_path = self.get_latest_checkpoint()

        if checkpoint_path is None or not os.path.exists(checkpoint_path):
            print("No checkpoint found, starting from scratch.")
            return

        print(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(
            checkpoint_path, map_location=self.config.device, weights_only=True
        )

        # Load model state
        if "state_dict" not in checkpoint:
            raise ValueError("Checkpoint does not contain model state")

        # If 'module.' not in prefix of keys, add it
        if self.ddp:
            if not any(key.startswith("module.") for key in checkpoint["state_dict"]):
                checkpoint["state_dict"] = {
                    f"module.{key}": val
                    for key, val in checkpoint["state_dict"].items()
                }

        self.model.load_state_dict(checkpoint["state_dict"])

        # Optionally load optimizer and scheduler state
        if self.config.only_load_model:
            print("Only loading model weights")
        else:
            self.optimizer.load_state_dict(checkpoint["optimizer_state"])
            self.scheduler.load_state_dict(checkpoint["scheduler_state"])
            self.curr_step = checkpoint["curr_step"]
            print(f"Resuming training at step {self.curr_step}")
            
            # Load running statistics state if available (backward compatible with EMA checkpoints)
            if "task_loss_means" in checkpoint and "task_loss_vars" in checkpoint:
                if checkpoint["task_loss_means"].shape == self.task_loss_means.shape:
                    self.task_loss_means = checkpoint["task_loss_means"].to(self.config.device)
                    self.task_loss_means.requires_grad_(False)
                    self.task_loss_vars = checkpoint["task_loss_vars"].to(self.config.device)
                    self.task_loss_vars.requires_grad_(False)
                    if "task_loss_initialized" in checkpoint:
                        self.task_loss_initialized = checkpoint["task_loss_initialized"].to(self.config.device)
                    else:
                        # If initialized flag not present, assume all are initialized if means are non-zero
                        self.task_loss_initialized = (self.task_loss_means != 0)
                    print(f"Loaded task loss statistics - means: {self.task_loss_means.cpu().tolist()}, stds: {torch.sqrt(self.task_loss_vars).cpu().tolist()}")
                else:
                    print(f"Warning: Statistics shape mismatch. Expected {self.task_loss_means.shape}, got {checkpoint['task_loss_means'].shape}. Starting with fresh statistics.")
            elif "task_loss_emas" in checkpoint:
                # Backward compatibility: convert old EMA checkpoints to new format
                if checkpoint["task_loss_emas"].shape == self.task_loss_means.shape:
                    # Convert EMA to mean, assume initial variance of 1
                    self.task_loss_means = checkpoint["task_loss_emas"].to(self.config.device)
                    self.task_loss_means.requires_grad_(False)
                    self.task_loss_vars = torch.ones_like(self.task_loss_means, device=self.config.device)
                    if "task_loss_ema_initialized" in checkpoint:
                        self.task_loss_initialized = checkpoint["task_loss_ema_initialized"].to(self.config.device)
                    else:
                        self.task_loss_initialized = (self.task_loss_means != 0)
                    print(f"Converted old EMA checkpoint - means: {self.task_loss_means.cpu().tolist()}")
                else:
                    print(f"Warning: EMA shape mismatch. Expected {self.task_loss_means.shape}, got {checkpoint['task_loss_emas'].shape}. Starting with fresh statistics.")

    def save_checkpoint(self, name: str):
        """Save model and training state to checkpoint file.

        Parameters
        ----------
        name : str
            Filename for the checkpoint
        """

        # if self.ddp and self.ddp_local_rank == 0:
        os.makedirs(self.config.checkpoint_dir, exist_ok=True)
        checkpoint_path = os.path.join(self.config.checkpoint_dir, name)
        checkpoint = {
            "config": self.model_config,
            "state_dict": self.raw_model.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "scheduler_state": self.scheduler.state_dict(),
            "curr_step": self.curr_step,
            # "task_loss_means": self.task_loss_means.cpu(),
            # "task_loss_vars": self.task_loss_vars.cpu(),
            # "task_loss_initialized": self.task_loss_initialized.cpu(),
        }
        torch.save(checkpoint, checkpoint_path)

    def manage_checkpoint(self):
        """
        Manages the number of temporary checkpoints by deleting the oldest ones
        if the count exceeds `max_checkpoints`. Permanent checkpoints are ignored.
        """
        ckpt_dir = self.config.checkpoint_dir
        limit = self.config.max_checkpoints

        # Filter for files with "ckpt" extension matching the pattern "step-*.ckpt"
        checkpoints = [
            f
            for f in os.listdir(ckpt_dir)
            if f.startswith("step-") and f.endswith(".ckpt")
        ]
        temp_checkpoints = []
        for ckpt in checkpoints:
            try:
                step = int(ckpt.split("-")[1].split(".")[0])
                # Consider a checkpoint temporary if its step is not divisible by save_perm_every
                if step % self.config.save_perm_every != 0:
                    temp_checkpoints.append((step, ckpt))
            except Exception as e:
                print(f"Error parsing checkpoint filenames: {e}")
                continue  # Ignore files that don't match the format

        # Sort temporary checkpoints by step number (ascending)
        temp_checkpoints.sort(key=lambda x: x[0])

        # Remove oldest temporary checkpoints if limit is exceeded
        num_to_delete = len(temp_checkpoints) - limit
        if num_to_delete > 0:
            for step, ckpt_name in temp_checkpoints[:num_to_delete]:
                ckpt_path = os.path.join(ckpt_dir, ckpt_name)
                try:
                    os.remove(ckpt_path)
                except Exception as e:
                    print(f"Error removing checkpoint {ckpt_path}: {e}")

    @ddp_cleanup
    def train(self):
        """Main training loop.

        Iterates through batches, processes them, updates model parameters,
        and handles checkpoint saving and metric logging.
        """
        if self.master_process:
            step_progress = tqdm(
                range(self.curr_step, self.config.max_steps), desc="Step", leave=True
            )
            # self.update_probs(batch_size=self.config.batch_size)
        else:
            step_progress = range(self.curr_step, self.config.max_steps)
            
        # if self.ddp:
        #     # Use the default process group for barrier
        #     torch.distributed.barrier()
            
        # Broadcast updated probabilities from master to all processes
        # probs_tensor = torch.tensor(self.probs, device=self.config.device)
        # torch.distributed.broadcast(probs_tensor, src=0)
        # # self.probs = probs_tensor.cpu().tolist()
        # self.probs = [round(prob, 3) for prob in self.probs]
        # self.probs[0] = 1.0 - sum(self.probs[1:])
        
        # Update the missingness inducer probabilities on all processes
        # self.train_dataset.missingness_inducer.probs = self.probs

        # train_dataloader = iter(self.train_dataloader)
        
        # val_dataloader can be used for validation if needed
        # if self.val_dataloader is not None:
        #     val_dataloader = iter(self.val_dataloader)
        

        for step in step_progress:
            torch.cuda.empty_cache()
            # Zero gradients at the start of each step
            self.optimizer.zero_grad(set_to_none=True)
            
            results = {
                "ce": 0.0,
                "mae": 0.0,
                "missing_ce": 0.0,
                "missing_mae": 0.0
            }
            
            # Generate batches for all missingness patterns and compute losses
            # original_probs = self.train_dataset.missingness_inducer.probs
            # names = ['mcar', 'mar', 'mnar']
            results_dict = {}
            # task_losses = []
            
            # for i in range(len(self.train_dataset.missingness_inducer.patterns)):
            #     self.train_dataset.missingness_inducer.probs = [0.0] * len(self.train_dataset.missingness_inducer.patterns)
            #     self.train_dataset.missingness_inducer.probs[i] = 1.0
            #     batch, pattern_weights = self.train_dataset.get_batch(self.config.batch_size)
            #     self.train_dataset.missingness_inducer.probs[i] = 0.0
            #     results_dict[names[i]] = self.run_batch(batch, pattern_weights=pattern_weights, is_train=True)
            #     # Collect the loss tensor for this task
            #     task_losses.append(results_dict[names[i]]["missing_ce_tensor"])
            
            # # Restore original probabilities
            # self.train_dataset.missingness_inducer.probs = original_probs
            
            # # Update running statistics for task losses and normalize
            # task_losses_tensor = torch.stack(task_losses)  # Shape: (num_tasks,)
            
            # if self.curr_step % self.config.update_probs_every == 0:
            #     # GradNorm weight update (first-order safe)
            #     shared_params = [p for p in self.raw_model.parameters() if p.requires_grad]
            #     self.weighter.update(task_losses_tensor, shared_params)
            
            # Weighted loss for actual training
            # total_loss = self.weighter.weighted_total_loss(task_losses_tensor)
            batch, pattern_weights = self.train_dataset.get_batch(self.config.batch_size)
            results = self.run_batch(batch, pattern_weights=pattern_weights, is_train=True)
            total_loss = results["missing_ce_tensor"]
            self.optimizer.zero_grad()
            total_loss.backward()
            
            # Gradient clipping if enabled
            # if self.config.gradient_clipping > 0:
            #     # if self.amp:
            #     #     self.scaler.unscale_(self.optimizer)
            #     nn.utils.clip_grad_norm_(
            #         self.model.parameters(), self.config.gradient_clipping
            #     )
            
            # # Update model parameters
            # if self.amp:
            #     self.scaler.step(self.optimizer)
            #     self.scaler.update()
            # else:
            #     self.optimizer.step()
            
            self.optimizer.step()
            
            # Update learning rate
            self.scheduler.step()

            # # Aggregate results for logging
            # for name in names:
            #     results["ce"] += results_dict[name]["ce"]
            #     results["mae"] += results_dict[name]["mae"]
            #     results["missing_ce"] += results_dict[name]["missing_ce"]
            #     results["missing_mae"] += results_dict[name]["missing_mae"]
            
            # Average results
            # # num_patterns = len(names)
            # results["ce"] /= num_patterns
            # results["mae"] /= num_patterns
            # results["missing_ce"] /= num_patterns
            # results["missing_mae"] /= num_patterns

            # Clear CUDA cache to free memory
            torch.cuda.empty_cache()

            self.curr_step = step + 1
            
            if self.master_process:
                print_vals = {
                    'ce': round(results["ce"], 3),
                    'mae': round(results["mae"], 3),
                    'missing_ce': round(results["missing_ce"], 3),
                    'missing_mae': round(results["missing_mae"], 3)
                }
                # Update progress bar with rounded values for cleaner display
                if step_progress is not None:
                    step_progress.set_postfix(**print_vals)

                # Save checkpoints
                is_temp_save = self.curr_step % self.config.save_temp_every == 0
                is_perm_save = self.curr_step % self.config.save_perm_every == 0

                if is_temp_save or is_perm_save:
                    ckpt_name = f"step-{self.curr_step}.ckpt"
                    self.save_checkpoint(name=ckpt_name)

                    # Manage checkpoint limit only for temporary checkpoints
                    if (
                        is_temp_save
                        and not is_perm_save
                        and self.config.max_checkpoints > 0
                    ):
                        self.manage_checkpoint()
            
            # Logging to Weights & Biases
            if self.wandb_run is not None:
                # for i, name in enumerate(names):
                #     results[f"missing_ce_{name}"] = results_dict[name]["missing_ce"]
                #     results[f"missing_mae_{name}"] = results_dict[name]["missing_mae"]
                #     results[f"weight_{name}"] = self.weighter.weights[i].item()
                wandb.log(print_vals, step=self.curr_step)

        # Save last checkpoint
        ckpt_name = f"step-{self.curr_step}.ckpt"
        self.save_checkpoint(name=ckpt_name)
        
    def update_probs(self, batch_size=20):
        """
        Update the missingness probabilities to upweight the missingness types that have worse loss
        and downweight the missingness types that have better loss
        """
        with torch.no_grad():
            # Create a batch of datasets with fixed missingness proportions
            pattern_losses = []
            
            # Store original probabilities
            original_probs = self.train_dataset.missingness_inducer.probs
            
            for i, pattern in enumerate(self.train_dataset.missingness_inducer.patterns):
                probs = [0.0] * len(self.train_dataset.missingness_inducer.patterns)
                probs[i] = 1.0
                
                # Set temporary probabilities
                self.train_dataset.missingness_inducer.probs = probs
                
                batch, pattern_weights = self.train_dataset.get_batch(batch_size)
                sub_results = self.run_batch(batch, pattern_weights=pattern_weights, is_train=False)
                pattern_losses.append(sub_results["missing_ce"])
                
            # Restore original probabilities
            self.train_dataset.missingness_inducer.probs = original_probs
            
            # Update mixing probabilities using softmax of negative loss
            if len(pattern_losses) > 1:  # Only update if we have multiple patterns
                # Convert losses to probabilities using softmax of negative loss
                negative_losses = torch.tensor([loss for loss in pattern_losses], device=self.config.device)
                new_probs = F.softmax(negative_losses, dim=0)
                self.probs = new_probs.cpu().tolist()

    def validate_micro_batch(self, micro_seq_len, micro_train_size):
        """
        Validate consistent sequence length and train size within a micro batch.

        Ensures all datasets in a micro batch share the same sequence length and
        train/test split position, required for efficient batch processing during
        gradient accumulation.

        Parameters
        ----------
        micro_seq_len : Tensor (micro_batch_size,)
            Sequence lengths for each dataset.

        micro_train_size : Tensor (micro_batch_size,)
            Training sizes (split positions) for each dataset.

        Returns
        -------
        tuple (int, int)
            The common (seq_len, train_size) for the micro batch.

        Raises
        ------
        ValueError
            If sequence lengths or train sizes are inconsistent.
        """
        if len(torch.unique(micro_seq_len)) > 1:
            raise ValueError(
                "All datasets in the micro batch must have the same sequence length."
            )

        if len(torch.unique(micro_train_size)) > 1:
            raise ValueError(
                "All datasets in the micro batch must have the same training size."
            )

        seq_len = micro_seq_len[0].item()
        train_size = micro_train_size[0].item()

        return seq_len, train_size

    def align_micro_batch(self, micro_X, micro_y, micro_d, seq_len):
        """
        Truncate micro batch tensors to required dimensions.

        Truncates sequence length and feature dimensions to the validated `seq_len`
        and the maximum active features (`micro_d.max()`) respectively. This optimizes
        memory and computation by removing unused tensor elements.

        Parameters
        ----------
        micro_X : Tensor (B, T, H)
            Input features per dataset.

        micro_y : Tensor (B, T)
            Target labels per dataset.

        micro_d : Tensor (B,)
            Number of active features per dataset.

        seq_len : int
            Validated sequence length for this micro batch.

        Returns
        -------
        tuple (Tensor, Tensor)
            Truncated (micro_X, micro_y) tensors with shapes
            (B, seq_len, micro_d.max()) and (B, seq_len).
        """
        # Truncate sequence length
        if micro_X.shape[1] > seq_len:
            micro_X = micro_X[:, :seq_len]

        if micro_y.shape[1] > seq_len:
            micro_y = micro_y[:, :seq_len]

        # Truncate feature dimension
        max_features = micro_d.max().item()
        if micro_X.shape[-1] > max_features:
            micro_X = micro_X[..., :max_features]

        return micro_X, micro_y

    def run_micro_batch(
        self,
        micro_batch,
        micro_pattern_weights,
        micro_batch_idx,
        num_micro_batches,
        is_train=True,
        is_tabpfn=False,
    ):
        """Process a micro batch for gradient accumulation.

        Parameters
        ----------
        micro_batch : tuple
            (micro_X, micro_y, micro_d, micro_seq_len, micro_train_size) tensors for the micro batch

        micro_batch_idx : int
            Index of the current micro batch

        num_micro_batches : int
            Total number of micro batches

        Returns
        -------
        dict
            Result dictionary
        """
        micro_X, micro_y, micro_d, micro_seq_len, micro_train_size = micro_batch
        # seq_len, train_size = self.validate_micro_batch(micro_seq_len, micro_train_size)
        seq_len = micro_seq_len[
            0
        ].item()  # this should be the same for all datasets in the micro batch
        micro_X, micro_y = self.align_micro_batch(micro_X, micro_y, micro_d, seq_len)

        # Move to device
        micro_X = micro_X.to(self.config.device)
        micro_y = micro_y.to(self.config.device)
        micro_d = micro_d.to(self.config.device)
        micro_train_size = micro_train_size.to(self.config.device)
        
        # micro_pattern_weights = micro_pattern_weights.to(self.config.device)

        # Compute mean along dim=2 (last dimension), ignoring NaNs
        # mean_vals = torch.nanmean(micro_X, dim=2, keepdim=True)  # shape: [1, 2000, 1]

        # Find the NaNs
        # nan_mask = torch.isnan(micro_X)  # shape: [1, 2000, 20]

        # Expand mean_vals to match x's shape for indexing
        # mean_vals_expanded = mean_vals.expand_as(micro_X)

        # Replace NaNs with corresponding mean values
        # micro_X[nan_mask] = mean_vals_expanded[nan_mask]

        y_train = micro_y.clone()

        # Create a mask for each row up to its train_size
        mask = torch.zeros_like(micro_y, dtype=torch.bool)
        for i in range(len(micro_train_size)):
            mask[i, : micro_train_size[i]] = True

        # Set values after train_size to nan for each row
        y_train[~mask] = torch.nan

        # Add a new column of mask to X at the first position
        micro_X = torch.cat([mask.unsqueeze(2), micro_X], dim=2)

        mask_reshaped = einops.rearrange(mask, "b t -> t b")

        # Set DDP gradient sync for last micro batch only
        if self.ddp:
            self.model.require_backward_grad_sync = (
                micro_batch_idx == num_micro_batches - 1
            )

        if is_train:  # train
            with self.amp_ctx:
                pred = self.model(
                    micro_X, y_train, micro_d
                )  # (B, test_size, max_classes)
                # pred = einops.rearrange(pred, "b t h -> t b h")
                # loss = self.bar_distribution(
                #     logits=pred, y=einops.rearrange(micro_y, "b t -> t b")
                # )
                loss = self.bar_distribution(logits=pred, y=micro_y)
                loss = einops.rearrange(loss, "b t -> t b")
                
                # mean = self.bar_distribution.mean(pred)

                # Get MSE loss
                # loss = (mean - einops.rearrange(micro_y, "b t -> t b")).pow(2)

            # Scale loss for gradient accumulation and backpropagate
            # micro_pattern_weights = micro_pattern_weights.unsqueeze(0)
            
            # loss = loss * micro_pattern_weights # reweight the loss by the pattern weights
            
            scaled_loss = (loss).mean() / num_micro_batches
            # self.scaler.scale(scaled_loss).backward()
            missing_loss = (loss[~mask_reshaped]).mean() / num_micro_batches
            
            # self.scaler.scale(missing_loss).backward()

        else:  # val
            with torch.no_grad():
                if not is_tabpfn:
                    pred = self.model(
                        micro_X, y_train, micro_d
                    )  # (B, test_size, max_classes)
                else:
                    pred = self.tabpfn_model.forward(
                        micro_X, y_train, micro_train_size
                    )  # (B, test_size, max_classes)

                # pred = einops.rearrange(pred, "b t h -> t b h")
                # loss = self.bar_distribution(
                #     logits=pred, y=einops.rearrange(micro_y, "b t -> t b")
                # )
                loss = self.bar_distribution(logits=pred, y=micro_y)
                loss = einops.rearrange(loss, "b t -> t b")
                
                micro_pattern_weights = micro_pattern_weights.unsqueeze(0)
                loss = loss * micro_pattern_weights # reweight the loss by the pattern weights
                
                scaled_loss = torch_nanmean(loss).mean() / num_micro_batches
                missing_loss = (
                    torch_nanmean(loss[~mask_reshaped]).mean() / num_micro_batches
                )

        # with torch.no_grad():
        micro_results = {}
        micro_results["ce"] = scaled_loss.item()
        micro_results["missing_ce"] = missing_loss.item()
        micro_results["missing_ce_tensor"] = missing_loss
        median = self.bar_distribution.median(logits=pred)
        
        accuracy = (median - micro_y).abs()  # mae
        accuracy = einops.rearrange(accuracy, "b t -> t b")
        
        micro_results["mae"] = torch_nanmean(accuracy).mean().item()
        micro_results["missing_mae"] = (
            torch_nanmean(accuracy[~mask_reshaped]).mean().item()
        )
            
        return micro_results

    def run_batch(self, batch, pattern_weights, is_train=True, is_tabpfn=False):
        """
        Trains the model on a batch of datasets. Handles gradient accumulation by
        splitting the batch into micro-batches. Supports variable-sized datasets
        by padding. Skips micro-batches on CUDA OOM errors. Updates model
        parameters and returns loss and accuracy metrics.

        Parameters
        ----------
        batch: tuple
            Contains tensors (X, y, d, seq_len, train_size) for the batch.
            X and y can be Tensors or NestedTensors (for variable sequence lengths).

        Returns
        ------
        dict
            Dictionary containing 'ce' (cross-entropy loss) and 'accuracy'.

        Raises
        ------
        RuntimeError
            If more than 10% of micro-batches fail due to OOM errors.
        """
        if is_train:
            self.model.train()
            # Note: optimizer.zero_grad() is called in the training loop before running batches
        else:
            self.model.eval()

        # Pad nested tensors to the same size
        batch = [t.to_padded_tensor(padding=0.0) if t.is_nested else t for t in batch]

        # Split the batch into micro-batches along the first dimension
        num_micro_batches = math.ceil(
            self.config.batch_size / self.config.micro_batch_size
        )
        micro_batches = [
            torch.split(t, self.config.micro_batch_size, dim=0) for t in batch
        ]
        micro_batches = list(zip(*micro_batches))
        micro_pattern_weights = torch.split(pattern_weights, self.config.micro_batch_size, dim=0)

        results = {"ce": 0.0, "mae": 0.0, "missing_ce": 0.0, "missing_mae": 0.0}
        missing_ce_tensors = []  # Collect loss tensors separately to preserve gradients
        failed_batches = 0

        for idx, (micro_batch, micro_pattern_weights) in enumerate(zip(micro_batches, micro_pattern_weights)):
            try:
                micro_results = self.run_micro_batch(
                    micro_batch, micro_pattern_weights, idx, num_micro_batches, is_train, is_tabpfn
                )
                for k, v in micro_results.items():
                    if k == "missing_ce_tensor":
                        missing_ce_tensors.append(v)
                    else:
                        results[k] = results[k] + v # not in-place
            except Exception as e:
                print(
                    f"Warning: OOM error in micro-batch {idx+1}/{num_micro_batches} at step {self.curr_step}. Skipping."
                )
                print(e)
                torch.cuda.empty_cache()
                failed_batches += 1
                continue
        
        # Average scalar results
        num_valid_batches = len(micro_batches) - failed_batches
        if num_valid_batches > 0:
            results["ce"] /= num_valid_batches
            results["mae"] /= num_valid_batches
            results["missing_ce"] /= num_valid_batches
            results["missing_mae"] /= num_valid_batches
            
            # Average loss tensors while preserving computation graph
            if missing_ce_tensors:
                # Stack and average to preserve gradients
                results["missing_ce_tensor"] = torch.stack(missing_ce_tensors).mean()
            else:
                results["missing_ce_tensor"] = torch.tensor(0.0, device=self.config.device, requires_grad=True)
        else:
            results["missing_ce_tensor"] = torch.tensor(0.0, device=self.config.device, requires_grad=True)
        failure_ratio = failed_batches / num_micro_batches
        if failure_ratio > 0.1:
            raise RuntimeError(
                f"({failure_ratio:.1%}) of micro-batches failed due to OOM at step {self.curr_step}. "
                f"Please check configuration to reduce memory consumption."
            )

        return results


if __name__ == "__main__":
    parser = build_parser()
    config = parser.parse_args()
    
    trainer = Trainer(config)

    trainer.curr_step = config.start_step
    trainer.train()
    
    # dummy_batch = (torch.randn(10, 20, 20), torch.randn(10, 20), torch.randn(10), torch.randn(10), torch.randn(10)) # (X, y, d, seq_len, train_size)
    # dummy_batch = [t.to(trainer.config.device) for t in dummy_batch]
    # model_output = trainer.model(dummy_batch[0], dummy_batch[1], dummy_batch[2])
    
    # print(model_output.shape)
    
    # loss = trainer.bar_distribution(logits=einops.rearrange(model_output, "b t h -> t b h"), y=einops.rearrange(dummy_batch[1], "b t -> t b"))
    
    # print(loss.shape)
