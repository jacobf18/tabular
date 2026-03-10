import torch
from torch import nn
from tqdm import tqdm
import time
from typing import Dict
from tabimpute.model.bar_distribution import FullSupportBarDistribution
import os
import importlib.resources
import math
from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR, OneCycleLR

from tabimpute.model.model_new import TabImputeModel
from tabimpute.prior.training_set_generation import MissingnessPrior
from tabimpute.train.callbacks import Callback, WandbLoggerCallback

# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

try:
    import schedulefree
except ImportError:
    schedulefree = None


def _build_optimizer(
    model: TabImputeModel,
    optimizer_name: str,
    lr: float,
    weight_decay: float,
    optimizer_betas: tuple[float, float],
    optimizer_eps: float,
):
    optimizer_name = optimizer_name.lower()
    params = model.parameters()

    if optimizer_name in {"schedulefree_adamw", "adamw_schedulefree"}:
        if schedulefree is None:
            print(
                "Warning: schedulefree is not installed. Falling back to torch.optim.AdamW."
            )
            return torch.optim.AdamW(
                params=params,
                lr=lr,
                weight_decay=weight_decay,
                betas=optimizer_betas,
                eps=optimizer_eps,
            )
        return schedulefree.AdamWScheduleFree(
            params=params,
            lr=lr,
            weight_decay=weight_decay,
            betas=optimizer_betas,
            eps=optimizer_eps,
        )

    if optimizer_name == "adamw":
        first_param = next(model.parameters())
        use_fused = first_param.is_cuda
        # Prefer fused AdamW when available; silently fallback if unsupported.
        try:
            return torch.optim.AdamW(
                params=params,
                lr=lr,
                weight_decay=weight_decay,
                betas=optimizer_betas,
                eps=optimizer_eps,
                fused=use_fused,
            )
        except (TypeError, RuntimeError):
            return torch.optim.AdamW(
                params=params,
                lr=lr,
                weight_decay=weight_decay,
                betas=optimizer_betas,
                eps=optimizer_eps,
            )

    if optimizer_name == "adam":
        return torch.optim.Adam(
            params=params,
            lr=lr,
            weight_decay=weight_decay,
            betas=optimizer_betas,
            eps=optimizer_eps,
        )

    raise ValueError(
        f"Unknown optimizer_name '{optimizer_name}'. Supported: "
        "schedulefree_adamw, adamw, adam"
    )


def _build_scheduler(
    optimizer,
    scheduler_name: str,
    total_steps: int,
    warmup_steps: int,
    min_lr_ratio: float,
    one_cycle_pct_start: float,
):
    scheduler_name = scheduler_name.lower()

    if scheduler_name in {"none", "constant"}:
        return None

    if scheduler_name == "cosine":
        base_lr = optimizer.param_groups[0]["lr"]
        return CosineAnnealingLR(
            optimizer=optimizer,
            T_max=max(1, total_steps),
            eta_min=base_lr * min_lr_ratio,
        )

    if scheduler_name == "warmup_cosine":
        warmup_steps = max(0, min(warmup_steps, total_steps - 1))

        def lr_lambda(current_step: int):
            step = current_step + 1
            if warmup_steps > 0 and step <= warmup_steps:
                return step / warmup_steps

            progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
            progress = min(max(progress, 0.0), 1.0)
            cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
            return min_lr_ratio + (1.0 - min_lr_ratio) * cosine

        return LambdaLR(optimizer=optimizer, lr_lambda=lr_lambda)

    if scheduler_name == "one_cycle":
        max_lr = [group["lr"] for group in optimizer.param_groups]
        final_div_factor = max(1.0, 1.0 / max(min_lr_ratio, 1e-6))
        return OneCycleLR(
            optimizer=optimizer,
            max_lr=max_lr,
            total_steps=max(1, total_steps),
            pct_start=one_cycle_pct_start,
            anneal_strategy="cos",
            div_factor=25.0,
            final_div_factor=final_div_factor,
        )

    raise ValueError(
        f"Unknown scheduler_name '{scheduler_name}'. Supported: "
        "none, cosine, warmup_cosine, one_cycle"
    )

def train(model: TabImputeModel, 
          prior: MissingnessPrior, 
          bar_distribution: FullSupportBarDistribution,
          criterion: nn.Module,
          epochs: int, 
          lr: float = 1e-4, 
          weight_decay: float = 1e-2,
          grad_clip_norm: float | None = 1.0,
          optimizer_name: str = "schedulefree_adamw",
          optimizer_betas: tuple[float, float] = (0.9, 0.999),
          optimizer_eps: float = 1e-8,
          scheduler_name: str = "none",
          warmup_ratio: float = 0.05,
          warmup_steps: int | None = None,
          min_lr_ratio: float = 0.1,
          one_cycle_pct_start: float = 0.1,
          checkpoint_every: int | None = 5000,
          return_metrics: bool = False,
          device: torch.device = None,
          callbacks: list[Callback] = None, 
          ckpt: Dict[str, torch.Tensor] = None, 
          multi_gpu: bool = False,
          run_name: str = 'tabimpute-new'):
    """
    Trains our model on the given prior using the given criterion.

    Args:
        model: (TabImputeModel) our PyTorch model
        prior: (MissingnessPrior) Missingness prior
        bar_distribution: (FullSupportBarDistribution) our bar distribution
        epochs: (int) the number of epochs we train for, the number of steps that constitute an epoch are decided by the prior
        device: (torch.device) the device we are using
        callbacks: A list of callback instances to execute at the end of each epoch. These can be used for
            logging, validation, or other custom actions.
        ckpt (Dict[str, torch.Tensor], optional): A checkpoint dictionary containing the model and optimizer states,
            as well as the last completed epoch. If provided, training resumes from this checkpoint.

    Returns:
        (TabImputeModel) trained model
    """
    work_dir = 'workdir/'+run_name
    os.makedirs(work_dir, exist_ok=True)
    if callbacks is None:
        callbacks = []
    if not device:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    compute_dtype = torch.bfloat16 if "cuda" in str(device) else torch.float32
    model.to(device=device, dtype=compute_dtype)
    
    optimizer = _build_optimizer(
        model=model,
        optimizer_name=optimizer_name,
        lr=lr,
        weight_decay=weight_decay,
        optimizer_betas=optimizer_betas,
        optimizer_eps=optimizer_eps,
    )
    total_steps = epochs
    if warmup_steps is None:
        warmup_steps = int(total_steps * warmup_ratio)
    scheduler = _build_scheduler(
        optimizer=optimizer,
        scheduler_name=scheduler_name,
        total_steps=total_steps,
        warmup_steps=warmup_steps,
        min_lr_ratio=min_lr_ratio,
        one_cycle_pct_start=one_cycle_pct_start,
    )
    
    if ckpt:
        optimizer.load_state_dict(ckpt['optimizer'])
        if scheduler is not None and ckpt.get('scheduler') is not None:
            scheduler.load_state_dict(ckpt['scheduler'])
        model.load_state_dict(ckpt['model'])
        print(f"Resuming training from epoch {ckpt['epoch'] + 1}")
    model.train()
    if hasattr(optimizer, "train"):
        optimizer.train()
    
    # Compile model for faster training (1.5-2x speedup on modern GPUs)
    # Compile after all setup is complete (device, checkpoint, train mode)
    # This matches the pattern used in model.py where train() is set before compilation
    # Must compile before DataParallel wrapping (if used)
    # If compilation fails (e.g., missing Python.h, unsupported GPU architecture), continue with uncompiled model
    if multi_gpu:
        model = nn.DataParallel(model)

    last_log_dict = {
        "loss_missing": float("nan"),
        "loss_total": float("nan"),
        "mae_missing": float("nan"),
        "mae_total": float("nan"),
        "lr": optimizer.param_groups[0]["lr"],
    }
    checkpoint_metrics = {}

    try:
        for epoch in tqdm(range(ckpt['epoch'] + 1 if ckpt else 1, epochs + 1)):
            # torch.cuda.empty_cache()
            epoch_start_time = time.time()
            optimizer.zero_grad(set_to_none=True)
            
            (batch_X, batch_target, _, _, _), _ = prior.get_batch()
            
            batch_X = batch_X.to(device=device, dtype=compute_dtype)
            batch_target = batch_target.to(device=device, dtype=compute_dtype)
            
            output = model(batch_X)
            
            missing_mask = torch.isnan(batch_X)
            
            losses = criterion(output, batch_target)
            
            loss_missing = losses[missing_mask].mean()
            
            with torch.no_grad():
                loss_total = losses.mean()
                medians = bar_distribution.median(output)
                missing_mae = (medians[missing_mask] - batch_target[missing_mask]).abs().mean()
                total_mae = (medians - batch_target).abs().mean()
            
            loss_missing.backward()
            if grad_clip_norm is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
            
            optimizer.step()
            if scheduler is not None:
                scheduler.step()
            
            end_time = time.time()
            
            log_dict = {
                'loss_missing': loss_missing.item(),
                'loss_total': loss_total.item(),
                'mae_missing': missing_mae.item(),
                'mae_total': total_mae.item(),
                'lr': optimizer.param_groups[0]['lr'],
            }
            last_log_dict = log_dict
            
            # print(f"Step {epoch}: Loss: {loss_total.item()}, Missing loss: {loss_missing.item()}")

            if checkpoint_every is not None and checkpoint_every > 0 and epoch % checkpoint_every == 0:
                training_state = {
                    'epoch': epoch,
                    'model': (model.module if multi_gpu else model).state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict() if scheduler is not None else None,
                }
                torch.save(training_state, work_dir+'/checkpoint_'+str(epoch)+'.pth')
                checkpoint_metrics[str(epoch)] = dict(log_dict)

            for callback in callbacks:
                if type(criterion) is FullSupportBarDistribution:
                    callback.on_epoch_end(epoch, end_time - epoch_start_time, loss_missing.item(), None, dist=criterion, log_dict=log_dict)
                else:
                    callback.on_epoch_end(epoch, end_time - epoch_start_time, loss_missing.item(), None, log_dict=log_dict)
        

    except KeyboardInterrupt:
        pass
    finally:
        for callback in callbacks:
            callback.close()

    if return_metrics:
        metrics = dict(last_log_dict)
        if checkpoint_metrics:
            metrics["checkpoint_metrics"] = checkpoint_metrics
        return (model.module if multi_gpu else model), metrics
    return (model.module if multi_gpu else model), last_log_dict["loss_missing"]

if __name__ == "__main__":
    # High-capacity preset aligned with train.py for strong performance.
    # Note: model_new has a different block design, so total params are slightly higher.
    num_attention_heads = 32
    embedding_size = 32 * num_attention_heads  # 1024
    mlp_hidden_size = 1024
    num_cls = 12
    num_layers = 12
    epochs = 100000
    lr = 2e-4
    weight_decay = 1e-2
    grad_clip_norm = 1.0
    optimizer_name = "adamw"
    scheduler_name = "warmup_cosine"
    warmup_ratio = 0.06
    min_lr_ratio = 0.1
    
    model = TabImputeModel(
        embedding_size=embedding_size,
        num_attention_heads=num_attention_heads,
        mlp_hidden_size=mlp_hidden_size,
        num_layers=num_layers,
        num_outputs=5000,
        num_cls=num_cls,
        use_rope=True,
        rope_base=10000.0,
        rope_fraction=1.0,
        use_absolute_positional_embeddings=False,
        positional_damping_factor=0.1,
    ).to('cuda')
    
    model = model.to(torch.bfloat16)
    
    # Note: Model compilation happens in train() function after full setup
    # This ensures consistent behavior whether called from __main__ or elsewhere
    
    p_missing = 0.4
    config = {
        "num_rows_low": 10,
        "num_rows_high": 50,
        "num_cols_low": 5,
        "num_cols_high": 50,
        "p_missing": p_missing,
        "apply_feature_warping_prob": 0.0,
        "apply_quantization_prob": 0.0,
        # Latent Factor configs
        "latent_rank_low": 1,
        "latent_rank_high": 11,
        "latent_spike_p": 0.3,
        "latent_slab_sigma": 2.0,
    }

    # Example, specify one data generation type and one missingness pattern
    prior = MissingnessPrior(
        generator_type="latent_factor",
        missingness_type="mcar",
        config=config,
        batch_size=16,
        verbose=False,
        entry_wise_features=False,
    )
    
    # (X_full, y_full, d, seq_lens, train_sizes), _ = prior.get_batch()
    
    print(f"Number of parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    borders_path = importlib.resources.files('tabimpute') / 'data' / 'borders.pt'
    with importlib.resources.as_file(borders_path) as path:
        bar_distribution = FullSupportBarDistribution(borders=torch.load(path).to(torch.device('cuda')))
    
    model.train()
    
    name = f"tabimpute-new-rope_v1-large-mcar_p{p_missing}-num_cls_{num_cls}-rank_1_11"
    # NOTE: a directory with the name of the run will be created in the workdir directory
    
    # NOTE: If not resuming training, set ckpt to None
    ckpt_path = '/home/jacobf18/tabular/mcpfn/src/tabimpute/workdir/tabimpute-new-rope_v1-large-mcar_p0.4-num_cls_12-rank_1_11/checkpoint_60000.pth'
    # ckpt = torch.load(ckpt_path)
    ckpt = None
    
    if ckpt is None:
        id_name = None
    else:
        id_name = ckpt_path.split('/')[-1].split('.')[0]
    
    callbacks = [
        WandbLoggerCallback(
            project="tabimpute",
            name=name,
            # id='tabimpute-mcar_p0.4-num_cls_8-rank_1_1120260211_124242',
            id=id_name,
            config={
                "embedding_size": embedding_size,
                "num_attention_heads": num_attention_heads,
                "mlp_hidden_size": mlp_hidden_size,
                "num_layers": num_layers,
                "batch_size": 16,
                "lr": lr,
                "weight_decay": weight_decay,
                "grad_clip_norm": grad_clip_norm,
                "optimizer_name": optimizer_name,
                "scheduler_name": scheduler_name,
                "warmup_ratio": warmup_ratio,
                "min_lr_ratio": min_lr_ratio,
                "epochs": epochs,
                "num_cls": num_cls,
                "use_rope": True,
                "rope_base": 10000.0,
                "rope_fraction": 1.0,
                "use_absolute_positional_embeddings": False,
            },
            log_dir='./wandb'
        )
    ]
    
    # mse_criterion = nn.MSELoss()
    
    train(
        model,
        prior,
        bar_distribution,
        bar_distribution,
        epochs=epochs,
        lr=lr,
        weight_decay=weight_decay,
        grad_clip_norm=grad_clip_norm,
        optimizer_name=optimizer_name,
        scheduler_name=scheduler_name,
        warmup_ratio=warmup_ratio,
        min_lr_ratio=min_lr_ratio,
        device='cuda',
        callbacks=callbacks,
        run_name=name,
        ckpt=ckpt,
    )
    print("Training complete")
