import torch
from torch import nn
from tqdm import tqdm
import time
from typing import Dict
from tabimpute.model.bar_distribution import FullSupportBarDistribution
import schedulefree
import os
from pathlib import Path
import importlib.resources

from tabimpute.model.model import TabImputeModel
from tabimpute.prior.training_set_generation import MissingnessPrior
from tabimpute.train.callbacks import Callback, WandbLoggerCallback

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

def train(model: TabImputeModel, 
          prior: MissingnessPrior, 
          bar_distribution: FullSupportBarDistribution,
          criterion: nn.Module,
          epochs: int, 
          lr: float = 1e-4, 
          device: torch.device = None,
          callbacks: list[Callback] = None, 
          ckpt: Dict[str, torch.Tensor] = None, 
          multi_gpu: bool = False,
          run_name: str = 'arbpfn'):
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
    if multi_gpu:
        model = nn.DataParallel(model)
    if callbacks is None:
        callbacks = []
    if not device:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)
    optimizer = schedulefree.AdamWScheduleFree(model.parameters(), lr=lr, weight_decay=0.0)
    
    if ckpt:
        optimizer.load_state_dict(ckpt['optimizer'])
        model.load_state_dict(ckpt['model'])
        
    model.train()
    optimizer.train()

    try:
        for epoch in tqdm(range(ckpt['epoch'] + 1 if ckpt else 1, epochs + 1)):
            epoch_start_time = time.time()
            
            (batch_X, batch_target, _, _, _), _ = prior.get_batch()
            
            batch_X = batch_X.to(torch.bfloat16)
            batch_target = batch_target.to(torch.bfloat16)
            
            batch_X = batch_X.to(device)
            batch_target = batch_target.to(device)
            
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
            
            optimizer.step()
            optimizer.zero_grad()
            
            end_time = time.time()
            
            log_dict = {
                'loss_missing': loss_missing.item(),
                'loss_total': loss_total.item(),
                'mae_missing': missing_mae.item(),
                'mae_total': total_mae.item()
            }
            
            # print(f"Step {epoch}: Loss: {loss_total.item()}, Missing loss: {loss_missing.item()}")

            training_state = {
                'epoch': epoch,
                'model': (model.module if multi_gpu else model).state_dict(),
                'optimizer': optimizer.state_dict()
            }
            if epoch % 1000 == 0:
                torch.save(training_state, work_dir+'/checkpoint_'+str(epoch)+'.pth')

            for callback in callbacks:
                if type(criterion) is FullSupportBarDistribution:
                    callback.on_epoch_end(epoch, end_time - epoch_start_time, loss_missing.item(), (model.module if multi_gpu else model), dist=criterion, log_dict=log_dict)
                else:
                    callback.on_epoch_end(epoch, end_time - epoch_start_time, loss_missing.item(), (model.module if multi_gpu else model), log_dict=log_dict)

    except KeyboardInterrupt:
        pass
    finally:
        for callback in callbacks:
            callback.close()

    return (model.module if multi_gpu else model), loss_missing.item()

if __name__ == "__main__":
    num_attention_heads = 32
    embedding_size = 32 * num_attention_heads
    mlp_hidden_size = 1024
    num_cls = 8
    num_layers = 12
    epochs = 40000
    lr = 2e-4
    
    model = TabImputeModel(
        embedding_size=embedding_size,
        num_attention_heads=num_attention_heads,
        mlp_hidden_size=mlp_hidden_size,
        num_layers=num_layers,
        num_outputs=5000,
        num_cls=num_cls,
    ).to('cuda')
    
    model = model.to(torch.bfloat16)
    
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
        "latent_rank_high": 15,
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
    
    name = f"tabimpute-large-pancake-model-mcar_mnar-p{p_missing}-num-cls-{num_cls}-rank-1-15"
    callbacks = [
        WandbLoggerCallback(
            project="tabimpute",
            name=name,
            # id='tabimpute-large-pancake-model-mcar-p0.4-num-cls-8-rank-1-1120260104_194958',
            config={
                "embedding_size": embedding_size,
                "num_attention_heads": num_attention_heads,
                "mlp_hidden_size": mlp_hidden_size,
                "num_layers": num_layers,
                "batch_size": 16,
                "lr": lr,
                "epochs": epochs,
                "num_cls": num_cls,
            },
            log_dir='./wandb'
        )
    ]
    
    mse_criterion = nn.MSELoss()
    
    # ckpt_path = '/home/jacobf18/tabular/mcpfn/src/tabimpute/workdir/tabimpute-large-pancake-model-mcar-p0.4-num-cls-8-rank-1-11/checkpoint_50000.pth'
    # ckpt = torch.load(ckpt_path)
    
    train(model, prior, bar_distribution, bar_distribution, epochs=epochs, lr=lr, device='cuda', callbacks=callbacks, run_name=name, ckpt=None)
    print("Training complete")