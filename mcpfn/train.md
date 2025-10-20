# Training the model

Make sure to always pull the newest version of the code before running the script. The training script is located in `mcpfn/scripts/train_stage1.sh`. Read this entire markdown file before running the script.

## Running the script

There are several parameters that need to be set in the script before running.

- `--wandb_log`: whether to log to wandb
- `--wandb_project`: the name of the wandb project
- `--wandb_name`: the name of the wandb run
- `--wandb_dir`: the directory to save the wandb logs
- `--batch_size`: the size of a single batch. For the generated data, this is usually 10000.
- `--micro_batch_size`: This should be set to something divisible by the batch size that doesn't exceed the GPU memory. Usually set to 100.
- `--lr`: the learning rate. Usually set to 1e-3.
- `--prior_dir`: the directory of the data to load. See the next section for how the data should be structured.
- `--checkpoint_dir`: the directory to save the checkpoints. This should be on a volume, not on the droplet.
- `--epochs`: the number of epochs to train for. Usually set to 100.
- `--save_every`: the number of epochs to save the model. Usually set to 15.

```bash
sh mcpfn/scripts/train_stage1.sh
```

## Data structure

The data should be structured as follows: There should be 10 batches, split across train and validation folders. Each batch should be a folder with the following structure:

```bash
train/
├── metadata.json
├── batch_000000.pt
├── batch_000001.pt
├── ...
├── batch_000007.pt
val/
├── metadata.json
├── batch_000000.pt
├── batch_000001.pt
```
