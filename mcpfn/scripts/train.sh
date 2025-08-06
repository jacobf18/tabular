# Loading from disk and training
# torchrun --standalone --nproc_per_node=1 /Users/jfeit/tabular/mcpfn/src/mcpfn/train/run.py \

echo "Training model"

# Set the save directory as an environment variable
BASE_DIR="/mnt/mcpfn_data"
PRIOR_DIR="${BASE_DIR}/data/"
CHECKPOINT_DIR="${BASE_DIR}/checkpoints/all_data/"

# mkdir -p ${CHECKPOINT_DIR}
# # Create a unique id for the checkpoint in a wand_id.txt file
# WAND_ID=wand$(date +%s)
# echo ${WAND_ID} > ${CHECKPOINT_DIR}/wand_id.txt

python3 /root/tabular/mcpfn/src/mcpfn/train/run.py \
            --wandb_log False \
            --wandb_project MCPFN \
            --wandb_name ${1} \
            --wandb_dir /root/tabular/mcpfn/wandb \
            --wandb_mode online \
            --device cuda \
            --dtype float32 \
            --np_seed 42 \
            --torch_seed 42 \
            --max_steps 8 \
            --batch_size 10000 \
            --micro_batch_size 100 \
            --lr ${1} \
            --scheduler cosine_warmup \
            --warmup_proportion 0.02 \
            --gradient_clipping 1.0 \
            --prior_dir ${PRIOR_DIR} \
            --load_prior_start 0 \
            --delete_after_load False \
            --prior_device cpu \
            --embed_dim 128 \
            --col_num_blocks 3 \
            --col_nhead 4 \
            --col_num_inds 128 \
            --row_num_blocks 3 \
            --row_nhead 8 \
            --row_num_cls 4 \
            --row_rope_base 100000 \
            --icl_num_blocks 12 \
            --icl_nhead 4 \
            --ff_factor 2 \
            --norm_first True \
            --checkpoint_dir ${CHECKPOINT_DIR} \
            --save_temp_every 50 \
            --save_perm_every 5000 \
            --epochs 100 \
            --encoder_path /root/tabular/mcpfn/src/mcpfn/model/encoder.pth \
            --borders_path /root/tabular/mcpfn/borders.pt \
            --model_name ${1}.ckpt \
            --save_every 15
            # --checkpoint_path ${CHECKPOINT_DIR}/epoch_100_${1}.ckpt