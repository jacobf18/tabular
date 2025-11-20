# Loading from disk and training
# torchrun --standalone --nproc_per_node=1 /Users/jfeit/tabular/mcpfn/src/mcpfn/train/run.py \

echo "Training model"

# Set the save directory as an environment variable
CHECKPOINT_NAME="masters/mcar_nonlinear"
BASE_DIR="/home/jacobf18/mcpfn_data/checkpoints"
CHECKPOINT_DIR="${BASE_DIR}/${CHECKPOINT_NAME}"

IF_SAVE=True
if [ "$IF_SAVE" = True ]; then
    mkdir -p ${CHECKPOINT_DIR}
    # Create a unique id for the checkpoint in a wand_id.txt file
    RANDOM_ID=$(cat /dev/random | tr -dc '[:alnum:]' | head -c 10)
    WAND_ID=wand$(date +%s)${RANDOM_ID}
    # WAND_ID="wand1762806111VIuvx2DgrD"
    echo ${WAND_ID} > ${CHECKPOINT_DIR}/wand_id.txt
fi
# python3 -m torch.distributed.run --nproc_per_node=1 /root/tabular/mcpfn/src/mcpfn/train/run.py \
python /home/jacobf18/tabular/mcpfn/src/tabimpute/train/run.py \
            --wandb_log ${IF_SAVE} \
            --wandb_project MCPFN \
            --wandb_name ${CHECKPOINT_NAME} \
            --wandb_dir /home/jacobf18/tabular/mcpfn/wandb \
            --wandb_mode online \
            --device cuda \
            --dtype float32 \
            --np_seed 42 \
            --torch_seed 42 \
            --max_steps 100000 \
            --start_step 0 \
            --batch_size 16 \
            --micro_batch_size 16 \
            --lr ${1} \
            --scheduler cosine_warmup \
            --warmup_proportion 0.02 \
            --gradient_clipping 1.0 \
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
            --save_temp_every 500 \
            --save_perm_every 5000 \
            --encoder_path /home/jacobf18/tabular/mcpfn/src/tabimpute/data/encoder.pth \
            --borders_path /home/jacobf18/tabular/mcpfn/src/tabimpute/data/borders.pt \
            --tabpfn_path /home/jacobf18/tabular/mcpfn/src/tabimpute/model/tabpfn_model.pt \
            --model_name ${1}.ckpt \
            --save_every 15 \
            --min_seq_len 5 \
            --max_seq_len 30 \
            --min_features 5 \
            --max_features 30 \
            --missingness_type mcar \
            --generator_type nonlinear_factor \
            --update_probs_every 50
            # --checkpoint_path /home/jacobf18/mcpfn_data/checkpoints/${CHECKPOINT_NAME}/step-81500.ckpt
            # --prior_dir ${PRIOR_DIR} \
            
