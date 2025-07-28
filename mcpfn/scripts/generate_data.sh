# ------------------------------------------------------
# Save prior datasets to disk and load them for training
# ------------------------------------------------------

# Saving to disk
python3 /root/tabular/mcpfn/src/mcpfn/prior/genload.py \
    --save_dir /mnt/volume_nyc2_1750872154988/data/small_20_10_nonlinear_factor_mar \
    --np_seed 42 \
    --torch_seed 42 \
    --num_batches 10 \
    --resume_from 0 \
    --batch_size 10000 \
    --batch_size_per_gp 10000 \
    --prior_type missing \
    --min_features 5 \
    --max_features 5 \
    --max_classes 10 \
    --max_seq_len 10 \
    --min_train_size 0.1 \
    --max_train_size 0.9 \
    --n_jobs -1 \
    --num_threads_per_generate 1 \
    --device cpu \
    --num_missing 10