# ------------------------------------------------------
# Save prior datasets to disk and load them for training
# ------------------------------------------------------

# First parameter is the type of missingness. Example: mcar, mar, mnar
# Second parameter is the type of generator. Example: linear_factor, nonlinear_factor, scm
# Third parameter is the number of rows
# Fourth parameter is the number of columns

echo "Generating data for ${1} ${2} ${3} ${4}"

# Set the save directory as an environment variable
BASE_DIR="/mnt/volume_nyc2_1750872154987/data"
SAVE_DIR="${BASE_DIR}/${1}_${2}_${3}_${4}"

mkdir -p $SAVE_DIR

# Saving to disk
python3 /root/tabular/mcpfn/src/mcpfn/prior/genload.py \
    --save_dir $SAVE_DIR \
    --np_seed 42 \
    --torch_seed 42 \
    --num_batches 10 \
    --resume_from 0 \
    --batch_size 5 \
    --batch_size_per_gp 5 \
    --prior_type missing \
    --min_features $4 \
    --max_features $4 \
    --max_classes 10 \
    --min_seq_len $3 \
    --max_seq_len $3 \
    --min_train_size 0.1 \
    --max_train_size 0.9 \
    --n_jobs -1 \
    --num_threads_per_generate 1 \
    --device cpu \
    --num_missing 10 \
    --missingness_type $1 \
    --missingness_generator_type $2

# Create the val and train directories
mkdir $SAVE_DIR/val
mkdir $SAVE_DIR/train

# Copy the metadata file to the val and train directories
cp $SAVE_DIR/metadata.json $SAVE_DIR/val/
cp $SAVE_DIR/metadata.json $SAVE_DIR/train/

# Move the files to the val and train directories
mv $SAVE_DIR/batch_000008.pt $SAVE_DIR/val/
mv $SAVE_DIR/batch_000009.pt $SAVE_DIR/val/

mv $SAVE_DIR/batch_000000.pt $SAVE_DIR/train/
mv $SAVE_DIR/batch_000001.pt $SAVE_DIR/train/
mv $SAVE_DIR/batch_000002.pt $SAVE_DIR/train/
mv $SAVE_DIR/batch_000003.pt $SAVE_DIR/train/
mv $SAVE_DIR/batch_000004.pt $SAVE_DIR/train/
mv $SAVE_DIR/batch_000005.pt $SAVE_DIR/train/
mv $SAVE_DIR/batch_000006.pt $SAVE_DIR/train/
mv $SAVE_DIR/batch_000007.pt $SAVE_DIR/train/

# Rename the files
mv $SAVE_DIR/val/batch_000008.pt $SAVE_DIR/val/batch_000000.pt
mv $SAVE_DIR/val/batch_000009.pt $SAVE_DIR/val/batch_000001.pt