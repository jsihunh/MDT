pip3 install torch==2.0 torchvision torchaudio
pip install -e .
python -m pip install git+https://github.com/sail-sg/Adan.git

OUTPUT_DIR=output_mdtv2_s2
MODEL_FLAGS="--image_size 256 --mask_ratio 0.30 --decode_layer 6 --model MDTv2_S_2"
DIFFUSION_FLAGS="--diffusion_steps 1000"
TRAIN_FLAGS="--batch_size 32 --lr 5e-4"
DATA_PATH=/dataset/imagenet-raw/train

python scripts/lightning_train.py \
  --data_dir ${DATA_PATH} \
  --devices 8 \
  --accelerator gpu \
  --strategy ddp \
  --output_dir ${OUTPUT_DIR} \
  $MODEL_FLAGS $DIFFUSION_FLAGS $TRAIN_FLAGS
