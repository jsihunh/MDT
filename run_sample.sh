
pip3 install torch==2.0 torchvision torchaudio
pip install -e .

CHECKPOINT=output_mdt_xl2/mdt-0002500.ckpt
OUTPUT_DIR=output_mdt_xl2_eval

python scripts/lightning_sample.py \
  --checkpoint_path ${CHECKPOINT} \
  --num_samples 50000 \
  --batch_size 256 \
  --cfg_cond True \
  --output_dir ${OUTPUT_DIR}

python evaluations/evaluator.py ../dataeval/VIRTUAL_imagenet256_labeled.npz ${OUTPUT_DIR}/samples_50000x256x256x3.npz

