# Masked Diffusion Transformer V2

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/masked-diffusion-transformer-is-a-strong/image-generation-on-imagenet-256x256)](https://paperswithcode.com/sota/image-generation-on-imagenet-256x256?p=masked-diffusion-transformer-is-a-strong)
[![HuggingFace space](https://img.shields.io/badge/🤗-HuggingFace%20Space-cyan.svg)](https://huggingface.co/spaces/shgao/MDT)

The official codebase for [Masked Diffusion Transformer is a Strong Image Synthesizer](https://arxiv.org/abs/2303.14389).

## MDTv2: Faster Convergeence & Stronger performance
**MDTv2 achieves superior image synthesis performance, e.g., a new SOTA FID score of 1.58 on the ImageNet dataset, and has more than 10× faster learning speed than the previous SOTA DiT.**

MDTv2 demonstrates a 5x acceleration compared to the original MDT.

[MDTv1 code](https://github.com/sail-sg/MDT/tree/mdtv1)
## Introduction

Despite its success in image synthesis, we observe that diffusion probabilistic models (DPMs) often lack contextual reasoning ability to learn the relations among object parts in an image, leading to a slow learning process. To solve this issue, we propose a Masked Diffusion Transformer (MDT) that introduces a mask latent modeling scheme to explicitly enhance the DPMs’ ability to contextual relation learning among object semantic parts in an image. 

During training, MDT operates in the latent space to mask certain tokens. Then, an asymmetric diffusion transformer is designed to predict masked tokens from unmasked ones while maintaining the diffusion generation process. Our MDT can reconstruct the full information of an image from its incomplete contextual input, thus enabling it to learn the associated relations among image tokens. We further improve MDT with a more efficient macro network structure and training strategy, named MDTv2. 

Experimental results show that MDTv2 achieves superior image synthesis performance, e.g., **a new SOTA FID score of 1.58 on the ImageNet dataset, and has more than 10× faster learning speed than the previous SOTA DiT**. 

<img width="800" alt="image" src="figures/vis.jpg">

# Performance

| Model| Dataset |  Resolution | FID-50K | Inception Score |
|---------|----------|-----------|---------|--------|
|MDT-XL/2 | ImageNet | 256x256   | 1.79    | 283.01|
|MDTv2-XL/2 | ImageNet | 256x256 | 1.58    | 314.73|

[Pretrained model download](https://huggingface.co/shgao/MDT-XL2/tree/main)

Model is hosted on hugglingface, you can also download it with:
```
from huggingface_hub import snapshot_download
models_path = snapshot_download("shgao/MDT-XL2")
ckpt_model_path = os.path.join(models_path, "mdt_xl2_v1_ckpt.pt")
```
A hugglingface demo is on [DEMO](https://huggingface.co/spaces/shgao/MDT).

**NEW SOTA on FID.**
# Setup

Prepare the Pytorch >=2.0 version. Download and install this repo.

```
git clone https://github.com/sail-sg/MDT
cd MDT
pip install -e .
```
Install [PyTorch Lightning](https://lightning.ai/docs/pytorch/latest/) and the Stable Diffusion VAE dependency used for encoding/decoding latents:

```
pip install pytorch-lightning diffusers
```
Install [Adan optimizer](https://github.com/sail-sg/Adan), Adan is a strong optimizer with faster convergence speed than AdamW. [(paper)](https://arxiv.org/abs/2208.06677)
```
python -m pip install git+https://github.com/sail-sg/Adan.git
```

**DATA** 
- For standard datasets like ImageNet and CIFAR, please refer to '[dataset](https://github.com/sail-sg/MDT/tree/main/datasets)' for preparation.
- When using customized dataset, change the image file name to `ClassID_ImgID.jpg`,
as the [ADM's dataloder](https://github.com/openai/guided-diffusion) gets the class ID from the file name. 

# Training

## PyTorch Lightning workflow

We now provide an end-to-end LightningModule (`masked_diffusion.lightning_module.MDTLightningModule`) that encapsulates the original training utilities. The entrypoints are located in `scripts/lightning_train.py` and `scripts/lightning_sample.py`.

### Training

```bash
export OUTPUT_DIR=output_mdtv2_s2
DATA_PATH=/dataset/imagenet

MODEL_FLAGS="--image_size 256 --mask_ratio 0.30 --decode_layer 6 --model MDTv2_S_2"
DIFFUSION_FLAGS="--diffusion_steps 1000"
TRAIN_FLAGS="--batch_size 32 --lr 5e-4"

python scripts/lightning_train.py \
  --data_dir ${DATA_PATH} \
  --devices 8 \
  --accelerator gpu \
  --strategy ddp \
  --output_dir ${OUTPUT_DIR} \
  $MODEL_FLAGS $DIFFUSION_FLAGS $TRAIN_FLAGS
```

`lightning_train.py` exposes the same modelling hyper-parameters as the legacy scripts while adding convenience options such as automatic checkpointing (`--checkpoint_every_n_steps`), gradient accumulation (`--accumulate_grad_batches`), and learning-rate monitoring (`--monitor_lr`).

### Sampling

Use the Lightning checkpoint produced during training to generate images:

```bash
python scripts/lightning_sample.py \
  --checkpoint_path ${OUTPUT_DIR}/mdt-0005000.ckpt \
  --num_samples 64 \
  --batch_size 16 \
  --cfg_cond True \
  --output_dir ${OUTPUT_DIR}/samples
```

The sampler handles classifier-free guidance, DDIM/ancestral sampling, and will automatically decode latents with the Stable Diffusion VAE. Set `--use_cpu` to run on CPU.

# Evaluation

The evaluation code is obtained from [ADM's TensorFlow evaluation suite](https://github.com/openai/guided-diffusion/tree/main/evaluations).
Please follow the instructions in the `evaluations` folder to set up the evaluation environment.

<details>
  <summary>Sampling and Evaluation (`run_sample.sh`): </summary>

```shell
CHECKPOINT=output_mdtv2_xl2/mdt-0002500.ckpt
OUTPUT_DIR=output_mdtv2_xl2_eval

python scripts/lightning_sample.py \
  --checkpoint_path ${CHECKPOINT} \
  --num_samples 50000 \
  --batch_size 256 \
  --cfg_cond True \
  --output_dir ${OUTPUT_DIR}

python evaluations/evaluator.py ../dataeval/VIRTUAL_imagenet256_labeled.npz ${OUTPUT_DIR}/samples_50000x256x256x3.npz
```

</details>

# Visualization

Run the `infer_mdt.py` script to generate images. It now understands both Lightning checkpoints and legacy `.pt` model weights.

# Citation

```
@misc{gao2023masked,
      title={Masked Diffusion Transformer is a Strong Image Synthesizer}, 
      author={Shanghua Gao and Pan Zhou and Ming-Ming Cheng and Shuicheng Yan},
      year={2023},
      eprint={2303.14389},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```

# Acknowledgement

This codebase is built based on the [DiT](https://github.com/facebookresearch/dit) and [ADM](https://github.com/openai/guided-diffusion). Thanks!
