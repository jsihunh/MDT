# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
from torchvision.utils import save_image

from masked_diffusion import create_diffusion, diffusion_defaults
from masked_diffusion.lightning_module import MDTLightningModule


# Setup PyTorch:
torch.manual_seed(1)
torch.set_grad_enabled(False)
device = "cuda" if torch.cuda.is_available() else "cpu"
num_sampling_steps = 250
cfg_scale = 4.0
pow_scale = 0.01 # large pow_scale increase the diversity, small pow_scale increase the quality.
model_path = 'mdt_xl2_v2_ckpt.pt'

# Load model:
image_size = 256
assert image_size in [256], "We provide pre-trained models for 256x256 resolutions for now."
latent_size = image_size // 8
try:
    module = MDTLightningModule.from_pretrained(
        model_path,
        strict=False,
        map_location=device,
    )
except Exception:
    diff_config = diffusion_defaults()
    diff_config.update(dict(diffusion_steps=num_sampling_steps, timestep_respacing=str(num_sampling_steps)))
    module = MDTLightningModule(
        model_name="MDTv2_XL_2",
        image_size=image_size,
        mask_ratio=None,
        decode_layer=4,
        diffusion_config=diff_config,
        class_cond=True,
    )
    state_dict = torch.load(model_path, map_location=lambda storage, loc: storage)
    module.model.load_state_dict(state_dict)

model = module.model.to(device)
model.eval()
diffusion = create_diffusion(str(num_sampling_steps))
module.to(device)
module.instantiate_first_stage()

# Labels to condition the model with:
class_labels = [19,23,106,108,278,282]

# Create sampling noise:
n = len(class_labels)
z = torch.randn(n, 4, latent_size, latent_size, device=device)
y = torch.tensor(class_labels, device=device)

# Setup classifier-free guidance:
z = torch.cat([z, z], 0)
y_null = torch.tensor([1000] * n, device=device)
y = torch.cat([y, y_null], 0)


model_kwargs = dict(y=y, cfg_scale=cfg_scale, scale_pow=pow_scale)

# Sample images:
samples = diffusion.p_sample_loop(
    model.forward_with_cfg, z.shape, z, clip_denoised=False, model_kwargs=model_kwargs, progress=True, device=device
)
samples, _ = samples.chunk(2, dim=0)  # Remove null class samples
samples = module.decode_first_stage(samples)

# Save and display images:
save_image(samples, "sample.jpg", nrow=3, normalize=True, value_range=(-1, 1))
