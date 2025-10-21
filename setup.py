from setuptools import setup

setup(
    name="masked-diffusion",
    py_modules=["masked_diffusion"],
    install_requires=[
        "blobfile>=1.0.5",
        "torch",
        "tqdm",
        "pytorch-lightning>=2.0",
        "diffusers>=0.17.0",
    ],
)
