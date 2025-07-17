from huggingface_hub import hf_hub_download
import torch
import os

vae_path = hf_hub_download(
    repo_id="hustvl/PixelHacker",
    filename="vae/diffusion_pytorch_model.bin",
    force_download=True,
    resume_download=True
)

assert os.path.isfile(vae_path), "下载失败或文件丢失"
