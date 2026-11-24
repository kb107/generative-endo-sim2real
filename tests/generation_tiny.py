"""Tiny randomly initialised PixArt/VAE fixtures; no downloads, CPU only."""
import pytest
import torch

pytest.importorskip('diffusers')
pytest.importorskip('peft')
from diffusers import (AutoencoderKL, DPMSolverMultistepScheduler,  # noqa: E402
                       PixArtTransformer2DModel)

from generative_sim2real.generation import model as gm  # noqa: E402


def tiny_transformer(seed=0):
    torch.manual_seed(seed)
    return PixArtTransformer2DModel(
        num_attention_heads=2, attention_head_dim=8, in_channels=4, out_channels=8,
        num_layers=4, cross_attention_dim=16, caption_channels=32, sample_size=8,
        patch_size=2, attention_bias=True, activation_fn='gelu-approximate',
        num_embeds_ada_norm=1000, norm_type='ada_norm_single',
        norm_elementwise_affine=False, norm_eps=1e-6).eval()


def tiny_vae(seed=0):
    torch.manual_seed(seed)
    return AutoencoderKL(
        in_channels=3, out_channels=3, latent_channels=4, norm_num_groups=4,
        block_out_channels=(8, 8, 8, 8), layers_per_block=1, sample_size=64,
        down_block_types=('DownEncoderBlock2D',) * 4,
        up_block_types=('UpDecoderBlock2D',) * 4).eval()


def tiny_scheduler():
    return DPMSolverMultistepScheduler(beta_schedule='linear', prediction_type='epsilon')


def tiny_prompt(seed=1):
    g = torch.Generator().manual_seed(seed)
    return {'positive': torch.randn(1, 6, 32, generator=g),
            'negative': torch.randn(1, 6, 32, generator=g),
            'positive_mask': torch.ones(1, 6, dtype=torch.long),
            'negative_mask': torch.ones(1, 6, dtype=torch.long)}


def tiny_model(controller_layers=2):
    transformer = tiny_transformer()
    return gm.assemble(transformer, controller_layers=controller_layers, rank=2, alpha=2)


def batch(n=2, seed=3):
    g = torch.Generator().manual_seed(seed)
    return (torch.randn(n, 4, 8, 8, generator=g), torch.randn(n, 4, 8, 8, generator=g))
