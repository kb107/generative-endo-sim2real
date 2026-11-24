"""Opt-in checks of the pinned official DA3 checkpoint (needs the DA3 venv and cached weights).

Run: DA3_REAL_MODEL=1 PYTHONPATH=src external/da3-venv/bin/python -m pytest tests/test_depth_model_real.py
"""
import os
from pathlib import Path
import subprocess

import pytest
import torch

pytest.importorskip('depth_anything_3')
pytestmark = pytest.mark.skipif(os.environ.get('DA3_REAL_MODEL') != '1', reason='set DA3_REAL_MODEL=1 to load the real model')

from generative_sim2real.depth import model as M


def test_pinned_checkout_is_clean_and_at_the_recorded_revision():
    M.verify_upstream_checkout()
    assert M.DA3_REVISION == '3d835ec1a5802d64a8b8b15f817a1ab54809bfe4'


def test_real_model_contract_head_only_sky_off_and_focal_scaling():
    model = M.DA3MetricDepth.from_pretrained()
    params = model.configure_head_training()
    assert sum(p.numel() for p in params) == 29_766_849
    assert not model.network.head.use_sky_head
    assert not any(p.requires_grad for p in model.network.backbone.parameters())
    assert not any('sky' in n for n, p in model.named_parameters() if p.requires_grad)
    model.eval()
    rgb = torch.rand(1, 3, 504, 504)
    K300 = torch.tensor([[[300., 0, 252], [0, 300, 252], [0, 0, 1]]])
    K600 = K300.clone()
    K600[:, :2] *= 2
    with torch.no_grad():
        a, b = model(rgb, K300), model(rgb, K600)
    assert a['depth_m'].shape == (1, 1, 504, 504) and torch.isfinite(a['depth_m']).all() and (a['depth_m'] > 0).all()
    assert torch.allclose(b['depth_m'], 2 * a['depth_m'], rtol=1e-5)       # the network does not see K
