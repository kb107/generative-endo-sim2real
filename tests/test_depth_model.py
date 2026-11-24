import pytest
import torch

from da3_fakes import FOCAL, fake_model, intrinsics
from generative_sim2real.depth.model import DA3MetricDepth


def batch_K(n=2, fx=FOCAL, fy=FOCAL, size=504):
    K = intrinsics(size).repeat(n, 1, 1)
    K[:, 0, 0], K[:, 1, 1] = fx, fy
    return K


def test_focal_conversion_uses_mean_focal_and_roundtrips():
    model = fake_model()
    K = batch_K(2)
    K[1, 0, 0], K[1, 1, 1] = 300., 900.      # mean 600 -> factor 2
    depth = torch.full((2, 1, 3, 3), 0.5)
    metric = model.canonical_to_metric(depth, K)
    assert torch.allclose(metric, torch.full_like(metric, 1.0))
    K[0, 0, 0] = K[0, 1, 1] = 150.           # factor 0.5
    assert torch.allclose(model.canonical_to_metric(depth, K)[0], torch.full((1, 3, 3), .25))
    assert torch.allclose(model.metric_to_canonical(model.canonical_to_metric(depth, K), K), depth, atol=1e-7)


@pytest.mark.parametrize('bad', ['nonpositive', 'nan', 'shape'])
def test_invalid_intrinsics_rejected(bad):
    model = fake_model()
    K = batch_K(2)
    if bad == 'nonpositive':
        K[0, 1, 1] = 0.
    elif bad == 'nan':
        K[1, 0, 2] = float('nan')
    else:
        K = K[:1]
    with pytest.raises(ValueError):
        model.canonical_to_metric(torch.ones(2, 1, 4, 4), K)


def test_resize_scales_intrinsics_before_conversion():
    model = fake_model()
    rgb = torch.rand(1, 3, 252, 504)
    K = intrinsics(504)[None].clone()
    K[0, 1, 1], K[0, 1, 2] = 600., 126.
    out = model(rgb, K)
    assert out['depth_m'].shape == (1, 1, 504, 504)
    assert out['K_processed'][0, 0, 0] == 600. and out['K_processed'][0, 1, 1] == 1200.
    assert out['K_processed'][0, 1, 2] == 252.
    assert torch.allclose(out['depth_m'], out['canonical_depth'] * (900. / 300.))


def test_imagenet_normalization_and_singleton_view_contract():
    model = fake_model()
    rgb = torch.rand(2, 3, 504, 504)
    model(rgb, batch_K(2))
    sent, kwargs = model.network.calls[-1]
    mean = torch.tensor([.485, .456, .406]).view(1, 1, 3, 1, 1)
    std = torch.tensor([.229, .224, .225]).view(1, 1, 3, 1, 1)
    assert sent.shape == (2, 1, 3, 504, 504)
    assert torch.allclose(sent, (rgb[:, None] - mean) / std, atol=1e-6)
    assert kwargs['extrinsics'] is None and kwargs['intrinsics'] is None and kwargs['infer_gs'] is False
    assert kwargs['use_ray_pose'] is False and kwargs['export_feat_layers'] == []


@pytest.mark.parametrize('bad', [lambda r: r * 2, lambda r: r - 1, lambda r: r.mul(float('nan')),
                                 lambda r: r[:, :2], lambda r: (r * 255).byte()])
def test_rgb_contract_rejected(bad):
    with pytest.raises((ValueError, RuntimeError)):
        fake_model().prepare_inputs(bad(torch.rand(1, 3, 504, 504)), batch_K(1))


def test_only_head_receives_gradients_and_sky_is_disabled():
    model = fake_model()
    assert model.network.head.use_sky_head is False
    params = model.configure_head_training()
    head = list(model.network.head.parameters())
    sky = list(model.network.head.scratch.sky_output_conv2.parameters())
    assert {id(p) for p in params} == {id(p) for p in head} - {id(p) for p in sky}
    assert not any(p.requires_grad for p in model.network.backbone.parameters())
    assert not any(p.requires_grad for p in sky)
    model.train()
    assert not model.network.backbone.training and model.network.head.training
    out = model(torch.rand(2, 3, 504, 504), batch_K(2))
    out['canonical_depth'].mean().backward()
    assert all(p.grad is None for p in model.network.backbone.parameters())
    assert all(p.grad is None for p in sky)
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in params)


def test_head_state_roundtrip_strict_and_sky_stays_disabled():
    model, other = fake_model(0), fake_model(1)
    other.configure_head_training()
    other.network.head.use_sky_head = True       # a stale flag must be corrected on load
    state = {k: v.clone() for k, v in model.head_state_dict().items()}
    other.load_head_state_dict(state)
    assert other.network.head.use_sky_head is False
    assert all(torch.equal(a, b) for a, b in zip(state.values(), other.head_state_dict().values()))
    assert not any(p.requires_grad for p in other.network.head.scratch.sky_output_conv2.parameters())
    del state['scratch.depth_conv.bias']
    with pytest.raises(RuntimeError):
        other.load_head_state_dict(state)


def test_process_size_must_be_patch_multiple():
    with pytest.raises(ValueError):
        DA3MetricDepth(fake_model().network, (500, 504))
