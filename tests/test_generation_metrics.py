"""Fixture tests for generation_metrics: tiny synthetic images, no model download.

The optional DINOv2 test runs only with GM_REAL_MODEL=1 and the model already in the HF cache."""
import json
import os
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from generative_sim2real.generation import p4_report
from generative_sim2real.generation_metrics import compare as cmp
from generative_sim2real.generation_metrics import distance as dist
from generative_sim2real.generation_metrics import features, images as im, pixel


class FakeExtractor:
    """Deterministic stand-in for DINOv2: 8x8 colour grid of the image (192-d)."""
    name = 'fake:grid8'
    revision = None

    def __call__(self, batch):
        return torch.nn.functional.adaptive_avg_pool2d(batch.float(), 8).flatten(1) * 10


def write_set(root, count, *, tint, seed, size=64):
    rng = np.random.default_rng(seed)
    root.mkdir(parents=True, exist_ok=True)
    for i in range(count):
        base = rng.uniform(0.1, 0.6, size=(size, size, 3)) * np.asarray(tint)
        base += rng.normal(0, 0.03, size=base.shape)
        Image.fromarray((base.clip(0, 1) * 255).astype(np.uint8)).save(root / f'{i:03d}.png')
    return root


# ---------------------------------------------------------------- distances ----
def test_frechet_matches_scipy_and_is_zero_for_identical_sets():
    from scipy.linalg import sqrtm

    g = torch.Generator().manual_seed(0)
    x, y = torch.randn(300, 6, generator=g), torch.randn(250, 6, generator=g) * 1.5 + 0.7
    assert dist.frechet_distance(x, x.clone()) == pytest.approx(0, abs=1e-8)
    s1, s2 = np.cov(x.double().numpy().T), np.cov(y.double().numpy().T)
    reference = (np.sum((x.double().mean(0).numpy() - y.double().mean(0).numpy()) ** 2)
                 + np.trace(s1 + s2 - 2 * sqrtm(s1 @ s2).real))
    assert dist.frechet_distance(x, y) == pytest.approx(reference, rel=1e-6)


def test_frechet_rank_deficient_covariance_is_finite_and_symmetric():
    g = torch.Generator().manual_seed(1)
    x, y = torch.randn(10, 40, generator=g), torch.randn(12, 40, generator=g)  # n < d
    a, b = dist.frechet_distance(x, y), dist.frechet_distance(y, x)
    assert np.isfinite(a) and a == pytest.approx(b, rel=1e-6)


def test_mmd_is_unbiased_for_equal_distributions_and_positive_for_shift():
    g = torch.Generator().manual_seed(2)
    same = [dist.mmd2_unbiased(torch.randn(60, 8, generator=g), torch.randn(60, 8, generator=g))
            for _ in range(40)]
    assert abs(np.mean(same)) < 0.02          # unbiased: no positive offset (negative values allowed)
    assert min(same) < 0
    shifted = dist.mmd2_unbiased(torch.randn(60, 8, generator=g), torch.randn(60, 8, generator=g) + 1)
    assert shifted > 0.1


def test_mmd_matches_naive_double_loop():
    g = torch.Generator().manual_seed(3)
    x, y = torch.randn(7, 5, generator=g).double(), torch.randn(9, 5, generator=g).double()
    k = lambda a, b: ((a @ b) / 5 + 1) ** 3
    xx = sum(k(x[i], x[j]) for i in range(7) for j in range(7) if i != j) / (7 * 6)
    yy = sum(k(y[i], y[j]) for i in range(9) for j in range(9) if i != j) / (9 * 8)
    xy = sum(k(a, b) for a in x for b in y) / (7 * 9)
    assert dist.mmd2_unbiased(x, y) == pytest.approx((xx + yy - 2 * xy).item(), rel=1e-9)
    result = dist.kid(x, y, subsets=4, subset_size=100)
    assert result['subset_size'] == 7 and result['full'] == pytest.approx(dist.mmd2_unbiased(x, y))


def test_pairwise_l1_and_w1_match_the_p4_definitions():
    g = torch.Generator().manual_seed(4)
    batch = torch.rand(9, 3, 16, 16, generator=g)
    other = (torch.rand(7, 3, 16, 16, generator=g) * 0.5 + 0.2)
    flat = torch.nn.functional.adaptive_avg_pool2d(batch, 4).flatten(1)
    assert dist.pairwise_l1(flat) == pytest.approx(p4_report.pairwise_diversity(batch, size=4))
    w1 = dist.channel_w1(batch, other)
    assert w1['mean'] == pytest.approx(p4_report.rgb_w1(batch, other))
    assert dist.channel_w1(batch, batch)['mean'] == pytest.approx(0, abs=1e-12)
    shifted = dist.channel_w1(torch.zeros(2, 3, 4, 4), torch.full((2, 3, 4, 4), 0.25))
    assert shifted == pytest.approx({'R': .25, 'G': .25, 'B': .25, 'mean': .25})


# -------------------------------------------------------------------- pixels ----
def test_hsv_saturation_known_colours_and_descriptors():
    colours = torch.tensor([[1., 0, 0], [0.5, 0.5, 0.5], [0, 0, 0], [1., 0.5, 0.5]]).view(4, 3, 1, 1)
    s = pixel.hsv_saturation(colours.expand(4, 3, 2, 2))
    assert s[:, 0, 0].tolist() == pytest.approx([1.0, 0.0, 0.0, 0.5])
    batch = torch.rand(3, 3, 128, 128)
    d = pixel.pixel_descriptors(batch)
    assert d['pix64'].shape == (3, 3 * 64 * 64) and d['strided'].shape == (3, 3, 64, 64)
    assert d['brightness'] == pytest.approx(
        pixel.to_size(batch, 512).mean((1, 2, 3)).tolist(), abs=1e-6)
    assert pixel.evenly_spaced(10, 4) == [0, 3, 6, 9]


# -------------------------------------------------------------------- sources ----
def test_generated_set_from_directory_and_manifest(tmp_path):
    root = write_set(tmp_path / 'imgs', 6, tint=(1, 1, 1), seed=0, size=32)
    s = im.generated_set(root)
    assert len(s) == 6 and next(s.batches(4)).shape == (4, 3, 32, 32)
    assert len(im.generated_set(root, limit=3)) == 3
    manifest = tmp_path / 'm.jsonl'
    manifest.write_text('\n'.join(json.dumps({'generated_rgb_path': f'imgs/{i:03d}.png'})
                                  for i in range(6)))
    assert im.generated_set(manifest, root=tmp_path).ids == [str(tmp_path / f'imgs/{i:03d}.png')
                                                            for i in range(6)]
    with pytest.raises(ValueError):
        im.generated_set(tmp_path / 'empty_dir_does_not_exist.txt')


def test_reference_set_refuses_target_test_and_unknown_roles(tmp_path):
    for split in ('target_test', 'target_appearance_test', 'anything'):
        manifest = tmp_path / f'{split}.jsonl'
        manifest.write_text('')
        manifest.with_suffix('.meta.json').write_text(json.dumps({'split': split}))
        with pytest.raises(ValueError, match='not an allowed reference'):
            im.reference_set(manifest, data_root=tmp_path)


def test_floor_partner_manifest_names(tmp_path):
    for split, partner in (('source_validation', 'source_train.jsonl'),
                           ('target_appearance_validation', 'target_appearance_train.jsonl')):
        m = tmp_path / 'v.jsonl'
        m.with_suffix('.meta.json').write_text(json.dumps({'split': split}))
        assert im.floor_manifest_for(m) == tmp_path / partner
    m.with_suffix('.meta.json').write_text(json.dumps({'split': 'source_train'}))
    assert im.floor_manifest_for(m) is None


# -------------------------------------------------------------------- compare ----
def descriptors(path):
    return features.describe(im.generated_set(path), FakeExtractor(), batch_size=4)


def test_compare_sets_orders_close_far_and_reports_floor(tmp_path):
    ref = descriptors(write_set(tmp_path / 'ref', 40, tint=(1.0, 0.7, 0.5), seed=1))
    pool = descriptors(write_set(tmp_path / 'train', 60, tint=(1.0, 0.7, 0.5), seed=2))
    near = descriptors(write_set(tmp_path / 'near', 30, tint=(1.0, 0.7, 0.5), seed=3))
    far = descriptors(write_set(tmp_path / 'far', 30, tint=(0.4, 0.5, 1.0), seed=4))
    kw = dict(draws=6, kid_subsets=5, kid_subset_size=20, seed=0)
    r_near = cmp.compare_sets(near, ref, pool, **kw)
    r_far = cmp.compare_sets(far, ref, pool, **kw)
    for key in ('frechet',):
        assert r_far['generated_vs_reference'][key] > 5 * r_near['generated_vs_reference'][key]
    assert r_far['generated_vs_reference']['kid']['full'] > r_near['generated_vs_reference']['kid']['full']
    assert (r_far['generated_vs_reference']['w1_rgb']['mean']
            > 3 * r_near['generated_vs_reference']['w1_rgb']['mean'])
    floor = r_near['floor_train_vs_reference']
    assert floor['n_train_per_draw'] == 30 and floor['n_reference'] == 40
    assert r_far['generated_over_floor_ratio']['frechet'] > r_near['generated_over_floor_ratio']['frechet']
    # near set behaves like the floor: ratio of order one
    assert 0.2 < r_near['generated_over_floor_ratio']['frechet'] < 5
    spread = r_near['generated_vs_reference_spread']['distances']['frechet']
    assert spread['std'] > 0 and spread['p2.5'] <= spread['mean'] <= spread['p97.5']
    assert r_far['pixel_stats']['generated']['brightness'] != pytest.approx(
        r_far['pixel_stats']['reference']['brightness'], abs=1e-3)
    assert 'DINOv2' in r_near['metric_note']
    # deterministic given the seed
    assert cmp.compare_sets(near, ref, pool, **kw)['floor_train_vs_reference']['distances'] == \
        floor['distances']


def test_run_end_to_end_writes_json(tmp_path, monkeypatch):
    write_set(tmp_path / 'ref', 24, tint=(1, .8, .6), seed=1)
    write_set(tmp_path / 'train', 40, tint=(1, .8, .6), seed=2)
    write_set(tmp_path / 'gen', 20, tint=(1, .8, .6), seed=3)

    def fake_reference(manifest, *, data_root=None, limit=None):
        return im.generated_set(tmp_path / Path(manifest).stem, limit=limit), 'source_validation'

    monkeypatch.setattr(im, 'reference_set', fake_reference)
    monkeypatch.setattr(im, 'floor_manifest_for', lambda m: tmp_path / 'train.jsonl')
    args = cmp.build_parser().parse_args([
        '--generated', str(tmp_path / 'gen'), '--reference', str(tmp_path / 'ref.jsonl'),
        '--out', str(tmp_path / 'out' / 'result.json'), '--draws', '4', '--kid-subsets', '3',
        '--kid-subset-size', '10', '--workers', '0', '--cache-dir', str(tmp_path / 'cache')])
    result = cmp.run(args, extractor=FakeExtractor())
    saved = json.loads((tmp_path / 'out' / 'result.json').read_text())
    assert saved['setup']['feature_extractor'] == 'fake:grid8'
    assert saved['n_generated'] == 20 and saved['n_reference'] == 24
    assert saved['floor_train_vs_reference']['n_train_per_draw'] == 20
    assert 'Inception' in saved['setup']['inception_fid']
    assert list((tmp_path / 'cache').glob('*.pt'))      # reference + pool descriptors cached
    again = cmp.run(args, extractor=FakeExtractor())    # second call reads the cache, same numbers
    assert again['generated_vs_reference']['frechet'] == pytest.approx(
        result['generated_vs_reference']['frechet'])


# ------------------------------------------------------------ optional model ----
@pytest.mark.skipif(os.environ.get('GM_REAL_MODEL') != '1',
                    reason='set GM_REAL_MODEL=1 (needs facebook/dinov2-base in the HF cache)')
def test_dinov2_extractor_real_model():
    ext = features.Dinov2Extractor(image_size=224, device='cpu')
    batch = torch.rand(2, 3, 96, 96)
    out = ext(batch)
    assert out.shape == (2, 768) and torch.isfinite(out).all()
    assert not torch.allclose(out[0], out[1])
    assert torch.allclose(ext(batch[:1]), out[:1], atol=1e-4)
