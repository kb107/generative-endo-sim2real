"""Fixture tests for the appearance-control variants (no datasets, no models)."""
import json

import numpy as np
import pytest
import torch

from da3_fakes import FakeSplit
from generative_sim2real.data import rgb_modes as rm
from generative_sim2real.variants import build, color


def smooth_image(seed, size=64, tint=(1., 1., 1.), scale=1.):
    rng = np.random.default_rng(seed)
    base = rng.uniform(.2, .8, (3, size // 8, size // 8)).astype(np.float32)
    up = torch.nn.functional.interpolate(torch.from_numpy(base)[None], size=(size, size), mode='bilinear')[0]
    return (up * torch.tensor(tint)[:, None, None] * scale).clamp(0, 1).float()


def lab_stats(image, mask):
    return color.lab_stats(color.lab_of(image), mask)


def test_reinhard_matches_reference_statistics_and_range():
    source = smooth_image(0)
    reference = smooth_image(1, tint=(1., .6, .3), scale=.9)
    mask = np.ones((64, 64), bool)
    out = color.reinhard_lab(source, reference, mask, mask)
    assert out.dtype == torch.float32 and out.shape == source.shape and 0 <= out.min() and out.max() <= 1
    mu_o, sd_o = lab_stats(out, mask)
    mu_r, sd_r = lab_stats(reference, mask)
    # Gamut clipping and 8-bit-free float round trips make this approximate, but close for a mild transfer.
    assert np.allclose(mu_o, mu_r, atol=2.5) and np.allclose(sd_o, sd_r, rtol=.12, atol=1.5)


def test_reinhard_identity_when_reference_is_source():
    source = smooth_image(2)
    mask = np.ones((64, 64), bool)
    assert torch.allclose(color.reinhard_lab(source, source, mask, mask), source, atol=2e-3)


def test_reinhard_statistics_ignore_border_pixels():
    reference = smooth_image(3, tint=(1., .5, .4))
    bordered = reference.clone()
    bordered[:, :, :20] = 0.                                  # black lens border
    border_mask = color.luminance_mask(bordered)
    assert not border_mask[:, :20].any() and border_mask[:, 20:].all()
    source = smooth_image(4)
    full = np.ones((64, 64), bool)
    masked = color.reinhard_lab(source, bordered, full, border_mask)
    naive = color.reinhard_lab(source, bordered, full, full)
    inside = color.reinhard_lab(source, bordered[:, :, 20:], full, np.ones((64, 44), bool))
    assert torch.allclose(masked, inside, atol=1e-5)          # border pixels contributed nothing
    assert not torch.allclose(masked, naive, atol=1e-2)       # whereas using them would change the result


def test_luminance_mask_refuses_mostly_black():
    with pytest.raises(ValueError):
        color.luminance_mask(torch.zeros(3, 16, 16))


def test_strong_params_ranges_and_determinism():
    draws = [color.draw_strong_params(np.random.default_rng(s)) for s in range(300)]
    again = color.draw_strong_params(np.random.default_rng(5))
    assert draws[5] == again
    for key, (low, high) in color.STRONG_RANGES.items():
        values = [v for d in draws for v in (d[key] if key == 'channel_gain' else [d[key]])]
        assert low <= min(values) and max(values) <= high
        assert max(values) - min(values) > .8 * (high - low)  # the full range is actually used
    assert all(len(d['channel_gain']) == 3 for d in draws)
    assert .35 < np.mean([d['blur_applied'] for d in draws]) < .65


def test_strong_photometric_identity_and_effects():
    image = smooth_image(6)
    assert torch.allclose(color.apply_strong_photometric(image, color.IDENTITY_PARAMS), image, atol=1e-6)
    out = color.apply_strong_photometric(image, {**color.IDENTITY_PARAMS, 'brightness': .5})
    assert torch.allclose(out, image * .5, atol=1e-6)
    gamma = color.apply_strong_photometric(image, {**color.IDENTITY_PARAMS, 'gamma': 1.4})
    assert torch.allclose(gamma, image ** 1.4, atol=1e-6)
    gray = color.apply_strong_photometric(image, {**color.IDENTITY_PARAMS, 'saturation': 0.})
    assert torch.allclose(gray[0], gray[1], atol=1e-5) and torch.allclose(gray[1], gray[2], atol=1e-5)
    hue = color.apply_strong_photometric(image, {**color.IDENTITY_PARAMS, 'hue_shift': .05})
    assert (hue - image).abs().mean() > .01 and torch.allclose(hue.mean(), image.mean(), atol=.05)
    blur = color.apply_strong_photometric(image, {**color.IDENTITY_PARAMS, 'blur_sigma': 1.5, 'blur_applied': True})
    assert blur.std() < image.std()
    noisy = {**color.IDENTITY_PARAMS, 'noise_sigma': .03, 'noise_seed': 11}
    a, b = (color.apply_strong_photometric(image, noisy) for _ in range(2))
    assert torch.equal(a, b) and .01 < (a - image).std() < .04
    for p in (color.draw_strong_params(np.random.default_rng(s)) for s in range(20)):
        out = color.apply_strong_photometric(image, p)
        assert out.dtype == torch.float32 and 0 <= out.min() and out.max() <= 1


def test_fda_identity_dc_and_high_frequencies():
    source = smooth_image(0, 128)
    reference = smooth_image(1, 128, tint=(1., .6, .3), scale=.9)
    assert torch.allclose(color.fda_transfer(source, source, .05), source, atol=1e-5)
    mild = smooth_image(2, 128, tint=(.95, .85, .8))  # no clipping: b = 0 swaps only the DC term -> channel means follow
    assert torch.allclose(color.fda_transfer(source, mild, 0.).mean((1, 2)), mild.mean((1, 2)), atol=2e-3)
    out = color.fda_transfer(source, reference, 0.)
    small, large = color.fda_transfer(source, reference, .01), color.fda_transfer(source, reference, .05)
    assert out.shape == source.shape and out.dtype == torch.float32 and 0 <= large.min() and large.max() <= 1
    # a larger box moves the image further from the source; the phase (edges) stays the source's
    assert (large - source).abs().mean() >= (small - source).abs().mean()
    with pytest.raises(ValueError):
        color.fda_transfer(source, reference, .6)


def test_fda_variants_reuse_t_reference_and_keep_t_recipe():
    assert set(build.FDA_BETA) <= set(build.PROVENANCE) and 'T_reinhard' in build.TARGET_VARIANTS
    assert build.PROVENANCE['T_reinhard']['target_adapter'] == 'reinhard_lab:c1_train'
    assert build.recipe_hash('F_fda_b01') != build.recipe_hash('F_fda_b05') != build.recipe_hash('T_reinhard')


def test_seeds_are_deterministic_and_id_specific():
    assert build.variant_seed('A_strong', 'a') == build.variant_seed('A_strong', 'a')
    assert build.variant_seed('A_strong', 'a') != build.variant_seed('A_strong', 'b')
    assert build.variant_seed('A_strong', 'a') != build.variant_seed('T_reinhard', 'a')
    picks = {build.reference_index(build.variant_seed('T_reinhard', f'id{i}'), 1895) for i in range(200)}
    assert len(picks) > 150 and all(0 <= p < 1895 for p in picks)


def fake_variant(tmp_path, name, count=5):
    """Mimic _work() on FakeSplit samples and write a manifest + PNGs under tmp_path/out."""
    source = FakeSplit(tmp_path, 'source_train', 'source_generation512', count, 512)
    reference = smooth_image(9, 512, tint=(1., .5, .4))
    records = []
    for index, row in enumerate(source.rows):
        sample = source[index]
        seed = build.variant_seed(name, row['sample_id'])
        image, extra = build.make_image(name, sample, seed, reference if name in build.TARGET_VARIANTS else None)
        path = tmp_path / 'out' / 'generated_variants' / name / f'{index:05d}.png'
        sha = build.png_sha256(image, path)
        records.append(build.make_record(name, row['sample_id'], row, source.sidecar, rm.geometry_sha256(sample), seed,
                                         path.relative_to(tmp_path / 'out').as_posix(), sha, extra))
    manifest = tmp_path / 'generated.jsonl'
    manifest.write_text(''.join(json.dumps(r) + '\n' for r in records))
    return source, records, manifest


@pytest.mark.parametrize('name', ['T_reinhard', 'A_strong', 'F_fda_b01', 'F_fda_b05'])
def test_manifest_records_are_accepted_by_g_and_m_loader(tmp_path, name):
    source, records, manifest = fake_variant(tmp_path, name)
    ids = [r['sample_id'] for r in records]
    for key in rm.GENERATED_PROVENANCE:
        assert records[0][key] is not None
        assert all(r[key] == records[0][key] for r in records if key != 'seed')
    assert len({r['seed'] for r in records}) == len(records)
    assert records[0]['base_model'] == 'none' and records[0]['controller_scale'] == 0.0
    for mode in ('G', 'M'):
        dataset = rm.RGBModeDataset(source, ids, mode=mode, seed=0, generated_manifest=manifest,
                                    outputs_root=tmp_path / 'out')
        assert len(dataset) == len(ids)
    dataset = rm.RGBModeDataset(source, ids, mode='G', seed=0, generated_manifest=manifest, outputs_root=tmp_path / 'out')
    sample = dataset[2]
    assert sample['rgb'].shape == (3, 504, 504) and sample['targets']['depth_m'].shape == (1, 504, 504)
    # the loaded image is the saved PNG (8-bit) resized: it is not the raw source
    assert not torch.allclose(sample['rgb'], rm.RGBModeDataset(source, ids, mode='R', seed=0)[2]['rgb'], atol=1e-3)


def test_tampered_variant_image_is_rejected(tmp_path):
    source, records, manifest = fake_variant(tmp_path, 'A_strong', count=3)
    (tmp_path / 'out' / records[1]['generated_rgb_path']).write_bytes(b'not a png')
    with pytest.raises(ValueError, match='checksum'):
        rm.RGBModeDataset(source, [r['sample_id'] for r in records], mode='G', seed=0, generated_manifest=manifest,
                          outputs_root=tmp_path / 'out')


def test_images_are_deterministic(tmp_path):
    for sub in ('a', 'b'):
        (tmp_path / sub).mkdir()
    _, first, _ = fake_variant(tmp_path / 'a', 'A_strong', count=3)
    _, second, _ = fake_variant(tmp_path / 'b', 'A_strong', count=3)
    assert [r['generated_rgb_sha256'] for r in first] == [r['generated_rgb_sha256'] for r in second]
