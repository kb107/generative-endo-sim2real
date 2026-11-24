import json
from types import SimpleNamespace

import pytest
import torch

pytest.importorskip('diffusers')
pytest.importorskip('peft')
import generation_tiny as tiny  # noqa: E402

from generative_sim2real.data.preprocessing import InverseDepthEncoding  # noqa: E402
from generative_sim2real.generation import inputs  # noqa: E402

ENCODING = InverseDepthEncoding(near_m=0.007058823946863413, far_m=0.08549019694328308)


def source_sample(size=64):
    depth = torch.linspace(0.003, 0.12, size * size).reshape(1, size, size)
    valid = torch.ones_like(depth, dtype=torch.bool)
    valid[:, -4:] = False
    return {'sample_id': 's', 'rgb': torch.rand(3, size, size),
            'meta': {'depth_type': 'camera_z'},
            'targets': {'depth_m': depth, 'depth_valid': valid}}


def test_roles_reject_forbidden_manifests(tmp_path):
    for split, stage in (('target_test', 'generation'), ('source_validation', 'source_lora'),
                         ('target_appearance_validation', 'target_lora'),
                         ('source_validation', 'controller')):
        path = tmp_path / f'{split}.jsonl'
        path.with_suffix('.meta.json').write_text(json.dumps({'split': split}))
        with pytest.raises(ValueError, match='forbidden'):
            inputs.generation_dataset(path, tmp_path, stage=stage)


def test_encoding_matches_sidecar_constants_polarity_and_invalid_fill():
    sample = source_sample()
    image = ENCODING.encode(sample['targets']['depth_m'], sample['targets']['depth_valid'])
    assert image.dtype == torch.float32 and image.shape == (3, 64, 64)
    assert image.min() >= 0 and image.max() <= 1
    assert (image[:, -4:] == 0).all()  # invalid pixels use fill zero
    assert image[0, 10, 0] > image[0, 50, 63]  # near is bright
    # Constants recorded by the P1 sidecar are the authoritative source.
    assert (ENCODING.near_m, ENCODING.far_m) == (0.007058823946863413, 0.08549019694328308)


def test_vae_pixels_reject_out_of_range_and_wrong_dtype():
    with pytest.raises(ValueError, match=r'\[0,1\]'):
        inputs.vae_pixels(torch.full((3, 8, 8), 1.5))
    with pytest.raises(ValueError, match='float32'):
        inputs.vae_pixels(torch.zeros(3, 8, 8, dtype=torch.float64))
    with pytest.raises(ValueError, match='finite'):
        inputs.vae_pixels(torch.full((3, 8, 8), float('nan')))
    out = inputs.vae_pixels(torch.tensor([0., 1.]).view(1, 1, 2).expand(3, 1, 2).contiguous())
    assert out.min() == -1 and out.max() == 1


def test_control_uses_mode_rgb_uses_sample():
    vae = tiny.tiny_vae()
    image = torch.rand(2, 3, 64, 64)
    a = inputs.encode_image(vae, image, control=True)
    b = inputs.encode_image(vae, image, control=True)
    assert torch.equal(a, b) and a.shape == (2, 4, 8, 8)
    g1 = torch.Generator().manual_seed(1)
    g2 = torch.Generator().manual_seed(1)
    s1 = inputs.encode_image(vae, image, control=False, generator=g1)
    s2 = inputs.encode_image(vae, image, control=False, generator=g2)
    assert torch.equal(s1, s2) and not torch.equal(s1, a)


def test_control_image_requires_labels_and_frame_tensors_split_rules():
    with pytest.raises(ValueError, match='labeled'):
        inputs.control_image({'meta': {'depth_type': 'camera_z'},
                              'targets': {'depth_m': None, 'depth_valid': None}},
                             {'depth_control_encoding': {'constants': ENCODING.to_dict()}})
    target = SimpleNamespace(split='target_appearance_train')
    with pytest.raises(ValueError, match='labeled source'):
        inputs.FrameTensors(target, with_control=True)


def test_frame_tensors_rgb_only_never_touches_labels():
    class Manifest:
        split = 'target_appearance_train'
        sidecar = {}

        def __len__(self):
            return 1

        def __getitem__(self, index):
            return {'sample_id': 't', 'rgb': torch.rand(3, 8, 8),
                    'targets': {'depth_m': None, 'depth_valid': None}}

    item = inputs.FrameTensors(Manifest(), with_control=False)[0]
    assert set(item) == {'rgb', 'sample_id'}


def test_conditioning_trace_records_each_hop(tmp_path):
    trace = inputs.conditioning_trace(tiny.tiny_vae(), ENCODING, source_sample(),
                                      scratch=tmp_path / 'control.pt')
    assert trace['float32_roundtrip_exact'] and trace['mode_is_deterministic']
    assert trace['control_image']['channels_identical']
    assert 0 <= trace['control_image']['min'] and trace['control_image']['max'] <= 1
    assert -1 <= trace['vae_input']['min'] and trace['vae_input']['max'] <= 1
    assert trace['depth_vs_intensity_correlation'] < 0
    assert trace['saturated_near_fraction'] > 0 and trace['saturated_far_fraction'] > 0


def test_prompt_cache_roundtrip_and_mismatch(tmp_path):
    prompt = tiny.tiny_prompt()
    path = tmp_path / 'cache.safetensors'
    inputs.save_prompt_cache(path, (prompt['positive'], prompt['positive_mask']),
                             (prompt['negative'], prompt['negative_mask']),
                             prompt='p', revision='r')
    loaded = inputs.load_prompt_cache(path, prompt='p', revision='r')
    assert torch.equal(loaded['positive'], prompt['positive'])
    with pytest.raises(ValueError, match='does not match'):
        inputs.load_prompt_cache(path, prompt='other', revision='r')
    with pytest.raises(ValueError, match='does not match'):
        inputs.load_prompt_cache(path, prompt='p', revision='other')


def test_control_encoding_requires_camera_z():
    sample = source_sample()
    sidecar = {'depth_control_encoding': {'constants': ENCODING.to_dict()}}
    for bad in ('ray_range', None):
        sample['meta'] = {'depth_type': bad}
        with pytest.raises(ValueError, match='camera_z'):
            inputs.control_image(sample, sidecar)
        with pytest.raises(ValueError, match='camera_z'):
            inputs.conditioning_trace(tiny.tiny_vae(), ENCODING, sample, scratch='unused.pt')


class FakeManifest:
    split = 'source_train'
    sidecar = {'depth_control_encoding': {'constants': ENCODING.to_dict()}}

    def __len__(self):
        return 4

    def __getitem__(self, index):
        sample = source_sample()
        generator = torch.Generator().manual_seed(index)
        sample['sample_id'] = f'frame{index}'
        sample['rgb'] = torch.rand(3, 64, 64, generator=generator)
        sample['targets']['depth_m'] = sample['targets']['depth_m'] * (1 + 0.1 * index)
        return sample


def test_loader_path_equals_inference_path_for_control_image_and_latent():
    from torch.utils.data import DataLoader

    manifest, vae = FakeManifest(), tiny.tiny_vae()
    dataset = inputs.FrameTensors(manifest, with_control=True)
    batch = next(iter(DataLoader(dataset, batch_size=2, shuffle=False, num_workers=2)))
    direct = [manifest[i] for i in range(2)]
    images = torch.stack([inputs.control_image(s, manifest.sidecar) for s in direct])
    assert torch.equal(batch['control'], images)
    assert torch.equal(batch['rgb'], torch.stack([s['rgb'] for s in direct]))
    assert torch.equal(inputs.encode_image(vae, batch['control'], control=True),
                       inputs.encode_image(vae, images, control=True))


def test_generation_dataset_roles_and_geometry(tmp_path, monkeypatch):
    calls = []

    def fake(path, root, purpose, geometry):
        calls.append((purpose, geometry))
        return 'dataset'

    monkeypatch.setattr(inputs, 'ManifestDataset', fake)
    accepted = {('source_lora', 'source_train', None), ('controller', 'source_train', None),
                ('target_lora', 'target_appearance_train', None),
                ('generation', 'source_train', None),
                ('generation', 'source_validation', 'source_generation512'),
                ('validation', 'source_validation', 'source_generation512'),
                ('validation', 'target_appearance_validation', None)}
    for stage, split, geometry in sorted(accepted, key=str):
        path = tmp_path / f'{split}.jsonl'
        path.with_suffix('.meta.json').write_text(json.dumps({'split': split}))
        assert inputs.generation_dataset(path, tmp_path, stage=stage) == 'dataset'
        assert calls[-1] == ('training', geometry)
    with pytest.raises(ValueError, match='forbidden'):
        path = tmp_path / 'x.jsonl'
        path.with_suffix('.meta.json').write_text(json.dumps({'split': 'source_train'}))
        inputs.generation_dataset(path, tmp_path, stage='target_lora')
    with pytest.raises(ValueError, match='forbidden'):
        inputs.generation_dataset(path, tmp_path, stage='unknown_stage')
