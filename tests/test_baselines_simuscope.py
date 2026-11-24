"""SimuScope-style baseline: frame choice, kohya command, recipe/manifest wiring. Fixtures only; CPU."""
import json
from pathlib import Path

import pytest
import torch

import da3_fakes as fakes
from generative_sim2real.baselines import simuscope as ss
from generative_sim2real.data import rgb_modes as rm
from generative_sim2real.generation import bulk

N = 7
CONFIG = ss.load_config(Path(__file__).resolve().parents[1] / 'configs' / 'simuscope.yaml')


def c1_rows(sizes):
    rows = []
    for fam, (n_v1, n_v2) in sizes.items():
        for version, n in (('v1', n_v1), ('v2', n_v2)):
            for k in range(n):
                seq = f'c1_{fam}_{version}'
                rows.append({'sample_id': f'c3vdv2/c1/{seq}/{k * 8:04d}', 'sequence_id': seq, 'frame_index': k * 8,
                             'rgb_path': f'x/{seq}/{k * 8:04d}.png'})
    return rows


SIZES = {f'seg{i}_t{i % 4 + 1}': (10 + i, 30 + 3 * i) for i in range(23)}


def test_selection_is_deterministic_family_diverse_and_exact():
    rows = c1_rows(SIZES)
    first = ss.select_style_frames(rows, 85)
    assert [r['sample_id'] for r in first] == [r['sample_id'] for r in ss.select_style_frames(rows[::-1], 85)]
    assert len(first) == 85 and len({r['sample_id'] for r in first}) == 85
    per_family = {}
    for row in first:
        per_family[ss.family(row['sequence_id'])] = per_family.get(ss.family(row['sequence_id']), 0) + 1
    assert set(per_family) == set(SIZES) and max(per_family.values()) - min(per_family.values()) <= 1


def test_selection_spreads_within_a_family_and_caps_at_availability():
    rows = c1_rows({'a_t1': (50, 50), 'b_t2': (1, 1)})
    chosen = ss.select_style_frames(rows, 85)
    a = sorted(r['frame_index'] for r in chosen if ss.family(r['sequence_id']) == 'a_t1')
    b = [r for r in chosen if ss.family(r['sequence_id']) == 'b_t2']
    assert len(b) == 2 and 42 <= len(a) <= 43 and len(chosen) == len(a) + 2   # b_t2 is capped by availability
    assert a[0] < 100 and a[-1] > 300


def test_selection_rejects_rows_with_label_fields():
    rows = c1_rows({'a_t1': (5, 5)})
    rows[0]['depth_path'] = 'x.tiff'
    with pytest.raises(ValueError, match='label'):
        ss.select_style_frames(rows, 4)


def test_selection_record_hashes_manifest_and_lists_frames(tmp_path):
    rows = c1_rows(SIZES)
    manifest = tmp_path / 'm.jsonl'
    manifest.write_text(''.join(json.dumps(r) + '\n' for r in rows))
    record = ss.selection_record(rows, manifest, 85)
    assert record['count'] == 85 == len(record['frames']) and sum(record['families'].values()) == 85
    assert record['source_manifest_sha256'] == bulk.file_sha256(manifest)
    assert all(set(f) == {'sample_id', 'sequence_id', 'family', 'frame_index', 'rgb_path'} for f in record['frames'])


def test_kohya_command_uses_the_official_training_settings():
    command = ss.kohya_command(CONFIG, '/data/img', '/models/sd15', root='.')
    text = ' '.join(command)
    for flag in ('--learning_rate=0.0012', '--network_dim=128', '--network_alpha=1', '--train_batch_size=5',
                 '--optimizer_type=Adafactor', '--min_snr_gamma=5', '--lr_scheduler=constant_with_warmup',
                 '--max_train_steps=13920', '--network_train_unet_only', '--resolution=512,512',
                 '--text_encoder_lr=0.0012', '--unet_lr=0.0012', '--max_grad_norm=1', '--mixed_precision=fp16'):
        assert flag in command, flag
    assert 'xformers' not in text and '--train_data_dir=/data/img' in command
    assert CONFIG['inference'] == {**CONFIG['inference'], 'steps': 20, 'denoising_strength': 0.65, 'cfg_scale': 7.0,
                                   'softedge_weight': 0.45, 'depth_weight': 0.65, 'lora_weight': 0.45, 'seed': 420420}


def test_export_uses_c1_rgb_loader_only(tmp_path, monkeypatch):
    split = fakes.FakeSplit(tmp_path, 'target_appearance_train', 'target_generation512', 6, 512)
    stages = []

    def fake_dataset(path, data_root, *, stage):
        stages.append(stage)
        return split
    monkeypatch.setattr('generative_sim2real.generation.inputs.generation_dataset', fake_dataset)
    config = json.loads(json.dumps(CONFIG))
    config['style_lora'].update(manifest='m.jsonl', work_dir='work')
    selection = {'frames': [{'sample_id': split.rows[i]['sample_id']} for i in (4, 1, 2)]}
    out = ss.export_kohya_dataset(config, selection, str(tmp_path), root=tmp_path)
    assert stages == ['target_lora']
    folder = out / f'1_{config["style_lora"]["caption"]}'    # kohya takes the caption from the folder name
    assert sorted(p.name for p in folder.iterdir()) == ['000.png', '001.png', '002.png']


def test_jpeg_roundtrip_is_deterministic_lossy_and_size_preserving():
    import numpy as np
    from PIL import Image
    rng = np.random.default_rng(0)
    ramp = np.linspace(0, 255, 64)[None, :, None] * np.ones((64, 1, 3))
    pil = Image.fromarray(np.clip(ramp + rng.normal(0, 12, (64, 64, 3)), 0, 255).astype(np.uint8))
    first, second = ss.jpeg_roundtrip(pil), ss.jpeg_roundtrip(pil)
    assert first.size == (64, 64) and first.mode == 'RGB' and np.array_equal(np.asarray(first), np.asarray(second))
    assert not np.array_equal(np.asarray(first), np.asarray(pil)) and np.abs(np.asarray(first).astype(int) - np.asarray(pil)).mean() < 10


# ------------------------------------------------- recipe, bulk run, manifest ----

@pytest.fixture(autouse=True)
def small_shards(monkeypatch):
    monkeypatch.setattr(bulk, 'SHARD_SIZE', 3)
    monkeypatch.setattr(bulk, 'per_id_seed', ss.official_seed)


@pytest.fixture
def source(tmp_path):
    split = fakes.FakeSplit(tmp_path, 'source_train', 'source_generation512', N, 512)
    split.sidecar['depth_control_encoding'] = {'constants': {'near_m': .01, 'far_m': .1, 'invalid_fill': 0.}}
    return split


def fake_files(tmp_path):
    files = {}
    for key in ('target_adapter', 'style_frames', 'controlnet_softedge', 'controlnet_depth'):
        path = tmp_path / f'{key}.bin'
        path.write_bytes(key.encode())
        files[key] = {'path': str(path), 'sha256': bulk.file_sha256(path)}
    return files


class FakeRender:
    def __init__(self):
        self.seeds = []

    def __call__(self, samples, seeds):
        self.seeds += list(seeds)
        return torch.stack([s['rgb'] for s in samples]).clamp(0, 1) * .5 + .25


def setup(tmp_path, source, config=CONFIG, subdir='generated_baselines/simuscope'):
    resolved = ss.resolve(config, fake_files(tmp_path), purpose='production')
    layout = ss.SimuScopeLayout(tmp_path / 'outputs', subdir)
    return resolved, layout, [r['sample_id'] for r in source.rows]


def test_run_manifest_is_accepted_by_rgb_modes_with_paths_under_outputs(tmp_path, source):
    resolved, layout, ids = setup(tmp_path, source)
    render = FakeRender()
    summary = bulk.generate(resolved, source, ids, layout, render, log=lambda *_: None, environment={
        'code_sha256': 'x', 'gpu': None, 'versions': {}})
    assert (summary['generated'], summary['failed']) == (N, 0)
    info = bulk.build_manifest(resolved, source, ids, layout)
    assert info['records'] == N and info['rgb_modes_validation']['G_all_items_loaded'] == N
    records = [json.loads(line) for line in layout.manifest.read_text().splitlines()]
    assert [r['sample_id'] for r in records] == ids
    assert all(r['generated_rgb_path'].startswith('generated_baselines/simuscope/shards/') for r in records)
    assert {r['seed'] for r in records} == {420420} and set(render.seeds) == {420420}   # official fixed seed
    first = records[0]
    assert first['source_adapter'].startswith('none') and first['adapter_weight'] == .45
    assert first['controller_scale'] == [.45, .65] and first['method'] == 'SimuScope-style'
    assert first['denoising_strength'] == .65 and first['guidance_scale'] == 7.0 and first['inference_steps'] == 20
    for key in rm.GENERATED_PROVENANCE:
        assert first[key] is not None
    # the consumer, independently of bulk, accepts it with outputs_root = outputs
    dataset = rm.RGBModeDataset(source, ids, mode='M', generated_manifest=layout.manifest, outputs_root=tmp_path / 'outputs')
    assert dataset[0]['rgb'].shape == (3, 504, 504)
    assert bulk.verify(resolved, source, ids, layout)['invalid_count'] == 0


def test_recipe_hash_covers_settings_and_weights(tmp_path, source):
    base, _, _ = setup(tmp_path, source)
    other = json.loads(json.dumps(CONFIG))
    other['inference']['denoising_strength'] = .5
    assert ss.resolve(other, fake_files(tmp_path), purpose='production')['recipe_hash'] != base['recipe_hash']
    files = fake_files(tmp_path)
    files['target_adapter']['sha256'] = '0' * 64
    assert ss.resolve(CONFIG, files, purpose='production')['recipe_hash'] != base['recipe_hash']
    assert ss.resolve(CONFIG, fake_files(tmp_path), purpose='smoke')['recipe_hash'] == base['recipe_hash']


def test_smoke_directory_is_labelled_and_production_refuses_smoke_paths():
    tmp_path = Path('/nonexistent/root')
    with pytest.raises(bulk.BulkError):
        bulk.check_destination('production', tmp_path / 'outputs' / CONFIG['output']['smoke_dir'])
    bulk.check_destination('smoke', tmp_path / 'outputs' / CONFIG['output']['smoke_dir'])
    bulk.check_destination('production', tmp_path / 'outputs' / CONFIG['output']['production_dir'])
