"""Generated, small dataset fixtures; no shared data/model dependency."""
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image
import pytest
import torch
import yaml

from generative_sim2real.data import manifests as m


@pytest.fixture
def fixture_views(tmp_path, monkeypatch):
    root = tmp_path / 'data'
    source = root / 'SimCol3D'
    (source / 'misc').mkdir(parents=True)
    (source / 'misc/train_file.txt').write_text(
        'scene/Frames_train\nscene/Frames_val\nscene2/Frames_val\n')
    (source / 'misc/test_file.txt').write_text('scene/Frames_test\n')
    scene = source / 'scene'
    scene.mkdir()
    (scene / 'cam.txt').write_text('1 0 0 0 1 0 0 0 1\n')
    for part in ('train', 'val', 'test'):
        (scene / f'SavedPosition_{part}.txt').write_text('0 0 0\n')
        (scene / f'SavedRotationQuaternion_{part}.txt').write_text('0 0 0 1\n')
    scene2 = source / 'scene2'
    scene2.mkdir()
    (scene2 / 'cam.txt').write_text('1 0 0 0 1 0 0 0 1\n')
    (scene2 / 'SavedPosition_val.txt').write_text('0 0 0\n')
    (scene2 / 'SavedRotationQuaternion_val.txt').write_text('0 0 0 1\n')
    target = root / 'C3VDv2'
    target.mkdir()
    (target / 'metadata.xlsx').write_bytes(b'fixture metadata')
    config = yaml.safe_load(Path('configs/data.yaml').read_text())
    config['source'].update(
        validation_sequences=['scene/Frames_val', 'scene2/Frames_val'],
        train_stride=1, validation_stride=1, train_cap=4, validation_cap=4)
    config['target'].update(validation_families=['cecum_t1'], train_stride=1,
                            validation_stride=1, train_cap=4, validation_cap=4)
    config['depth_encoding'].update(fit_frames=2, pixels_per_frame=8)
    config['diagnostics']['conditions'] = 4
    config['output_directory'] = 'artifacts'
    config['smoke']['frames_per_split'] = 1
    views = {}
    for split in m.SPLITS:
        records = []
        for index in range(6):
            if split.startswith('source'):
                sequence = 'Frames_val' if split.endswith('validation') else 'Frames_train'
                source_scene = scene2 if split.endswith('validation') and index >= 3 else scene
                directory = source_scene / sequence
                sample_id = f'simcol3d/{source_scene.name}/{sequence}/{index:04d}'
                common = dict(
                    sequence_key=f'{source_scene.name}/{sequence}',
                    intrinsics_path=source_scene / 'cam.txt',
                    position_path=source_scene / f'SavedPosition_{sequence[7:]}.txt',
                    rotation_path=source_scene / f'SavedRotationQuaternion_{sequence[7:]}.txt')
            else:
                colon = 'c2' if split == 'target_test' else 'c1'
                family = 'cecum_t1' if split.endswith('validation') else 'ascending_t2'
                sequence = f'{colon}_{family}_v1'
                directory = target / 'frames' / colon.upper() / sequence / 'rgb'
                if split == 'target_test':
                    directory = directory.parent / sequence / 'rgb'
                sample_id = f'c3vdv2/{colon}/{sequence}/{index:04d}'
                common = {}
            directory.mkdir(parents=True, exist_ok=True)
            if split == 'target_test':
                (directory.parent / 'pose.txt').write_text('fixture target pose\n')
            rgb = directory / f'{index:04d}.png'
            rgb.write_bytes(b'fixture rgb')
            if not split.startswith('target_appearance'):
                depth = directory / f'{index:04d}.depth'
                depth.write_bytes(b'fixture depth')
                common['depth_path'] = depth
            records.append(SimpleNamespace(sample_id=sample_id, sequence_id=sequence,
                                            frame_index=index, rgb_path=rgb, **common))
        class FakeView:
            def __init__(self, records, rgb_only):
                self.records = records
                self.rgb_only = rgb_only

            def __getitem__(self, index):
                # A spatial pattern exercises paired interpolation and positive quantiles.
                values = torch.linspace(.01 + index * .001, .08 + index * .001, 24).reshape(1, 4, 6)
                return {'sample_id': self.records[index].sample_id,
                        'rgb': values.repeat(3, 1, 1),
                        'targets': {'depth_m': None if self.rgb_only else values,
                                    'depth_valid': None if self.rgb_only else values > 0,
                                    'sparse_depth': None, 'instances': None},
                        'camera': {'K': torch.eye(3), 'model': 'pinhole',
                                   'T_world_from_camera': None, 'distortion': None},
                        'meta': {'original_hw': (4, 6),
                                 'sequence_id': self.records[index].sequence_id}}
        views[split] = FakeView(records, split.startswith('target_appearance'))
    monkeypatch.setattr(m, 'build_view', lambda split, config, root: views[split])
    monkeypatch.setattr(m, 'SimCol3D', lambda *args, **kwargs: SimpleNamespace(records=[
        SimpleNamespace(sample_id='simcol3d/scene/Frames_test/0000')]))
    monkeypatch.chdir(tmp_path)
    cfg = tmp_path / 'config.yaml'
    cfg.write_text(yaml.safe_dump(config))
    return root, config, cfg, views


def test_build_repeat_reload_exact_selection_and_rgb_boundary(fixture_views):
    root, config, cfg, views = fixture_views
    report = m.build_manifests(cfg, root)
    assert report['counts']['source_train'] == 4
    assert report['counts']['target_test'] == 6
    output = Path(config['output_directory'])
    before = {p.name: p.read_bytes() for p in output.iterdir()}
    m.build_manifests(cfg, root)
    assert before == {p.name: p.read_bytes() for p in output.iterdir()}
    loaded = m.ManifestDataset(output / 'source_train.jsonl', root)
    assert loaded.indices == [0, 1, 3, 5]
    assert loaded[2]['sample_id'] == views['source_train'].records[3].sample_id
    assert loaded[2]['rgb'].shape == (3, 512, 512)
    expected = m.apply_geometry(views['source_train'][3], 'source_generation512')
    torch.testing.assert_close(loaded[2]['targets']['depth_m'], expected['targets']['depth_m'])
    assert torch.equal(loaded[2]['targets']['depth_valid'], expected['targets']['depth_valid'])
    torch.testing.assert_close(loaded[2]['camera']['K'], expected['camera']['K'])
    validation = m.ManifestDataset(output / 'source_validation.jsonl', root)
    assert validation[0]['rgb'].shape == (3, 4, 6)
    appearance = m.ManifestDataset(output / 'target_appearance_validation.jsonl', root)
    assert all(not any(token in key for token in ('depth', 'pose', 'mask'))
               for row in appearance.rows for key in row)
    assert appearance[0]['targets']['depth_m'] is None
    diag = json.loads((output / 'diagnostics.json').read_text())
    assert all(a != b for a, b in zip(diag['sample_ids'], diag['wrong_depth_sample_ids']))
    assert diag['seeds'] == [17, 43]
    assert diag['downstream_N'] is None


def test_hash_path_encoding_and_role_tampering(fixture_views, monkeypatch):
    root, config, cfg, _ = fixture_views
    m.build_manifests(cfg, root)
    output = Path(config['output_directory'])
    path = output / 'source_train.jsonl'
    original = path.read_bytes()
    path.write_bytes(original + b'\n')
    with pytest.raises(ValueError, match='hash'):
        m.ManifestDataset(path, root)
    path.write_bytes(original)
    side_path = path.with_suffix('.meta.json')
    side = json.loads(side_path.read_text())
    rows = [json.loads(line) for line in original.splitlines()]
    for key, value, match in (('rgb_path', '../outside.png', 'Unsafe'),
                               ('rgb_path', '/outside.png', 'Unsafe'),
                               ('depth_encoding', 'meters', 'differ'),
                               ('transform_id', 'source_native', 'differ')):
        changed = [{**rows[0], key: value}, *rows[1:]]
        content = m.manifest_bytes(changed)
        path.write_bytes(content)
        side_path.write_bytes(m.json_bytes({**side, 'manifest_sha256': m.digest(content)}))
        with pytest.raises(ValueError, match=match):
            m.ManifestDataset(path, root)
    monkeypatch.setattr(
        m, 'build_view', lambda *args: pytest.fail('C2 adapter constructed for training'))
    with pytest.raises(ValueError, match='target_test'):
        m.ManifestDataset(output / 'target_test.jsonl', root)


def test_governing_hash_and_label_schema(fixture_views):
    root, config, cfg, _ = fixture_views
    m.build_manifests(cfg, root)
    path = Path(config['output_directory']) / 'target_appearance_train.jsonl'
    side_path = path.with_suffix('.meta.json')
    side = json.loads(side_path.read_text())
    rows = [json.loads(line) for line in path.read_bytes().splitlines()]
    rows[0]['depth_path'] = rows[0]['rgb_path']
    content = m.manifest_bytes(rows)
    path.write_bytes(content)
    side_path.write_bytes(m.json_bytes({**side, 'manifest_sha256': m.digest(content)}))
    with pytest.raises(ValueError, match='label'):
        m.ManifestDataset(path, root)
    (root / 'SimCol3D/scene/cam.txt').write_text('changed')
    with pytest.raises(ValueError, match='Governing'):
        m.ManifestDataset(Path(config['output_directory']) / 'source_train.jsonl', root)


def test_split_checks_reject_sequence_and_family_leakage():
    def row(identity, sequence):
        return {'sample_id': identity, 'sequence_id': sequence}
    rows = {'source_train': [row('s1', 'scene/a')],
            'source_validation': [row('s2', 'scene/b')],
            'target_appearance_train': [row('c1', 'c1_cecum_t1_v1')],
            'target_appearance_validation': [row('c2', 'c1_sigmoid1_t3_v1')],
            'target_test': [row('c3', 'c2_ascending_t1_v1')]}
    assert m.check_splits(rows, {'t'})['official_source_test_overlap'] == 0
    with pytest.raises(ValueError, match='test contamination'):
        m.check_splits(rows, {'s1'})
    rows['source_validation'][0]['sequence_id'] = 'scene/a'
    with pytest.raises(ValueError, match='trajectory'):
        m.check_splits(rows, {'t'})
    rows['source_validation'][0]['sequence_id'] = 'scene/b'
    rows['target_appearance_validation'][0]['sequence_id'] = 'c1_cecum_t1_v2'
    with pytest.raises(ValueError, match='family'):
        m.check_splits(rows, {'t'})


def test_paired_da3_chain_and_k():
    sample = {'sample_id': 's', 'rgb': torch.ones(3, 475, 475),
              'targets': {'depth_m': torch.ones(1, 475, 475) * .03,
                          'depth_valid': torch.ones(1, 475, 475, dtype=torch.bool),
                          'sparse_depth': None, 'instances': None},
              'camera': {'K': torch.tensor([[200., 0, 237], [0, 210, 238], [0, 0, 1]])},
              'meta': {}}
    out = m.apply_geometry(sample, 'source_da3_504')
    assert out['targets']['depth_m'].shape == (1, 504, 504)
    assert torch.all(out['targets']['depth_valid'])
    assert torch.all(out['targets']['depth_m'] == .03)
    torch.testing.assert_close(out['camera']['K'][:2], sample['camera']['K'][:2] * (504 / 475))
    assert len(out['meta']['transform_history']) == 2


def test_symlink_boundary(tmp_path):
    root, external = tmp_path / 'root', tmp_path / 'external'
    (root / 'C3VDv2').mkdir(parents=True)
    external.mkdir()
    (external / 'image.png').write_bytes(b'fixture')
    (root / 'C3VDv2/frames').symlink_to(external, target_is_directory=True)
    assert m.data_path(root, 'C3VDv2/frames/image.png', 'C3VDv2').is_file()
    (root / 'escape').symlink_to(external, target_is_directory=True)
    with pytest.raises(ValueError, match='symlink'):
        m.data_path(root, 'escape/image.png', 'C3VDv2')


def test_diagnostic_cross_scene_bijection_reduced_gate_and_determinism():
    rows, samples = [], []
    for scene, count in (('scene_a', 21), ('scene_b', 24)):
        for index in range(count):
            identity = f'simcol3d/{scene}/Frames_val/{index:04d}'
            rows.append({'sample_id': identity, 'sequence_id': f'{scene}/Frames_val',
                         'dataset': 'simcol3d', 'split': 'source_validation'})
            samples.append({'targets': {
                'depth_m': torch.full((1, 2, 2), .01 + index * .001),
                'depth_valid': torch.ones((1, 2, 2), dtype=torch.bool)}})
    dataset = SimpleNamespace(rows=rows, raw_sample=samples.__getitem__)
    result = m.diagnostic_selection(dataset, 32, [17, 43])
    assert result == m.diagnostic_selection(dataset, 32, [17, 43])
    assert result['scene_ids'].count('scene_a') == 16
    assert result['scene_ids'].count('scene_b') == 16
    assert result['wrong_depth_permutation'] == list(range(31, -1, -1))
    for prefix in ('', 'reduced_gate_'):
        ids = result[prefix + 'sample_ids']
        scenes = result[prefix + 'scene_ids']
        wrong_ids = result[prefix + 'wrong_depth_sample_ids']
        permutation = result[prefix + 'wrong_depth_permutation']
        assert len(ids) == len(set(ids))
        assert set(ids) == set(wrong_ids)
        assert wrong_ids == [ids[i] for i in permutation]
        assert all(a != b for a, b in zip(ids, wrong_ids))
        assert all(scenes[i] != scenes[j] for i, j in enumerate(permutation))
    assert len(result['reduced_gate_sample_ids']) == 8
    assert set(result['reduced_gate_sample_ids']) <= set(result['sample_ids'])

    with pytest.raises(ValueError, match='even'):
        m.diagnostic_selection(dataset, 31, [17, 43])
    with pytest.raises(ValueError, match='Insufficient'):
        m.diagnostic_selection(dataset, 44, [17, 43])
    with pytest.raises(ValueError, match='two validation scenes'):
        m.diagnostic_selection(SimpleNamespace(rows=rows[:21], raw_sample=samples.__getitem__),
                               8, [17, 43])
    duplicate = SimpleNamespace(rows=[rows[0], rows[0]], raw_sample=samples.__getitem__)
    with pytest.raises(ValueError, match='unique'):
        m.diagnostic_selection(duplicate, 2, [17, 43])
    forbidden = SimpleNamespace(rows=[{**rows[0], 'split': 'target_test'}],
                                raw_sample=lambda index: pytest.fail('Target depth decoded'))
    with pytest.raises(ValueError, match='source_validation'):
        m.diagnostic_selection(forbidden, 2, [17, 43])


@pytest.mark.parametrize('input_hw,content_hw,top_left', [
    ((512, 512), (240, 240), (0, 40)),
    ((504, 630), (240, 300), (0, 10)),
    ((630, 504), (240, 192), (0, 64)),
])
def test_panel_preserves_aspect_and_shared_rgb_depth_control_bounds(
    tmp_path, input_hw, content_hw, top_left,
):
    assert m.panel_content_geometry(input_hw) == (content_hw, top_left)
    height, width = input_hw
    sample = {'sample_id': 'fixture/panel', 'rgb': torch.ones(3, height, width),
              'targets': {'depth_m': torch.full((1, height, width), .02),
                          'depth_valid': torch.ones((1, height, width), dtype=torch.bool)}}
    path = tmp_path / 'panel.png'
    m.smoke_panel([sample], path, m.InverseDepthEncoding(.01, .1))
    panel = np.asarray(Image.open(path))
    expected = (top_left[0], top_left[1],
                top_left[0] + content_hw[0], top_left[1] + content_hw[1])
    for column in range(3):
        tile = panel[20:260, column * 320:(column + 1) * 320]
        yy, xx = np.nonzero(np.any(tile != 37, axis=-1))
        assert (yy.min(), xx.min(), yy.max() + 1, xx.max() + 1) == expected
    # Nearest depth/control displays contain only their original grayscale + padding.
    for column in (1, 2):
        tile = panel[20:260, column * 320:(column + 1) * 320]
        assert len(np.unique(tile)) == 2


def test_c2_pose_hash_detects_mutation_before_loader_construction(fixture_views, monkeypatch):
    root, config, cfg, views = fixture_views
    m.build_manifests(cfg, root)
    path = Path(config['output_directory']) / 'target_test.jsonl'
    side = json.loads(path.with_suffix('.meta.json').read_text())
    pose = views['target_test'].records[0].rgb_path.parent.parent / 'pose.txt'
    relative_pose = pose.relative_to(root).as_posix()
    assert relative_pose in side['governing_files_sha256']
    assert pose.parent.name == pose.parent.parent.name  # Nested release payload fixture.
    pose.write_text('changed target pose\n')
    monkeypatch.setattr(m, 'build_view', lambda *args: pytest.fail('Stale C2 adapter constructed'))
    with pytest.raises(ValueError, match='Governing data file hash mismatch.*pose.txt'):
        m.ManifestDataset(path, root, purpose='audit')


@pytest.mark.parametrize('nested', [False, True])
def test_target_governing_hashes_use_actual_payload_and_never_c1_labels(tmp_path, nested):
    target = tmp_path / 'C3VDv2'
    target.mkdir()
    metadata = target / 'metadata.xlsx'
    metadata.write_bytes(b'fixture metadata')
    payload = target / 'frames/C2/sequence'
    if nested:
        payload = payload / 'sequence'
    (payload / 'rgb').mkdir(parents=True)
    pose = payload / 'pose.txt'
    pose.write_text('fixture pose')
    records = [SimpleNamespace(rgb_path=payload / 'rgb/0000.png')]
    config = {'target': {'directory': 'C3VDv2'}}
    result = m.governing_hashes(SimpleNamespace(records=records), 'target_test', tmp_path, config)
    assert result == {'C3VDv2/metadata.xlsx': m.file_digest(metadata),
                      pose.relative_to(tmp_path).as_posix(): m.file_digest(pose)}

    class RGBLabelBoundary:
        @property
        def parent(self):
            pytest.fail('C1 pose/depth path construction attempted')

    rgb_only = SimpleNamespace(records=[SimpleNamespace(rgb_path=RGBLabelBoundary())])
    for split in ('target_appearance_train', 'target_appearance_validation'):
        result = m.governing_hashes(rgb_only, split, tmp_path, config)
        assert result == {'C3VDv2/metadata.xlsx': m.file_digest(metadata)}
