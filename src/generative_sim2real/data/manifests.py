"""Deterministic P1 manifests and explicit audit-only C2 data access.

Run ``python -m generative_sim2real.data.manifests --config configs/data.yaml``.
No models or GPU operations are imported by this command.
"""
import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import tempfile

import numpy as np
from PIL import Image, ImageDraw
from torch.utils.data import Dataset
import yaml

from .c3vdv2 import C3VDv2, C3VDv2RGB
from .simcol3d import SimCol3D
from .transforms import CenterCropSquare, ResizeSample
from .preprocessing import (
    InverseDepthEncoding, depth_audit, fit_inverse_depth, uniform_indices,
    valid_depth_pixels,
)

SPLITS = ('source_train', 'source_validation', 'target_appearance_train',
          'target_appearance_validation', 'target_test')
SOURCE_ENCODING = 'simcol3d_uint16_png_div_65280_times_0.20m'
TARGET_ENCODING = 'c3vdv2_uint16_tiff_div_65535_times_0.1m'


def json_bytes(value) -> bytes:
    return (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + '\n').encode()


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def file_digest(path: Path) -> str:
    return digest(path.read_bytes())


def atomic_write(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix='.' + path.name, dir=path.parent)
    try:
        with os.fdopen(fd, 'wb') as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(name, path)
    finally:
        if os.path.exists(name):
            os.unlink(name)


def safe_relative(value: str) -> str:
    path = PurePosixPath(value)
    if (not value or path.is_absolute() or '..' in path.parts
            or '\\' in value or path.as_posix() != value or value == '.'):
        raise ValueError(f'Unsafe or noncanonical relative path: {value!r}')
    return value


def data_path(root: Path, value: str, target_directory: str) -> Path:
    """Permit only the explicitly configured C3VD frames/metadata view links."""
    path = root / safe_relative(value)
    logical = path.absolute()
    if not logical.is_relative_to(root.absolute()):
        raise ValueError('Path escapes DATA_ROOT')
    current = root.absolute()
    permitted = {root / target_directory / 'frames',
                 root / target_directory / 'metadata.xlsx'}
    for component in logical.relative_to(root.absolute()).parts:
        current = current / component
        if current.is_symlink() and current not in permitted:
            raise ValueError(f'Unconfigured data symlink: {current}')
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def relative(path: Path, root: Path) -> str:
    try:
        value = path.absolute().relative_to(root.absolute()).as_posix()
    except ValueError as error:
        raise ValueError(f'Adapter path does not preserve DATA_ROOT view: {path}') from error
    return safe_relative(value)


def validate_config(config: dict) -> None:
    if config.get('schema_version') != 1:
        raise ValueError('Only manifest schema version 1 is supported')
    for section in ('source', 'target'):
        safe_relative(config[section]['directory'])
    target = config['target']
    if not target['versions'] or set(target['versions']) - {'v1', 'v2'}:
        raise ValueError('This milestone permits only target v1/v2')
    if set(target['quality_scores']) - {1, 2}:
        raise ValueError('This milestone permits only quality 1-2')
    if not config['source']['validation_sequences'] or not target['validation_families']:
        raise ValueError('Validation holdouts must be explicit')
    for part in ('train', 'validation'):
        for domain in ('source', 'target'):
            for suffix in ('stride', 'cap'):
                value = config[domain][f'{part}_{suffix}']
                if type(value) is not int or value < 1:
                    raise ValueError('Strides and caps must be positive integers')
    if target['test_stride'] != 5 or target['rectified_hw'] != [504, 630]:
        raise ValueError('C2 protocol requires stride 5 and full 504x630 view')
    if config['generation_size'] != 512:
        raise ValueError('Milestone generation geometry is fixed to 512 square')


def build_view(split: str, config: dict, root: Path):
    """The sole dataset dispatch; training callers never request target_test."""
    if split not in SPLITS:
        raise ValueError(f'Unknown manifest role: {split}')
    source, target = config['source'], config['target']
    if split.startswith('source_'):
        validation = split == 'source_validation'
        options = {'split': 'train', 'frame_stride': source[
            'validation_stride' if validation else 'train_stride'], 'show_progress': False}
        options['include_sequences' if validation else 'exclude_sequences'] = tuple(
            source['validation_sequences'])
        return SimCol3D(root / source['directory'], **options)
    options = {
        'split': 'test' if split == 'target_test' else 'train',
        'colons': ('c2',) if split == 'target_test' else ('c1',),
        'textures': tuple(target['textures']), 'video_versions': tuple(target['versions']),
        'quality_scores': tuple(target['quality_scores']),
        'allow_v3_clean_proxy_depth': False,
        'rectified_hw': tuple(target['rectified_hw']),
        'rectified_focal_px': target['focal_px'], 'center_crop_square': False,
        'frame_stride': target['test_stride'] if split == 'target_test' else target[
            'validation_stride' if split.endswith('validation') else 'train_stride'],
        'show_progress': False,
    }
    constructor = C3VDv2 if split == 'target_test' else C3VDv2RGB
    return constructor(root / target['directory'], **options)


def selected_indices(dataset, split: str, config: dict) -> list[int]:
    indices = list(range(len(dataset.records)))
    if split.startswith('target_appearance'):
        validation = split.endswith('validation')
        held = set(config['target']['validation_families'])
        indices = [
            i for i in indices
            if ('_'.join(dataset.records[i].sequence_id.split('_')[1:-1]) in held)
            == validation
        ]
    if split == 'target_test':
        return indices
    domain = 'source' if split.startswith('source') else 'target'
    part = 'validation' if split.endswith('validation') else 'train'
    cap = config[domain][part + '_cap']
    return [indices[i] for i in uniform_indices(len(indices), cap)]


def transform_id(split: str) -> str:
    return {'source_train': 'source_generation512',
            'source_validation': 'source_native', 'target_test': 'target_rectified_full',
            'target_appearance_train': 'target_generation512',
            'target_appearance_validation': 'target_generation512'}[split]


def transform_definitions(config: dict) -> dict:
    return {
        'source_native': {'adapter_hw': [475, 475], 'operations': [],
                          'K': 'scene cam.txt, in pixels'},
        'source_generation512': {'adapter_hw': [475, 475],
            'operations': [{'name': 'resize', 'size_hw': [512, 512]}],
            'K': 'scale first two rows of cam.txt K by 512/475'},
        'source_da3_504': {'adapter_hw': [475, 475],
            'operations': [{'name': 'resize', 'size_hw': [512, 512]},
                           {'name': 'resize', 'size_hw': [504, 504]}],
            'K': 'scale first two rows successively by 512/475 and 504/512'},
        'target_rectified_full': {'adapter_hw': config['target']['rectified_hw'],
            'operations': [], 'K': [[config['target']['focal_px'], 0, 314.5],
                                  [0, config['target']['focal_px'], 251.5], [0, 0, 1]],
            'rectification': 'official Scaramuzza calibration in c3vdv2.py'},
        'target_generation512': {'adapter_hw': config['target']['rectified_hw'],
            'operations': [{'name': 'center_crop_square', 'top': 0, 'left': 63,
                            'size_hw': [504, 504]},
                           {'name': 'resize', 'size_hw': [512, 512]}],
            'K': ('full rectified K: subtract crop origin from principal point, '
                  'then scale rows by 512/504')},
        'interpolation': {'rgb': 'OpenCV cubic, clipped [0,1]',
                          'depth_m': 'nearest', 'depth_valid': 'nearest'},
    }


def apply_geometry(sample: dict, identity: str) -> dict:
    if identity == 'source_generation512':
        return ResizeSample((512, 512))(sample)
    if identity == 'source_da3_504':
        return ResizeSample((504, 504))(ResizeSample((512, 512))(sample))
    if identity == 'target_generation512':
        return ResizeSample((512, 512))(CenterCropSquare()(sample))
    if identity in {'source_native', 'target_rectified_full'}:
        return sample
    raise ValueError(f'Unknown geometry: {identity}')


def make_row(record, split: str, root: Path, config: dict) -> dict:
    source = split.startswith('source')
    row = {'sample_id': record.sample_id, 'dataset': 'simcol3d' if source else 'c3vdv2',
           'sequence_id': record.sequence_key if source else record.sequence_id,
           'frame_index': record.frame_index, 'split': split,
           'rgb_path': relative(record.rgb_path, root),
           'transform_id': transform_id(split), 'caption': config['caption']}
    if source or split == 'target_test':
        row.update(depth_path=relative(record.depth_path, root),
                   depth_encoding=SOURCE_ENCODING if source else TARGET_ENCODING,
                   depth_definition='camera_z',
                   validity_policy=(
                       'raw_gt_zero' if source else 'raw_gt_zero_not_65535_and_remap_inside'))
    if source:
        row['intrinsics_path'] = relative(record.intrinsics_path, root)
        row['position_path'] = relative(record.position_path, root)
        row['rotation_path'] = relative(record.rotation_path, root)
    else:
        row['calibration_reference'] = 'c3vdv2.py:official_olympus_cf_hq190l_ocam'
    return row


def module_hashes() -> dict:
    directory = Path(__file__).parent
    return {name: file_digest(directory / name) for name in
            ('simcol3d.py', 'c3vdv2.py', 'transforms.py', 'manifests.py', 'preprocessing.py')}


def governing_hashes(dataset, split: str, root: Path, config: dict) -> dict:
    if split.startswith('source'):
        paths = {root / config['source']['directory'] / 'misc' / f'{part}_file.txt'
                 for part in ('train', 'test')}
        # Include the untouched official test split in source provenance.
        for split_path in tuple(paths):
            for line in split_path.read_text().splitlines():
                entry = line.strip().rstrip('/')
                if not entry:
                    continue
                safe_relative(entry)
                scene, sequence = entry.split('/')
                scene_dir = root / config['source']['directory'] / scene
                trajectory = sequence.removeprefix('Frames_')
                paths.update((scene_dir / 'cam.txt',
                              scene_dir / f'SavedPosition_{trajectory}.txt',
                              scene_dir / f'SavedRotationQuaternion_{trajectory}.txt'))
    else:
        paths = {root / config['target']['directory'] / 'metadata.xlsx'}
        if split == 'target_test':
            # The indexed RGB parent identifies the actual direct/nested payload.
            # Appearance views must never construct or inspect these label paths.
            paths.update(record.rgb_path.parent.parent / 'pose.txt'
                         for record in dataset.records)
    return {relative(path, root): file_digest(path) for path in sorted(paths)}


def manifest_bytes(rows: list[dict]) -> bytes:
    return b''.join((json.dumps(row, sort_keys=True, allow_nan=False) + '\n').encode()
                    for row in rows)


class ManifestDataset(Dataset):
    """Validate role, hashes and exact adapter selection before decoding a frame."""
    def __init__(
        self, path: str | Path, data_root: str | Path,
        purpose: str = 'training', geometry: str | None = None,
    ):
        if purpose not in {'training', 'audit', 'evaluation'}:
            raise ValueError('purpose must be training, audit, or evaluation')
        self.path, self.root = Path(path), Path(data_root).expanduser().absolute()
        self.sidecar = json.loads(self.path.with_suffix('.meta.json').read_text())
        side = self.sidecar
        if side.get('schema_version') != 1 or side.get('split') not in SPLITS:
            raise ValueError('Invalid manifest schema or role')
        self.split = side['split']
        if self.split == 'target_test' and purpose == 'training':
            raise ValueError('target_test is confined to explicit audit/evaluation purposes')
        config = side['config']
        validate_config(config)
        if digest(json_bytes(config)) != side['config_sha256']:
            raise ValueError('Configuration hash mismatch')
        content = self.path.read_bytes()
        if digest(content) != side['manifest_sha256']:
            raise ValueError('Manifest hash mismatch')
        if side['source_code_sha256'] != module_hashes():
            raise ValueError('Adapter/preprocessing source code hash mismatch; rebuild manifests')
        if side['transform_definitions'] != transform_definitions(config):
            raise ValueError('Transform definition mismatch')
        control = side['depth_control_encoding']
        InverseDepthEncoding.from_dict(control['constants'])
        if (control['polarity'] != 'near_is_bright' or control['metric_labels_modified']
                or control['channels'] != 3 or control['dtype'] != 'float32'):
            raise ValueError('Invalid source-fitted controller encoding')
        self.rows = [json.loads(line) for line in content.splitlines()]
        if not self.rows or len(self.rows) != side['count']:
            raise ValueError('Empty manifest or count mismatch')
        for row in self.rows:
            if row.get('split') != self.split:
                raise ValueError('Record role mismatch')
            if self.split.startswith('target_appearance') and any(
                    token in key for key in row
                    for token in ('depth', 'pose', 'mask', 'position', 'rotation')):
                raise ValueError('RGB appearance manifest contains label references')
            for key, value in row.items():
                if key.endswith('_path'):
                    data_path(self.root, value, config['target']['directory'])
        for value, expected in side['governing_files_sha256'].items():
            if file_digest(data_path(self.root, value, config['target']['directory'])) != expected:
                raise ValueError(f'Governing data file hash mismatch: {value}')
        self.dataset = build_view(self.split, config, self.root)
        self.indices = selected_indices(self.dataset, self.split, config)
        expected = [make_row(self.dataset.records[i], self.split, self.root, config)
                    for i in self.indices]
        if self.rows != expected:
            raise ValueError(
                'Manifest selection, paths, encoding or geometry differ from adapter records')
        per_sequence = dict(sorted(Counter(row['sequence_id'] for row in self.rows).items()))
        if per_sequence != side['per_sequence']:
            raise ValueError('Per-sequence counts mismatch')
        governing_files = governing_hashes(self.dataset, self.split, self.root, config)
        if governing_files != side['governing_files_sha256']:
            raise ValueError('Governing data references mismatch')
        self.geometry = geometry or transform_id(self.split)
        allowed = {transform_id(self.split)}
        if self.split.startswith('source'):
            allowed.update(('source_generation512', 'source_da3_504', 'source_native'))
        if self.geometry not in allowed:
            raise ValueError('Requested geometry incompatible with manifest role')

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        sample = self.dataset[self.indices[index]]
        sample = apply_geometry(sample, self.geometry)
        sample['meta'] = {**sample['meta'], 'split': self.split,
                          'manifest_transform_id': self.geometry}
        return sample

    def raw_sample(self, index):
        """Adapter-native geometry, for source fit and explicitly requested audits."""
        return self.dataset[self.indices[index]]


def check_splits(rows_by_split: dict, official_test_ids: set[str]) -> dict:
    all_ids = set()
    for split, rows in rows_by_split.items():
        ids = [row['sample_id'] for row in rows]
        if len(ids) != len(set(ids)) or all_ids.intersection(ids):
            raise ValueError(f'Duplicate or overlapping sample IDs in {split}')
        all_ids.update(ids)
    source_ids = {row['sample_id'] for split in ('source_train', 'source_validation')
                  for row in rows_by_split[split]}
    if source_ids.intersection(official_test_ids):
        raise ValueError('Official source test contamination')
    train_seq = {row['sequence_id'] for row in rows_by_split['source_train']}
    val_seq = {row['sequence_id'] for row in rows_by_split['source_validation']}
    if train_seq.intersection(val_seq):
        raise ValueError('Source validation trajectory leakage')
    def families(split):
        return {'_'.join(row['sequence_id'].split('_')[1:-1]) for row in rows_by_split[split]}
    if families('target_appearance_train') & families('target_appearance_validation'):
        raise ValueError('Target C1 family leakage')
    return {'sample_ids_disjoint': True, 'source_trajectories_disjoint': True,
            'c1_families_disjoint': True, 'official_source_test_overlap': 0,
            'official_source_test_count': len(official_test_ids)}


def diagnostic_selection(
    dataset: ManifestDataset, conditions: int, seeds: list[int]
) -> dict:
    """Balance two validation scenes and pair opposite depth ranks across scenes."""
    if type(conditions) is not int or conditions < 2 or conditions % 2:
        raise ValueError('Cross-scene diagnostics require a positive even condition count')
    scenes = {}
    seen_ids = set()
    for i, row in enumerate(dataset.rows):
        if row.get('split') != 'source_validation' or row.get('dataset') != 'simcol3d':
            raise ValueError('Diagnostics may only select labeled source_validation frames')
        sample_id = row['sample_id']
        if sample_id in seen_ids:
            raise ValueError('Diagnostic source-validation IDs must be unique')
        seen_ids.add(sample_id)
        scene, separator, _ = row['sequence_id'].partition('/')
        if not separator or not scene:
            raise ValueError('Diagnostic source scene requires a qualified sequence_id')
        values = valid_depth_pixels(dataset.raw_sample(i))
        if not len(values):
            raise ValueError(f'Validation diagnostic frame has no valid depth: {sample_id}')
        scenes.setdefault(scene, []).append((float(np.median(values)), sample_id))
    if len(scenes) != 2:
        raise ValueError('Cross-scene diagnostics require exactly two validation scenes')
    half = conditions // 2
    scene_ids = sorted(scenes)
    selected = []
    for scene in scene_ids:
        ordered = sorted(scenes[scene])
        if len(ordered) < half:
            raise ValueError(f'Insufficient source-validation frames in scene {scene}')
        selected.append([ordered[i] for i in uniform_indices(len(ordered), half)])

    def paired_selection(groups):
        ids = [sample_id for group in groups for _, sample_id in group]
        group_size = len(groups[0])
        permutation = (list(range(2 * group_size - 1, group_size - 1, -1))
                       + list(range(group_size - 1, -1, -1)))
        wrong_ids = [ids[i] for i in permutation]
        group_scenes = [scene for scene in scene_ids for _ in range(group_size)]
        if (len(ids) != len(set(ids)) or set(ids) != set(wrong_ids)
                or any(ids[i] == wrong_ids[i]
                       or group_scenes[i] == group_scenes[permutation[i]]
                       for i in range(len(ids)))):
            raise ValueError('Cross-scene diagnostic permutation is not a valid bijection')
        return ids, permutation, wrong_ids, group_scenes

    ids, permutation, wrong_ids, group_scenes = paired_selection(selected)
    reduced_half = min(4, half)
    reduced = [
        [group[i] for i in uniform_indices(len(group), reduced_half)]
        for group in selected
    ]
    reduced_ids, reduced_permutation, reduced_wrong_ids, reduced_scenes = paired_selection(
        reduced)
    return {
        'selection': ('equal counts from two source-validation scenes; uniform integer '
                      'endpoints of median-depth ranks within each scene'),
        'scene_order': scene_ids, 'conditions_per_scene': half,
        'sample_ids': ids, 'scene_ids': group_scenes,
        'median_depth_m': [value for group in selected for value, _ in group],
        'seeds': seeds,
        'wrong_depth_rule': 'bijective cross-scene reversed median-depth rank pairing',
        'wrong_depth_sample_ids': wrong_ids, 'wrong_depth_permutation': permutation,
        'reduced_gate_sample_ids': reduced_ids, 'reduced_gate_scene_ids': reduced_scenes,
        'reduced_gate_wrong_depth_sample_ids': reduced_wrong_ids,
        'reduced_gate_wrong_depth_permutation': reduced_permutation,
        'reduced_gate_seed': seeds[0], 'downstream_N': None,
        'downstream_N_status': 'freeze ordered source_train IDs only after P3 throughput gate',
    }


def panel_content_geometry(
    input_hw: tuple[int, int], tile_hw: tuple[int, int] = (240, 320)
) -> tuple[tuple[int, int], tuple[int, int]]:
    """Return one aspect-preserving content size and center-padding origin."""
    input_h, input_w = input_hw
    tile_h, tile_w = tile_hw
    scale = min(tile_h / input_h, tile_w / input_w)
    content_hw = (max(1, round(input_h * scale)), max(1, round(input_w * scale)))
    top_left = ((tile_h - content_hw[0]) // 2, (tile_w - content_hw[1]) // 2)
    return content_hw, top_left


def panel_picture(
    picture: Image.Image, content_hw: tuple[int, int], top_left: tuple[int, int],
    resampling: Image.Resampling,
) -> Image.Image:
    """Use the sample's shared geometry for RGB and nearest-resampled labels."""
    height, width = content_hw
    top, left = top_left
    tile = Image.new('RGB', (320, 240), '#252525')
    tile.paste(picture.resize((width, height), resampling), (left, top))
    return tile


def smoke_panel(samples: list[dict], path: Path, encoding: InverseDepthEncoding) -> None:
    """Display metric depth in a common 0..far scale, with invalid pixels black."""
    tiles, width, height = [], 320, 280
    for sample in samples:
        rgb = (sample['rgb'].permute(1, 2, 0).numpy() * 255).clip(0, 255).astype('uint8')
        content_hw, top_left = panel_content_geometry(sample['rgb'].shape[-2:])
        rgb_image = panel_picture(
            Image.fromarray(rgb), content_hw, top_left, Image.Resampling.BICUBIC)
        depth = sample['targets']['depth_m']
        images = [rgb_image]
        labels = ['RGB']
        if depth is not None:
            valid = sample['targets']['depth_valid'].numpy()[0]
            values = depth.numpy()[0]
            shade = (np.clip(values / encoding.far_m, 0, 1) * 255).astype('uint8')
            colors = np.stack((shade, shade, shade), axis=-1)
            colors[~valid] = 0
            images.append(panel_picture(
                Image.fromarray(colors), content_hw, top_left, Image.Resampling.NEAREST))
            labels.append(f'camera-z 0..{encoding.far_m * 1000:.1f} mm; invalid black')
            control = encoding.encode(depth, sample['targets']['depth_valid'])[0].numpy()
            control_image = Image.fromarray((control * 255).astype('uint8')).convert('RGB')
            images.append(panel_picture(
                control_image, content_hw, top_left, Image.Resampling.NEAREST))
            labels.append('source-fitted inverse depth; near bright')
        else:
            labels[0] += '; appearance labels unavailable'
        tile = Image.new('RGB', (3 * width, height), '#252525')
        draw = ImageDraw.Draw(tile)
        draw.text((4, 1), sample['sample_id'], fill='white')
        for j, (picture, label) in enumerate(zip(images, labels)):
            tile.paste(picture, (j * width, 20))
            draw.text((j * width + 4, 262), label, fill='white')
        tiles.append(tile)
    panel = Image.new('RGB', (3 * width, height * len(tiles)))
    for i, tile in enumerate(tiles):
        panel.paste(tile, (0, i * height))
    import io
    buffer = io.BytesIO()
    panel.save(buffer, format='PNG')
    atomic_write(path, buffer.getvalue())


def build_manifests(config_path: Path, root: Path) -> dict:
    config = yaml.safe_load(config_path.read_text())
    validate_config(config)
    root = root.expanduser().absolute()
    output = Path(config['output_directory'])
    safe_relative(output.as_posix())
    datasets, rows_by_split, metas = {}, {}, {}
    for split in SPLITS:
        inventory = ('RGB-only inventory' if split.startswith('target_appearance')
                     else 'full file-pair inventory')
        print(f'Indexing {split} ({inventory}; no image decoding)', flush=True)
        dataset = build_view(split, config, root)
        indices = selected_indices(dataset, split, config)
        rows = [make_row(dataset.records[i], split, root, config) for i in indices]
        if not rows:
            raise ValueError(f'Empty selection for {split}')
        for row in rows:
            for key, value in row.items():
                if key.endswith('_path'):
                    data_path(root, value, config['target']['directory'])
        datasets[split], rows_by_split[split] = dataset, rows
        content = manifest_bytes(rows)
        metas[split] = {
            'schema_version': 1, 'split': split, 'config': config,
            'config_sha256': digest(json_bytes(config)),
            'config_file_sha256': file_digest(config_path),
            'manifest_sha256': digest(content), 'count': len(rows),
            'per_sequence': dict(sorted(Counter(row['sequence_id'] for row in rows).items())),
            'indexed_frames_before_family_filter_and_cap': len(dataset.records),
            'transform_definitions': transform_definitions(config),
            'source_code_sha256': module_hashes(),
            'governing_files_sha256': governing_hashes(dataset, split, root, config),
            'selection': ('per-sequence stride from frame zero, stable adapter ordering, '
                          'uniform integer endpoint cap'),
            'selection_seed': None,
            'integrity_scope': ('JSONL, config, adapter/preprocessing code, '
                                'metadata/calibration/pose/split files; image contents not hashed'),
            'depth_encodings': {SOURCE_ENCODING: 'uint16/65280*0.20 meters; zero invalid',
                                TARGET_ENCODING: 'uint16/65535*0.1 meters; zero and 65535 invalid'},
        }
    official_test = SimCol3D(root / config['source']['directory'], split='test',
                            show_progress=False)
    checks = check_splits(rows_by_split, {r.sample_id for r in official_test.records})
    depth_config = config['depth_encoding']
    train = datasets['source_train']
    train_rows = rows_by_split['source_train']
    train_indices = selected_indices(train, 'source_train', config)
    fit_indices = uniform_indices(len(train_rows), depth_config['fit_frames'])
    encoding, fit = fit_inverse_depth(
        ((train_rows[i], train[train_indices[i]]) for i in fit_indices),
        near_quantile=depth_config['near_quantile'], far_quantile=depth_config['far_quantile'],
        pixels_per_frame=depth_config['pixels_per_frame'], max_frames=depth_config['fit_frames'])
    if depth_config['invalid_fill'] != encoding.invalid_fill:
        raise ValueError('Configured invalid fill conflicts with controller contract')
    for sidecar in metas.values():
        sidecar['depth_control_encoding'] = fit
    for split in SPLITS:
        atomic_write(output / f'{split}.jsonl', manifest_bytes(rows_by_split[split]))
        atomic_write(output / f'{split}.meta.json', json_bytes(metas[split]))
    loaded = {split: ManifestDataset(output / f'{split}.jsonl', root, purpose='audit')
              for split in SPLITS}
    print('Decoding bounded source fit, validation geometry and C2 smoke samples', flush=True)
    smoke_count = config['smoke']['frames_per_split']
    statistics = {}
    for split, dataset in loaded.items():
        indices = uniform_indices(len(dataset), smoke_count)
        if split == 'source_train':
            audit_indices = fit_indices
        else:
            audit_indices = indices
        if split.startswith('target_appearance'):
            statistics[split] = {
                'depth_statistics': 'n/a: RGB-only C1; labels never indexed or decoded'}
        else:
            statistics[split] = depth_audit(
                (dataset.raw_sample(i) for i in audit_indices), encoding)
            statistics[split]['sample_ids'] = [dataset.rows[i]['sample_id'] for i in audit_indices]
        smoke_panel([dataset[i] for i in indices], output / f'{split}_smoke.png', encoding)
    diagnostics = diagnostic_selection(loaded['source_validation'],
                                       config['diagnostics']['conditions'],
                                       config['diagnostics']['seeds'])
    atomic_write(output / 'depth_encoding.json', json_bytes(fit))
    atomic_write(output / 'diagnostics.json', json_bytes(diagnostics))
    report = {'schema_version': 1, 'split_checks': checks,
              'counts': {split: len(rows) for split, rows in rows_by_split.items()},
              'depth_audits': statistics,
              'validation_diagnostic_geometry_frames': len(loaded['source_validation']),
              'verification_scope': ('full indexed RGB/depth pairing and governing-file audit; '
                                     'bounded decoded fit/smoke, all capped validation diagnostic '
                                     'depths; no full image integrity or model evaluation'),
              'artifacts_sha256': {p.name: file_digest(p) for p in sorted(output.iterdir())
                                   if p.is_file() and p.name != 'audit.json'}}
    atomic_write(output / 'audit.json', json_bytes(report))
    print(json.dumps(report['counts'], sort_keys=True), flush=True)
    print(f'Fixed inverse-depth range: {encoding.near_m:.9f}..{encoding.far_m:.9f} m', flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=Path('configs/data.yaml'))
    parser.add_argument('--check', action='store_true',
                        help='verify hashes and bounded reload without rebuilding')
    args = parser.parse_args()
    if not os.environ.get('DATA_ROOT'):
        parser.error('DATA_ROOT must be set (scripts/prepare_data.sh sources .env)')
    root = Path(os.environ['DATA_ROOT']).expanduser().absolute()
    if not args.check:
        build_manifests(args.config, root)
        return
    config = yaml.safe_load(args.config.read_text())
    output = Path(config['output_directory'])
    audit = json.loads((output / 'audit.json').read_text())
    for name, expected in audit['artifacts_sha256'].items():
        if file_digest(output / safe_relative(name)) != expected:
            raise ValueError(f'Artifact hash mismatch: {name}')
    for split in SPLITS:
        dataset = ManifestDataset(output / f'{split}.jsonl', root, purpose='audit')
        if dataset.sidecar['config_file_sha256'] != file_digest(args.config):
            raise ValueError('Current config file differs from frozen config')
        for index in uniform_indices(len(dataset), config['smoke']['frames_per_split']):
            dataset[index]
        print(f'{split}: {len(dataset)} records; hashes and bounded reload passed', flush=True)


if __name__ == '__main__':
    main()
