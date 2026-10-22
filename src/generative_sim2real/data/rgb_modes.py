"""One source geometry and occurrence-local RGB randomness for R/A/G/M.

Generated RGB replaces source RGB at the 512 stage, before the paired 504
resize. Generated records bind the exact 512-stage label/mask/K bytes through
``source_geometry_sha256`` and the source row/transform policy fingerprint.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image
import torch

from .transforms import ResizeSample

AUGMENTATION = {'brightness': [.9, 1.1], 'color': [.9, 1.1], 'gamma': [.9, 1.1]}
GENERATED_PROVENANCE = ('base_model', 'base_revision', 'source_adapter', 'target_adapter',
                        'adapter_weight', 'controller', 'controller_scale', 'prompt',
                        'seed', 'scheduler', 'inference_steps', 'precision', 'size_hw')


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def file_sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def source_fingerprint(row, sidecar):
    """Portable source reference and shared transform contract; no machine paths."""
    return digest({'source_row': row, 'source_manifest_sha256': sidecar['manifest_sha256'],
                   'transform_definitions': sidecar['transform_definitions']})


def geometry_sha256(sample):
    """Exact generated-label identity at the 512-square substitution point."""
    value = hashlib.sha256()
    for tensor in (sample['targets']['depth_m'], sample['targets']['depth_valid'], sample['camera']['K']):
        if tensor is None:
            raise ValueError('Source labels and intrinsics are required')
        tensor = tensor.detach().cpu().contiguous()
        value.update(str(tensor.dtype).encode())
        value.update(str(tuple(tensor.shape)).encode())
        value.update(tensor.numpy().tobytes())
    value.update(json.dumps(sample['meta'].get('transform_history', []), sort_keys=True).encode())
    return value.hexdigest()


def load_matched_ids(path, source, *, smoke=False):
    artifact = json.loads(Path(path).read_text())
    if artifact.get('schema_version') != 1 or artifact.get('purpose') != ('smoke' if smoke else 'experiment'):
        raise ValueError('Matched IDs require explicit experiment or smoke artifact')
    if not smoke and artifact.get('frozen') is not True:
        raise ValueError('Production R/A/G/M require a frozen common ordered ID list')
    if artifact.get('source_manifest_sha256') != file_sha256(source.path) or artifact.get('source_sidecar_sha256') != file_sha256(source.path.with_suffix('.meta.json')):
        raise ValueError('Matched ID artifact does not bind current source manifest and sidecar')
    ids = artifact.get('sample_ids')
    known = {row['sample_id'] for row in source.rows}
    if not isinstance(ids, list) or not ids or len(ids) != len(set(ids)) or any(v not in known for v in ids):
        raise ValueError('Matched IDs must be unique nonempty source-training IDs')
    return ids


class RGBModeDataset:
    """Source-only wrapper; get_occurrence decouples appearance from sample order."""
    def __init__(self, source, sample_ids, *, mode='R', seed=0, generated_manifest=None, outputs_root=None):
        if source.split != 'source_train' or source.geometry != 'source_generation512':
            raise ValueError('RGB modes require source_train with source_generation512 geometry')
        if mode not in {'R', 'A', 'G', 'M'}:
            raise ValueError('RGB mode must be R, A, G, or M')
        self.source, self.mode, self.seed = source, mode, seed
        self.sample_ids = list(sample_ids)
        lookup = {r['sample_id']: i for i, r in enumerate(source.rows)}
        if not self.sample_ids or len(set(self.sample_ids)) != len(self.sample_ids) or any(v not in lookup for v in self.sample_ids):
            raise ValueError('Invalid matched source IDs')
        self.indices = [lookup[v] for v in self.sample_ids]
        self.generated = {}
        self.generated_path = None
        if mode in {'G', 'M'}:
            if generated_manifest is None or outputs_root is None:
                raise ValueError('G/M require a complete generated manifest and outputs root')
            self.generated_path = Path(generated_manifest)
            root = Path(outputs_root).resolve()
            records = [json.loads(line) for line in self.generated_path.read_text().splitlines()]
            if len(records) != len(self.sample_ids) or {r.get('sample_id') for r in records} != set(self.sample_ids):
                raise ValueError('Generated manifest must contain exactly all matched IDs once')
            first = records[0] if records else {}
            for record in records:
                sid = record['sample_id']
                row = source.rows[lookup[sid]]
                if record.get('source_sample_id') != sid or record.get('source_depth_path') != row['depth_path'] or record.get('source_transform_id') != 'source_generation512':
                    raise ValueError('Generated source identity/depth/transform mismatch')
                if record.get('preprocessing_sha256') != source_fingerprint(row, source.sidecar):
                    raise ValueError('Generated source preprocessing fingerprint mismatch')
                if not isinstance(record.get('source_geometry_sha256'), str) or len(record['source_geometry_sha256']) != 64:
                    raise ValueError('Generated geometry fingerprint missing')
                if any(record.get(key) is None for key in GENERATED_PROVENANCE) or record['size_hw'] != [512, 512]:
                    raise ValueError('Generated provenance missing, null, or not 512-square')
                # One recipe for every ID: only the per-ID seed may differ.
                if any(record[key] != first[key] for key in GENERATED_PROVENANCE if key != 'seed'):
                    raise ValueError('Generation provenance differs across records')
                relative = Path(record['generated_rgb_path'])
                path = (root / relative).resolve()
                if relative.is_absolute() or not path.is_relative_to(root) or not path.is_file():
                    raise ValueError('Generated RGB path escapes outputs root or is missing')
                if file_sha256(path) != record['generated_rgb_sha256']:
                    raise ValueError('Generated RGB checksum mismatch')
                self.generated[sid] = (record, path)
        elif generated_manifest is not None:
            raise ValueError('R/A must not accept a generated manifest')

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        return self.get_occurrence(index, index)

    def get_occurrence(self, index, occurrence):
        sample = self.source[self.indices[index]]
        sid = self.sample_ids[index]
        if sample['sample_id'] != sid or sample['rgb'].shape != (3, 512, 512):
            raise ValueError('Source sample identity/512 geometry mismatch')
        local_seed = int(digest({'seed': self.seed, 'sample_id': sid, 'occurrence': occurrence})[:16], 16)
        rng = np.random.default_rng(local_seed)
        replace = self.mode == 'G' or (self.mode == 'M' and rng.random() < .5)
        if self.mode in {'G', 'M'}:
            record, path = self.generated[sid]
            if geometry_sha256(sample) != record['source_geometry_sha256']:
                raise ValueError('Generated depth/mask/K/transform fingerprint mismatch')
        if replace:
            if file_sha256(path) != record['generated_rgb_sha256']:
                raise ValueError('Generated RGB changed after manifest validation')
            with Image.open(path) as image:
                if image.mode != 'RGB' or image.size != (512, 512):
                    raise ValueError('Generated image must be RGB 512-square')
                rgb = torch.from_numpy(np.array(image, dtype=np.float32).transpose(2, 0, 1).copy()) / 255.
            sample = {**sample, 'rgb': rgb}
        elif self.mode == 'A':
            rgb = sample['rgb']
            brightness = float(rng.uniform(*AUGMENTATION['brightness']))
            color = torch.tensor(rng.uniform(*AUGMENTATION['color'], size=3), dtype=rgb.dtype)[:, None, None]
            gamma = float(rng.uniform(*AUGMENTATION['gamma']))
            sample = {**sample, 'rgb': (rgb * brightness * color).clamp(0, 1).pow(gamma)}
        return ResizeSample((504, 504))(sample)


class OccurrenceOrder:
    """Infinite seeded permutations; cursor counts consumed source occurrences."""
    def __init__(self, size, seed):
        if size < 1:
            raise ValueError('Cannot sample an empty source')
        self.size, self.seed = size, seed
        self._epoch, self._order = None, None

    def at(self, cursor):
        if type(cursor) is not int or cursor < 0:
            raise ValueError('Occurrence cursor must be a nonnegative integer')
        epoch, offset = divmod(cursor, self.size)
        if epoch != self._epoch:
            # Hash (seed, epoch) so neighbouring seeds do not share shifted epoch permutations.
            generator = torch.Generator().manual_seed(int(digest({'order_seed': self.seed, 'epoch': epoch})[:15], 16))
            self._order = torch.randperm(self.size, generator=generator).tolist()
            self._epoch = epoch
        return self._order[offset]


def write_smoke_ids(manifest, count, output):
    """Write a clearly labeled smoke-only ID artifact; it is never accepted for experiments."""
    manifest = Path(manifest)
    rows = [json.loads(line) for line in manifest.read_text().splitlines()]
    sidecar = json.loads(manifest.with_suffix('.meta.json').read_text())
    if sidecar.get('split') != 'source_train' or not 1 <= count <= len(rows):
        raise ValueError('Smoke IDs require a source_train manifest and a valid count')
    # Evenly spread over the manifest so several trajectories are represented.
    picks = sorted({round(i * (len(rows) - 1) / max(1, count - 1)) for i in range(count)})
    artifact = {'schema_version': 1, 'purpose': 'smoke', 'frozen': False,
                'label': 'SMOKE ONLY: not the production downstream ID list',
                'source_manifest_sha256': file_sha256(manifest),
                'source_sidecar_sha256': file_sha256(manifest.with_suffix('.meta.json')),
                'sample_ids': [rows[i]['sample_id'] for i in picks]}
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(artifact, indent=2, allow_nan=False) + '\n')
    return artifact


def freeze_ids(manifest, count, output):
    """Freeze the production downstream ordered ID list (evenly spread over source_train)."""
    manifest = Path(manifest)
    output = Path(output)
    if output.exists():
        raise FileExistsError(f'{output} is frozen; changing N or IDs is a protocol change')
    artifact = write_smoke_ids(manifest, count, output)
    if len(artifact['sample_ids']) != count:
        output.unlink()
        raise ValueError('Even spreading did not yield exactly the requested count')
    artifact.update(purpose='experiment', frozen=True, label='Frozen common ordered downstream IDs for R/A/G/M')
    output.write_text(json.dumps(artifact, indent=2, allow_nan=False) + '\n')
    return artifact


def main():
    parser = argparse.ArgumentParser(description=write_smoke_ids.__doc__)
    parser.add_argument('--manifest', default='outputs/manifests/milestone_v1/source_train.jsonl')
    parser.add_argument('--count', type=int, required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--freeze', action='store_true', help='write the frozen production list instead of a smoke list')
    args = parser.parse_args()
    write = freeze_ids if args.freeze else write_smoke_ids
    print(len(write(args.manifest, args.count, args.output)['sample_ids']), 'frozen IDs written' if args.freeze else 'smoke IDs written')


if __name__ == '__main__':
    main()
