"""Build "one fixed image per ID" appearance-control sets as G-style generated manifests.

Variants:
  T_reinhard  Reinhard Lab colour transfer of each SimCol frame to one seeded C1-train RGB frame
  A_strong    strong generic photometric augmentation, one fixed draw per ID (no target data)
  F_fda_b01 / F_fda_b05  Fourier Domain Adaptation (low-frequency amplitude swap, beta 0.01 / 0.05) to the SAME
              seeded C1-train frame that T_reinhard uses for each ID (reference chosen from T_reinhard's seed)

The output is consumed unchanged by ``MODE=G GENERATED_MANIFEST=... bash scripts/train_da3.sh``: records follow
generation/bulk.py ``collect_records`` and are validated by data/rgb_modes.py ``RGBModeDataset``. Only source
RGB/labels (source_train at the 512 stage) and C1 appearance-train RGB are read; no C1 depth/pose and no C2.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing
import os
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
import torch

from ..data import rgb_modes as rm
from ..data.manifests import ManifestDataset
from . import color

SOURCE_TRAIN = 'outputs/manifests/milestone_v1/source_train.jsonl'
SOURCE_VALIDATION = 'outputs/manifests/milestone_v1/source_validation.jsonl'
TARGET_TRAIN = 'outputs/manifests/milestone_v1/target_appearance_train.jsonl'
IDS = 'outputs/manifests/milestone_v1/downstream_ids.json'

# Provenance describing the variant honestly. Every GENERATED_PROVENANCE key must be non-null and identical
# across records except 'seed'; 'none'/0 mean "not applicable", never a hidden generator.
PROVENANCE = {
    'T_reinhard': {'base_model': 'none', 'base_revision': 'none', 'source_adapter': 'none',
                   'target_adapter': 'reinhard_lab:c1_train', 'adapter_weight': 0.0, 'controller': 'none',
                   'controller_scale': 0.0, 'prompt': 'none', 'scheduler': 'none', 'inference_steps': 0,
                   'precision': 'float32', 'size_hw': [512, 512]},
    'A_strong': {'base_model': 'none', 'base_revision': 'none', 'source_adapter': 'none',
                 'target_adapter': 'photometric_strong_v1', 'adapter_weight': 0.0, 'controller': 'none',
                 'controller_scale': 0.0, 'prompt': 'none', 'scheduler': 'none', 'inference_steps': 0,
                 'precision': 'float32', 'size_hw': [512, 512]},
}
FDA_BETA = {'F_fda_b01': .01, 'F_fda_b05': .05}
for _name, _beta in FDA_BETA.items():
    PROVENANCE[_name] = {**PROVENANCE['T_reinhard'], 'target_adapter': f'fda_lowfreq_amplitude_beta{_beta}:c1_train'}
RECIPE_DEFINITION = {
    'T_reinhard': {'transfer': 'reinhard_lab_mean_std', 'border_luminance': color.BORDER_LUMINANCE,
                   'reference_pool': 'target_appearance_train (RGB only, target_generation512 view)',
                   'reference_choice': 'seeded rng.integers per ID'},
    'A_strong': {'ranges': color.STRONG_RANGES, 'blur_probability': color.BLUR_PROBABILITY,
                 'order': 'brightness,contrast,gamma,saturation,hue,channel_gain,blur,noise'},
    **{name: {'transfer': 'fda_low_frequency_amplitude_swap', 'beta': beta, 'phase': 'source',
              'reference_pool': 'target_appearance_train (RGB only, target_generation512 view)',
              'reference_choice': "T_reinhard's seeded reference per ID (same C1 frame as T)"}
       for name, beta in FDA_BETA.items()}}
TARGET_VARIANTS = ('T_reinhard', *FDA_BETA)


def variant_seed(name, sample_id):
    """Deterministic per-(variant, ID) seed from a hash; independent of ID order and process count."""
    return int(rm.digest({'variant': name, 'sample_id': sample_id})[:15], 16)


def recipe_hash(name):
    return rm.digest({'name': name, 'provenance': PROVENANCE[name], 'definition': RECIPE_DEFINITION[name]})[:16]


def make_image(name, sample, seed, reference_rgb=None):
    """-> (float32 [3,512,512] image in [0,1], extra record fields). Pure given its inputs."""
    rng = np.random.default_rng(seed)
    if name == 'T_reinhard':
        # Reference choice is made by the caller (it needs the pool size); rng here is unused for T.
        source_mask = color.luminance_mask(sample['rgb'], sample['targets']['depth_valid'])
        reference_mask = color.luminance_mask(reference_rgb)
        return color.reinhard_lab(sample['rgb'], reference_rgb, source_mask, reference_mask), {}
    if name in FDA_BETA:
        return color.fda_transfer(sample['rgb'], reference_rgb, FDA_BETA[name]), {}
    if name == 'A_strong':
        params = color.draw_strong_params(rng)
        return color.apply_strong_photometric(sample['rgb'], params), {'photometric_params': params}
    raise ValueError(f'Unknown variant {name!r}')


def reference_index(seed, pool_size):
    """One C1-train frame per ID: an rng stream separate from the variant stream, fixed by the seed."""
    return int(np.random.default_rng(int(rm.digest({'reference_for_seed': seed})[:15], 16)).integers(pool_size))


def png_sha256(tensor, path):
    array = (tensor.clamp(0, 1) * 255).round().byte().permute(1, 2, 0).numpy()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f'.tmp-{os.getpid()}-{path.name}')
    Image.fromarray(array).save(temporary, format='PNG')
    os.replace(temporary, path)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def make_record(name, sample_id, source_row, source_sidecar, geometry, seed, relative_path, image_sha, extra):
    """Same fields as generation/bulk.py collect_records, plus variant-describing extras."""
    return {'sample_id': sample_id, 'source_sample_id': sample_id, 'source_depth_path': source_row['depth_path'],
            'source_transform_id': 'source_generation512',
            'preprocessing_sha256': rm.source_fingerprint(source_row, source_sidecar),
            'source_geometry_sha256': geometry, 'generated_rgb_path': relative_path,
            'generated_rgb_sha256': image_sha, **PROVENANCE[name], 'seed': seed,
            'recipe_hash': recipe_hash(name), 'variant': name, **extra}


_STATE = {}


def _init_worker(name, data_root, source_path, target_path, out_dir, outputs_root):
    torch.set_num_threads(1)
    cv2.setNumThreads(1)
    _STATE['source'] = ManifestDataset(source_path, data_root, purpose='training', geometry='source_generation512')
    _STATE['lookup'] = {r['sample_id']: i for i, r in enumerate(_STATE['source'].rows)}
    if name in TARGET_VARIANTS:
        _STATE['target'] = ManifestDataset(target_path, data_root, purpose='training')  # RGB-only role, no labels
    _STATE.update(name=name, out_dir=Path(out_dir), outputs_root=Path(outputs_root))


def _work(job):
    position, sid = job
    name, source = _STATE['name'], _STATE['source']
    index = _STATE['lookup'][sid]
    sample = source[index]
    seed = variant_seed(name, sid)
    reference, extra = None, {}
    if name in TARGET_VARIANTS:
        # FDA variants reuse T_reinhard's reference for the same ID, so T and F differ only in the transfer.
        ref_index = reference_index(variant_seed('T_reinhard', sid), len(_STATE['target']))
        reference = _STATE['target'][ref_index]['rgb']
        extra = {'reference_sample_id': _STATE['target'].rows[ref_index]['sample_id']}
    image, more = make_image(name, sample, seed, reference)
    path = _STATE['out_dir'] / 'images' / f'{position:05d}.png'
    sha = png_sha256(image, path)
    relative = path.resolve().relative_to(_STATE['outputs_root'].resolve()).as_posix()
    return make_record(name, sid, source.rows[index], source.sidecar, rm.geometry_sha256(sample), seed, relative,
                       sha, {**extra, **more})


def build(name, *, ids_path=IDS, smoke=False, source_path=SOURCE_TRAIN, target_path=TARGET_TRAIN, out_dir=None,
          outputs_root='outputs', workers=8, overwrite=False):
    data_root = os.environ['DATA_ROOT']
    out_dir = Path(out_dir or Path(outputs_root) / 'generated_variants' / name)
    manifest = out_dir / 'generated_manifest.jsonl'
    if manifest.exists() and not overwrite:
        raise FileExistsError(f'{manifest} exists; pass overwrite to rebuild (deterministic)')
    source = ManifestDataset(source_path, data_root, purpose='training', geometry='source_generation512')
    ids = rm.load_matched_ids(ids_path, source, smoke=smoke)
    initargs = (name, data_root, source_path, target_path, out_dir, outputs_root)
    jobs = list(enumerate(ids))
    if workers > 1:
        with multiprocessing.get_context('spawn').Pool(workers, _init_worker, initargs) as pool:
            records = pool.map(_work, jobs, chunksize=8)
    else:
        _init_worker(*initargs)
        records = [_work(job) for job in jobs]
    text = ''.join(json.dumps(r, allow_nan=False) + '\n' for r in records)
    manifest.write_text(text)
    summary = {'variant': name, 'records': len(records), 'smoke': smoke, 'recipe_hash': recipe_hash(name),
               'recipe': RECIPE_DEFINITION[name], 'provenance': PROVENANCE[name],
               'manifest': manifest.as_posix(), 'manifest_sha256': hashlib.sha256(text.encode()).hexdigest(),
               'ids_file': str(ids_path), 'outputs_root': str(outputs_root),
               'source_manifest': source_path, 'target_appearance_manifest': target_path if name in TARGET_VARIANTS else None,
               'labels_read': 'source_train depth/K only (geometry fingerprint); C1 RGB only; no C1 depth, no C2'}
    if name in TARGET_VARIANTS:
        summary['distinct_references'] = len({r['reference_sample_id'] for r in records})
    summary['verification'] = verify(manifest, ids, source, outputs_root)
    (out_dir / 'summary.json').write_text(json.dumps(summary, indent=1, allow_nan=False) + '\n')
    return summary


def verify(manifest, ids, source, outputs_root, probes=4):
    """The trainer's own loader: G (and M) must accept the manifest; load a few occurrences end to end."""
    report = {}
    for mode in ('G', 'M'):
        dataset = rm.RGBModeDataset(source, ids, mode=mode, seed=0, generated_manifest=manifest,
                                    outputs_root=outputs_root)
        report[f'{mode}_len'] = len(dataset)
    dataset = rm.RGBModeDataset(source, ids, mode='G', seed=0, generated_manifest=manifest, outputs_root=outputs_root)
    picks = sorted({round(i * (len(dataset) - 1) / max(1, probes - 1)) for i in range(probes)})
    for i in picks:
        sample = dataset.get_occurrence(i, 7 * i)
        rgb = sample['rgb']
        if rgb.shape != (3, 504, 504) or rgb.dtype != torch.float32 or not 0 <= rgb.min() <= rgb.max() <= 1:
            raise ValueError('Loaded G sample has unexpected shape/range')
        if sample['targets']['depth_m'].shape != (1, 504, 504):
            raise ValueError('Loaded G label has unexpected shape')
        if not torch.equal(sample['rgb'], dataset.get_occurrence(i, 7 * i)['rgb']):
            raise ValueError('G occurrence is not deterministic')
    report['G_probed_indices'] = picks
    return report


def write_smoke_config(base, manifest, ids, output, outputs_root='outputs'):
    """Copy of configs/da3_smoke.yaml in G mode on this manifest: 2 optimizer steps, trainer validation path."""
    import yaml

    config = yaml.safe_load(Path(base).read_text())
    config['data'].update(mode='G', generated_manifest=str(manifest), matched_ids=str(ids), outputs_root=outputs_root)
    config['training'].update(optimizer_steps=2, global_batch_size=4, microbatch_size=2, warmup_steps=1,
                              validation_every=2, checkpoint_every=1)
    config['evaluation']['validation_frames'] = 4
    Path(output).write_text(yaml.safe_dump(config, sort_keys=False))
    return config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    b = sub.add_parser('build')
    b.add_argument('--name', required=True, choices=sorted(PROVENANCE))
    b.add_argument('--ids', default=IDS)
    b.add_argument('--smoke', action='store_true', help='smoke-purpose ID artifact (never the production list)')
    b.add_argument('--out-dir')
    b.add_argument('--workers', type=int, default=8)
    b.add_argument('--overwrite', action='store_true')
    s = sub.add_parser('smoke-ids')
    s.add_argument('--count', type=int, default=24)
    s.add_argument('--output', required=True)
    c = sub.add_parser('smoke-config')
    c.add_argument('--manifest', required=True)
    c.add_argument('--ids', required=True)
    c.add_argument('--output', required=True)
    c.add_argument('--base', default='configs/da3_smoke.yaml')
    args = parser.parse_args()
    if args.command == 'build':
        result = build(args.name, ids_path=args.ids, smoke=args.smoke, out_dir=args.out_dir, workers=args.workers,
                       overwrite=args.overwrite)
        print(json.dumps({k: result[k] for k in ('variant', 'records', 'manifest', 'manifest_sha256', 'verification')}, indent=1))
    elif args.command == 'smoke-ids':
        print(len(rm.write_smoke_ids(SOURCE_TRAIN, args.count, args.output)['sample_ids']), 'smoke IDs')
    else:
        write_smoke_config(args.base, args.manifest, args.ids, args.output)


if __name__ == '__main__':
    main()
