"""Build the fixed comparison inputs for the foundation-model study (DA3 venv; reads RGB only
for C1, source RGB for SimCol; never touches C1 labels or C2).

Writes outputs/foundation/inputs/inputs.json plus refs/ref{1,2}.png and real300/NNN.png.
Run: PYTHONPATH=src external/da3-venv/bin/python -m generative_sim2real.foundation.inputs
"""
import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np
from PIL import Image

from . import selection

MATRIX = 'outputs/p5-diagnostics/matrix_lc3000_seed17'
MANIFESTS = 'outputs/manifests/milestone_v1'
GENERATED = {'ours_controllora': 'outputs/generated/a9743a4845dca5be',
             'ours_fullctrl': 'outputs/generated/c80ff98494cdb36d',
             'simuscope': 'outputs/generated_baselines/simuscope'}


def _png(tensor, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    array = (tensor.clamp(0, 1) * 255).round().byte().permute(1, 2, 0).numpy()
    Image.fromarray(array).save(path)


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def fixed8(matrix_dir):
    matrix = json.loads((Path(matrix_dir) / 'matrix.json').read_text())['render']
    conds = selection.pick_conditions(len(matrix['ids']))
    rows = []
    for c in conds:
        rows.append({'cond': c, 'sample_id': matrix['ids'][c], 'seed': matrix['seed'] + c,
                     **{k: f'{matrix_dir}/images/{k}/c{c}.png'
                        for k in ('real', 'control', 'source_correct', 'target_correct')}})
    return rows


def references(data_root, count=2):
    from generative_sim2real.generation.inputs import generation_dataset
    dataset = generation_dataset(f'{MANIFESTS}/target_appearance_train.jsonl', data_root,
                                 stage='target_lora')
    means, sequences = [], []
    for i in range(len(dataset)):
        means.append(dataset[i]['rgb'].mean(dim=(1, 2)).numpy())
        sequences.append(dataset.rows[i]['sequence_id'])
    chosen = selection.nearest_to_mean(means, sequences, count)
    out = []
    for k, i in enumerate(chosen, 1):
        sample = dataset[i]
        path = Path('outputs/foundation/inputs/refs') / f'ref{k}.png'
        _png(sample['rgb'], path)
        out.append({'file': str(path), 'sample_id': sample['sample_id'],
                    'sequence_id': dataset.rows[i]['sequence_id'], 'mean_rgb': means[i].tolist(),
                    'distance_to_c1_mean': float(np.linalg.norm(means[i] - np.mean(means, axis=0)))})
    return out, {'n_frames': len(means), 'c1_train_mean_rgb': np.mean(means, axis=0).tolist()}


def subset(data_root, count=300):
    from generative_sim2real.generation.inputs import generation_dataset
    ids = json.loads(Path(f'{MANIFESTS}/downstream_ids.json').read_text())['sample_ids']
    chosen = selection.evenly_spaced(ids, count)
    paths = {}
    for name, root in GENERATED.items():
        rows = {r['sample_id']: r['generated_rgb_path'] for r in map(
            json.loads, Path(root, 'generated_manifest.jsonl').read_text().splitlines())}
        paths[name] = {s: 'outputs/' + rows[s] for s in chosen}  # KeyError = missing image
    dataset = generation_dataset(f'{MANIFESTS}/source_train.jsonl', data_root, stage='generation')
    index = {row['sample_id']: i for i, row in enumerate(dataset.rows)}
    out = []
    for k, s in enumerate(chosen):
        real = Path('outputs/foundation/inputs/real300') / f'{k:03d}.png'
        if not real.exists():
            _png(dataset[index[s]]['rgb'], real)
        out.append({'index': k, 'sample_id': s, 'real': str(real),
                    **{name: paths[name][s] for name in GENERATED}})
    return out


def depth_maps(data_root, spec, out_dir='outputs/foundation/inputs/depth300'):
    """Write each subset300 SimCol depth map as the controller's own control image (near = bright, invalid = 0),
    at the same 512 px crop as real300/, for editors that take a depth image instead of the frame."""
    from generative_sim2real.generation.inputs import control_image, generation_dataset
    dataset = generation_dataset(f'{MANIFESTS}/source_train.jsonl', data_root, stage='generation')
    index = {row['sample_id']: i for i, row in enumerate(dataset.rows)}
    written = 0
    for entry in spec['subset300']:
        path = Path(out_dir) / f'{entry["index"]:03d}.png'
        if not path.exists():
            _png(control_image(dataset[index[entry['sample_id']]], dataset.sidecar), path)
            written += 1
    return written


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', default=os.environ.get('DATA_ROOT'))
    parser.add_argument('--out', default='outputs/foundation/inputs/inputs.json')
    parser.add_argument('--depth300', action='store_true',
                        help='only write depth300/NNN.png control images for an existing inputs.json')
    args = parser.parse_args()
    if not args.data_root:
        raise SystemExit('DATA_ROOT is not set (source .env)')
    if args.depth300:
        print('depth maps written:', depth_maps(args.data_root, json.loads(Path(args.out).read_text())))
        return
    refs, ref_info = references(args.data_root)
    result = {
        'rules': {
            'fixed8': 'matrix conditions 13,3,12,7 + 4 evenly spaced (np.linspace, rounded) over the '
                      'sorted remaining 28 condition indices; images from ' + MATRIX,
            'subset300': '300 of the 2,000 frozen downstream IDs at np.linspace(0,1999,300) rounded',
            'refs': 'two C1 target_appearance_train frames nearest (L2 of per-frame mean RGB, '
                    '512 px) to the mean over all C1-train frames; the second must come from a '
                    'different sequence than the first'},
        'matrix_dir': MATRIX, 'fixed8': fixed8(MATRIX), 'refs': refs, 'refs_info': ref_info,
        'subset300': subset(args.data_root),
        'generated_roots': GENERATED}
    result['refs_sha256'] = [_sha(r['file']) for r in refs]
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(result, indent=1) + '\n')
    print('fixed8 conditions', [r['cond'] for r in result['fixed8']])
    print('refs', [(r['sample_id'], round(r['distance_to_c1_mean'], 4)) for r in refs])
    print('subset300', len(result['subset300']))


if __name__ == '__main__':
    main()
