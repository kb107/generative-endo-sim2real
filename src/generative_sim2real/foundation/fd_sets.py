"""Matched-ID image sets for FD at n=300: one directory of symlinks per arm, same 300 IDs.

The 300 IDs are `inputs.json: subset300` (evenly spaced over downstream_ids). Arms come from that file (real SimCol,
Control-LoRA, full controller, SimuScope-style), from G-style manifests looked up by sample_id (Gp, S, T, FDA), and
from the closed editor's outputs `<index:03d>.png`. A missing image (e.g. a request the editor's content filter
blocked) is left out and counted; every arm's directory lists exactly which IDs it holds (ids.json).
"""
import argparse
import json
from pathlib import Path

MANIFEST_ARMS = {
    'Gp_controllora_genseed2': 'outputs/generated/e8413cc36b857c2d/generated_manifest.jsonl',
    'S_source_look': 'outputs/generated/316eecddb37769b5/generated_manifest.jsonl',
    'T_reinhard': 'outputs/generated_variants/T_reinhard/generated_manifest.jsonl',
    'F_fda_b01': 'outputs/generated_variants/F_fda_b01/generated_manifest.jsonl',
    'F_fda_b05': 'outputs/generated_variants/F_fda_b05/generated_manifest.jsonl',
}
INPUT_ARMS = {'real_simcol': 'real', 'G_controllora': 'ours_controllora', 'G_fullctrl': 'ours_fullctrl',
              'simuscope': 'simuscope'}
EDITOR_ARMS = {'nano_banana_2.1': 'outputs/foundation/gemini/gemini-nano-banana-2.1/subset300/{index:03d}.png'}


def build_sets(inputs_path, out_dir, outputs_root='outputs'):
    subset = json.loads(Path(inputs_path).read_text())['subset300']
    out_dir = Path(out_dir)
    arms = {}
    for arm, key in INPUT_ARMS.items():
        arms[arm] = {e['index']: Path(e[key]) for e in subset}
    for arm, manifest in MANIFEST_ARMS.items():
        if not Path(manifest).exists():
            continue
        by_id = {}
        for line in Path(manifest).read_text().splitlines():
            row = json.loads(line)
            by_id[row['sample_id']] = Path(outputs_root) / row['generated_rgb_path']
        arms[arm] = {e['index']: by_id[e['sample_id']] for e in subset if e['sample_id'] in by_id}
    for arm, template in EDITOR_ARMS.items():
        arms[arm] = {e['index']: Path(template.format(index=e['index'])) for e in subset}
    report = {}
    for arm, files in arms.items():
        present = {i: p for i, p in files.items() if p.exists()}
        directory = out_dir / arm
        directory.mkdir(parents=True, exist_ok=True)
        for old in directory.glob('*.png'):
            old.unlink()
        for index, path in sorted(present.items()):
            (directory / f'{index:03d}.png').symlink_to(path.resolve())
        (directory / 'ids.json').write_text(json.dumps(sorted(present)) + '\n')
        report[arm] = {'n': len(present), 'missing': sorted(set(files) - set(present))}
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--inputs', default='outputs/foundation/inputs/inputs.json')
    parser.add_argument('--out', default='outputs/foundation/fd300/sets')
    args = parser.parse_args()
    report = build_sets(args.inputs, args.out)
    Path(args.out, 'sets.json').write_text(json.dumps(report, indent=1) + '\n')
    for arm, r in report.items():
        print(f"{arm}: n={r['n']} missing={len(r['missing'])}")


if __name__ == '__main__':
    main()
