"""Markdown table of several compare.py result files (one row per result; floor shown as mean +- std).

  python -m generative_sim2real.generation_metrics.table a.json b.json ... [--labels a,b,...]"""
import argparse
import json
from pathlib import Path


def pm(spread):
    return f"{spread['mean']:.4g} +- {spread['std']:.2g}"


def rows(paths, labels=None):
    out = []
    for i, path in enumerate(paths):
        r = json.loads(Path(path).read_text())
        d, g = r['generated_vs_reference'], r['pixel_stats']['generated']
        floor = r.get('floor_train_vs_reference')
        sd = r['generated_vs_reference_spread']['distances']
        out.append({
            'label': labels[i] if labels else Path(path).stem, 'n': r['n_generated'],
            'n_ref': r['n_reference'], 'kid': d['kid']['full'], 'kid_sd': sd['kid']['std'],
            'fd': d['frechet'], 'fd_sd': sd['frechet']['std'], 'w1': d['w1_rgb']['mean'],
            'w1_sd': sd['w1_rgb_mean']['std'], 'div': g['diversity_l1_64'], 'sat': g['sat_mean'],
            'sat95': g['sat_p95'], 'bright': g['brightness'],
            'floor': None if floor is None else {
                'kid': pm(floor['distances']['kid']), 'fd': pm(floor['distances']['frechet']),
                'w1': pm(floor['distances']['w1_rgb_mean']),
                'div': pm(floor['train_pixel_stats']['diversity_l1_64']),
                'sat': pm(floor['train_pixel_stats']['sat_mean']),
                'sat95': pm(floor['train_pixel_stats']['sat_p95']),
                'bright': pm(floor['train_pixel_stats']['brightness']), 'n': floor['n_train_per_draw']},
            'ref_px': r['pixel_stats']['reference']})
    return out


def markdown(paths, labels=None):
    lines = ['| set | n / n_ref | KID(DINOv2) | FD(DINOv2) | W1 RGB | diversity | sat mean | sat p95 | brightness |',
             '| --- | --- | --- | --- | --- | --- | --- | --- | --- |']
    seen = set()
    for r in rows(paths, labels):
        lines.append(f"| {r['label']} | {r['n']} / {r['n_ref']} | {r['kid']:.3f} (sd {r['kid_sd']:.2f}) | "
                     f"{r['fd']:.1f} (sd {r['fd_sd']:.1f}) | {r['w1']:.4f} (sd {r['w1_sd']:.4f}) | "
                     f"{r['div']:.3f} | {r['sat']:.3f} | {r['sat95']:.3f} | {r['bright']:.3f} |")
        if r['floor'] and (r['floor']['fd'], r['n_ref']) not in seen:
            f = r['floor']
            seen.add((f['fd'], r['n_ref']))
            lines.append(f"| floor: real train n={f['n']} vs same ref | {f['n']} / {r['n_ref']} | {f['kid']} | "
                         f"{f['fd']} | {f['w1']} | {f['div']} | {f['sat']} | {f['sat95']} | {f['bright']} |")
        px = r['ref_px']
        lines.append(f"| reference set pixel stats | - / {px['n']} | | | | {px['diversity_l1_64']:.3f} | "
                     f"{px['sat_mean']:.3f} | {px['sat_p95']:.3f} | {px['brightness']:.3f} |")
    return '\n'.join(lines)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('results', nargs='+')
    p.add_argument('--labels', default=None, help='comma-separated row labels')
    args = p.parse_args()
    print(markdown(args.results, args.labels.split(',') if args.labels else None))


if __name__ == '__main__':
    main()
