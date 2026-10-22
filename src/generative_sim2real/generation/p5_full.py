"""Combined two-seed analysis of the full P5 diagnostic expansion (declared in configs/p5_full.yaml).

Each seed was rendered and scored by p5_matrix.py (configs/p5_full_seed*.yaml). This module reads their
render.json/proxy.json, evaluates the declared M1-M4/R0 rules per seed and for the condition mean over the two
seeds, and writes outputs/p5-diagnostics/matrix_full/analysis.json plus the combined failure panels and
8-row panels. It never renders and never touches the generator code path."""
import argparse
import json
from pathlib import Path

import yaml

from . import p4_report as report
from . import p5_matrix as p5
from .provenance import provenance

ADAPTERS = ['none', 'source', 'target']
ARM_ORDER = [f'{a}_{c}' for a in ADAPTERS for c in ('off', 'correct', 'wrong')]


def spearman_rows(rows):
    """Negate correlations so that lower is better, and reuse the abs_rel key names (as p5_matrix does)."""
    return [{**r, 'abs_rel_own': -r['spearman_own'], 'abs_rel_partner': -r['spearman_partner']}
            for r in rows]


def analyse(rows, partner, rules):
    """One metric's per-adapter checks, retention comparison, R0 audit and swap verdict."""
    count = len(partner)
    own = report.condition_means(rows, 'real', 'abs_rel_own', count)
    other = report.condition_means(rows, 'real', 'abs_rel_partner', count)
    audit = report.direction_check([b - a for a, b in zip(own, other)], min_conditions=rules['min_conditions'],
                                   require_positive_mean=rules['require_positive_mean_gain'])
    per_adapter = p5.matrix_checks(rows, partner, rules, ADAPTERS)
    return {'R0_audit_real_rgb': audit, 'per_adapter': per_adapter,
            'retention_source_vs_target': p5.retention_comparison(per_adapter),
            'swap_verdict': p5.swap_verdict(per_adapter, audit['passed'])}


def summarize(analysis):
    """Compact per-adapter table: check -> (conditions in expected direction, mean gain)."""
    return {a: {'control_retained': e['control_retained'],
                **{k: [v['conditions_in_expected_direction'], v['mean_gain']] for k, v in e['checks'].items()}}
            for a, e in analysis['per_adapter'].items()}


def load_seed(directory):
    directory = Path(directory)
    render = json.loads((directory / 'render.json').read_text())
    proxy = json.loads((directory / 'proxy.json').read_text())
    return render, proxy


def full_analysis(seed_dirs, rules):
    loaded = [load_seed(d) for d in seed_dirs]
    first = loaded[0][0]
    for render, _ in loaded[1:]:
        if render['ids'] != first['ids'] or render['partner'] != first['partner']:
            raise ValueError('The two seeds must use the same conditions and derangement')
    partner = first['partner']
    out = {'conditions': len(partner), 'seeds': [r['seed'] for r, _ in loaded], 'ids': first['ids'],
           'partner': partner, 'rules': rules, 'tests': {r['seed']: r['tests'] for r, _ in loaded}}
    for metric, label in (('abs_rel', 'primary_abs_rel'), ('spearman', 'secondary_spearman')):
        prep = (lambda rows: rows) if metric == 'abs_rel' else spearman_rows
        per_seed = {str(r['seed']): analyse(prep(p['rows']), partner, rules) for r, p in loaded}
        combined = analyse(prep([row for _, p in loaded for row in p['rows']]), partner, rules)
        out[label] = {'combined': combined, 'per_seed': per_seed,
                      'verdicts': {'combined': combined['swap_verdict'],
                                   **{s: a['swap_verdict'] for s, a in per_seed.items()}},
                      'seeds_agree_with_combined': all(a['swap_verdict'] == combined['swap_verdict']
                                                       for a in per_seed.values())}
    cells = p5.failure_cells(out['primary_abs_rel']['combined']['per_adapter'], ADAPTERS, 6)
    out['failure_cells_lowest_combined_M1_gain'] = [
        {'adapter': a, 'condition': i, 'M1_gain': g} for g, a, i in cells]
    return out, cells


def failure_panels(seed_dirs, cells, partner, out_dir):
    from . import gate

    for directory in seed_dirs:
        directory = Path(directory)
        load = lambda arm, i: report.png_tensor(directory / 'images' / arm / f'c{i}.png')  # noqa: E731
        rows = [[load('real', i), load('control', i), load(f'{a}_off', i), load(f'{a}_correct', i),
                 load(f'{a}_wrong', i), load('control', partner[i])] for _, a, i in cells]
        gate.save_panel(Path(out_dir) / f'failure_panel_{directory.name}.png', rows,
                        ['real rgb', 'control', 'off', 'correct', 'wrong', 'wrong: supplied control'])


def chunk_panels(seed_dirs, partner, out_dir, size=8):
    """8-row panels of the fixed condition order, all 12 columns, same geometry for every image."""
    from . import gate

    columns = ['real rgb', 'control', 'wrong: supplied control'] + ARM_ORDER
    for directory in seed_dirs:
        directory = Path(directory)
        load = lambda arm, i: report.png_tensor(directory / 'images' / arm / f'c{i}.png')  # noqa: E731
        for start in range(0, len(partner), size):
            rows = [[load('real', i), load('control', i), load('control', partner[i])]
                    + [load(arm, i) for arm in ARM_ORDER] for i in range(start, min(start + size, len(partner)))]
            gate.save_panel(Path(out_dir) / f'panel_{directory.name}_c{start}-{start + len(rows) - 1}.png',
                            rows, columns)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='configs/p5_full.yaml')
    args = parser.parse_args()
    config = yaml.safe_load(Path(args.config).read_text())
    out_dir = Path(config['output_dir'])
    out_dir.mkdir(parents=True, exist_ok=True)
    result, cells = full_analysis(config['seed_runs'], config['rules'])
    result['code_provenance'] = provenance()
    failure_panels(config['seed_runs'], cells, result['partner'], out_dir)
    chunk_panels(config['seed_runs'], result['partner'], out_dir)
    (out_dir / 'analysis.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    for label in ('primary_abs_rel', 'secondary_spearman'):
        entry = result[label]
        print(label, entry['verdicts'], 'seeds agree:', entry['seeds_agree_with_combined'])
        for name, analysis in [('combined', entry['combined'])] + [(f'seed {s}', a) for s, a in entry['per_seed'].items()]:
            print(' ', name, 'R0', analysis['R0_audit_real_rgb']['conditions_in_expected_direction'])
            for adapter, row in summarize(analysis).items():
                print('    ', adapter, row['control_retained'],
                      {k[:2]: (v[0], round(v[1], 3)) for k, v in row.items() if k != 'control_retained'})


if __name__ == '__main__':
    main()
