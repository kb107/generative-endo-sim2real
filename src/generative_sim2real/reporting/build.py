"""Build every P7 result asset that does not depend on real-phantom (C2) results.

    PYTHONPATH=src external/da3-venv/bin/python -m generative_sim2real.reporting.build \
        --outputs outputs --out docs/results

Reads saved evidence only; renders nothing with a model and never opens C2 data.
"""
import argparse
import json
from pathlib import Path

from . import figures, tables
from .common import read_json
from .matrix import MatrixRun


def pipeline_stages(outputs, c2_status):
    """Stage text and 'actually run' status derived from files on disk."""
    outputs = Path(outputs)
    runs = outputs / 'p4-appearance' / 'runs'

    def finished(name):
        path = runs / name / 'final.json'
        if not path.exists():
            return None
        final = read_json(path)
        return final if final.get('completed_all_steps') else None

    def lora(name):
        final = finished(name)
        return ('run', f"{final['steps_completed']} steps (settings: configs/p4.yaml)") if final else \
            ('not run', 'not found')

    src_status, src_detail = lora('source_lora')
    tgt_status, tgt_detail = lora('target_lora_v2')
    ctl = finished('controller_pilot')
    ctl_detail = f"{ctl['steps_completed']} steps (pilot)" if ctl else 'not found'

    manifests = sorted((outputs / 'generated').glob('*/generated_manifest.jsonl'))
    n_generated = sum(1 for _ in open(manifests[0])) if manifests else 0

    da3 = {}
    for method in 'RAGM':
        path = outputs / 'runs' / f'da3_{method}_seed0' / 'status.json'
        if path.exists():
            s = read_json(path)
            da3[method] = (s['completed_steps'], s['target_steps'])
    parts = []
    for method in 'RAGM':
        if method in da3:
            done, target = da3[method]
            parts.append(f"{method}: {done}/{target}")
        else:
            parts.append(f'{method}: none')
    all_done = all(m in da3 and da3[m][0] >= da3[m][1] for m in 'RAGM')
    any_done = any(m in da3 and da3[m][0] >= da3[m][1] for m in 'RAGM')
    return {
        'data_source': {'text': 'SimCol3D source', 'kind': 'data', 'status': 'run',
                        'detail': 'RGB + metric depth\ntrain / validation / test split\nvalidation never trained on'},
        'data_c1': {'text': 'C3VDv2 C1 (RGB only)', 'kind': 'data', 'status': 'run',
                    'detail': 'appearance train / validation\nno depth or pose indexed\nor decoded'},
        'base': {'text': 'Base: PixArt-alpha 512 (frozen)', 'kind': 'frozen', 'status': 'run', 'detail': ''},
        'lora_source': {'text': 'Source appearance LoRA', 'kind': 'trained', 'status': src_status,
                        'detail': f'{src_detail}\ntrained on SimCol3D RGB'},
        'lora_target': {'text': 'Target appearance LoRA (v2)', 'kind': 'trained', 'status': tgt_status,
                        'detail': f'{tgt_detail}\ntrained on C1 RGB only'},
        'controller': {'text': 'Depth controller', 'kind': 'trained', 'status': 'run' if ctl else 'not run',
                       'detail': f'{ctl_detail}\ntrained with source LoRA + backbone\nFROZEN; saved separately'},
        'swap': {'text': 'Swap: target LoRA on,\nsource LoRA off', 'kind': 'frozen', 'status': 'run',
                 'detail': 'controller unchanged;\nsource -> target -> source\nrecovery tested'},
        'generated': {'text': 'Generated pairs', 'kind': 'artifact', 'status': 'run' if n_generated else 'not run',
                      'detail': f'{n_generated} images + exact\nSOURCE depth labels\n(one per frozen ID)'},
        'da3': {'text': 'DA3Metric-L head training', 'kind': 'trained',
                'status': 'run' if all_done else ('partial' if any_done or da3 else 'not run'),
                'detail': 'R raw / A augmented /\nG generated / M mixture\nbackbone frozen; head trained\nsteps done / target:\n' + '  '.join(parts[:2]) + '\n' + '  '.join(parts[2:])},
        'c2': {'text': 'Locked C2 evaluation', 'kind': 'frozen', 'status': 'not run',
               'detail': f'{c2_status}'},
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--outputs', default='outputs')
    parser.add_argument('--out', default='docs/results')
    parser.add_argument('--c2-status', default='not run here\nfrozen registry required\n(excluded from these assets)')
    args = parser.parse_args(argv)
    outputs, out = Path(args.outputs), Path(args.out)
    p5 = outputs / 'p5-diagnostics'
    p4 = outputs / 'p4-appearance'
    summary = {}

    v2 = MatrixRun(p5 / 'matrix_reduced_v2')
    encoding = read_json(outputs / 'manifests' / 'milestone_v1' / 'depth_encoding.json')
    summary['fig1_generation_matrix'] = figures.generation_figure(
        v2, encoding, out / 'fig1_generation_matrix_v2.png')

    rule = ('rank every (adapter, condition) cell by M1 gain ascending, regardless of sign, and show the 6 lowest '
            '(configs/p5.yaml failure_panel_rule).')
    summary['fig2a_failures_reduced_v2'] = figures.failure_figure(
        v2, v2.matrix['failure_cells_lowest_M1_gain'], out / 'fig2a_failure_panel_reduced_v2.png',
        rule_text=f'{rule} Here: 3 adapters x 8 conditions = 24 cells, recipe v2, one seed.',
        title='Figure 2a. Failure panel, reduced matrix (v2).')
    analysis = read_json(p5 / 'matrix_full' / 'analysis.json')
    full_rule = (f'{rule} Here: combined M1 gain (mean over seeds) over 3 adapters x {analysis["conditions"]} conditions = '
                 f'{3 * analysis["conditions"]} cells (configs/p5_full.yaml failure_panel).')
    for seed in analysis['seeds']:
        run = MatrixRun(p5 / f'matrix_full_seed{seed}')
        cells = analysis['failure_cells_lowest_combined_M1_gain']
        summary[f'fig2b_failures_full_seed{seed}'] = figures.failure_figure(
            run, cells, out / f'fig2b_failure_panel_full_seed{seed}.png', rule_text=full_rule,
            title=f'Figure 2b. Failure panel, 32x2 expansion (v2), images from seed {seed}.',
            shown_gain=[run.m1_gains(c['adapter'])[c['condition']] for c in cells],
            seed_label=f'noise seed {seed} + index', extra_gain_label='combined')

    appearance = read_json(p4 / 'appearance' / 'lora1000_controller_pilot2000' / 'appearance.json')
    reload_smoke = read_json(outputs / 'generator-integration' / 'smoke' / 'reload.json')
    if appearance['source_validation_condition_ids'] != v2.ids[:len(appearance['source_validation_condition_ids'])]:
        raise SystemExit('Appearance conditions differ from the reduced matrix conditions; cannot reuse control images.')
    summary['fig3_reload_example'] = figures.reload_figure(
        p4 / 'appearance' / 'lora1000_controller_pilot2000', p5 / 'matrix_reduced_v2' / 'images', appearance,
        reload_smoke, out / 'fig3_source_target_source_reload.png')
    rec = appearance['swap_recovery_trained_adapters']['source_second_vs_first']
    sm = reload_smoke
    (out / 'reload_tolerance.md').write_text(
        '# Source -> target -> source reload tolerance (measured)\n\n'
        f"Measured max abs error {rec['max_abs_error']:.1g} (mean {rec['mean_abs_error']:.1g}) over {rec['elements']:,} values "
        f"with controller on, after swapping the trained source LoRA to the trained target LoRA and back "
        f"(declared tolerance atol {rec['atol']:g} / rtol {rec['rtol']:g}; {rec['violations']} violations); the P3 fresh-reload "
        f"smoke gives {sm['source_reload_error']['max_abs_error']:.1g} (source) and {sm['target_reload_error']['max_abs_error']:.1g} "
        f"(target) over {sm['target_reload_error']['elements']:,} values. Sources: "
        '`outputs/p4-appearance/appearance/*/appearance.json`, `outputs/generator-integration/smoke/reload.json`.\n')

    v1_ref = read_json(p4 / 'appearance_review' / 'lora1000' / 'reference.json')
    v2_ref = read_json(p4 / 'appearance_review' / 'target_lora_v2_2000' / 'reference.json')
    table = tables.appearance_table(v1_ref, v2_ref, out / 'table_appearance_metrics')
    summary['fig4_appearance'] = figures.appearance_figure(
        p4 / 'appearance' / 'lora1000_controller_pilot2000' / 'grid_c1_validation.png',
        p4 / 'appearance_review' / 'lora1000', p4 / 'appearance_review' / 'target_lora_v2_2000', table,
        out / 'fig4_appearance_c1_v1_v2.png')

    summary['table_gate'] = len(tables.gate_table(p4 / 'gate_runs', p4 / 'rubric_blind_independent.json',
                                                  out / 'table_controller_gate_p4'))
    reduced = [('v1 reduced 8x1', p5 / 'matrix_reduced'), ('v2 reduced 8x1', p5 / 'matrix_reduced_v2')]
    a, b = tables.matrix_tables(reduced, p5 / 'matrix_full' / 'analysis.json', out)
    summary['table_matrix_rows'] = [len(a), len(b)]
    da3 = tables.da3_table(outputs / 'runs', out / 'table_da3_source_validation')
    summary['table_da3_rows'] = len(da3)

    stages = pipeline_stages(outputs, args.c2_status)
    summary['fig6_pipeline'] = figures.pipeline_figure(stages, out / 'fig6_pipeline.png')
    (outputs / 'p7-results').mkdir(parents=True, exist_ok=True)
    (outputs / 'p7-results' / 'build_summary.json').write_text(json.dumps(summary, indent=1) + '\n')
    print(json.dumps(summary, indent=1))


if __name__ == '__main__':
    main()
