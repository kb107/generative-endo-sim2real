"""Markdown + CSV tables built from saved json evidence. Nothing is typed in by hand."""
from pathlib import Path

from .common import fmt, read_json, write_table

CHECKS = ['M1_correct_beats_off', 'M2_wrong_beats_off_against_supplied', 'M3_wrong_tracks_supplied_not_original',
          'M4_correct_beats_wrong_against_original']
SHORT = ['M1', 'M2', 'M3', 'M4']


def count_cell(check):
    return f"{check['conditions_in_expected_direction']}/{check['conditions']} ({check['mean_gain']:+.3f})"


# ---------------------------------------------------------------- (a) controller gate (P4)

def _yes(values):
    return sum(1 for v in values if v == 'yes')


def gate_rows(gate_root, blind):
    """One row per gate run, in (set, step) order. Blind ratings come from `rubric_blind_independent.json`."""
    runs = []
    for gate_path in Path(gate_root).glob('*/gate.json'):
        gate = read_json(gate_path)
        runs.append((gate['render']['set'] != 'overfit', gate['render']['step'], gate_path.parent.name, gate))
    rows = []
    for _, step, name, gate in sorted(runs, key=lambda r: (r[0], r[1])):
        checks = gate['primary_abs_rel']['checks']
        sec = gate['secondary_spearman_negated']
        own = gate.get('visual_rubric')
        if own:
            ratings = own['ratings']
            n = len(ratings['layout_follows_control'])
            own_cell = (f"layout {_yes(ratings['layout_follows_control'])}/{n}, wrong-depth "
                        f"{_yes(ratings['wrong_depth_moves_layout'])}/{n}: {'pass' if own['passed'] else 'FAIL'}")
        else:
            own_cell = 'not rated'
        blind_cells = []
        for seed in (0, 1):
            grid = (blind or {}).get('grids', {}).get(f'{name}/grid_seed{seed}.png')
            blind_cells.append('not rated' if grid is None else
                               f"{grid['yes_count']}/24 yes: {'pass' if grid['pass'] else 'FAIL'}")
        gset = 'overfit (training pairs)' if gate['render']['set'] == 'overfit' else 'held-out source val.'
        rows.append([f"{gate['render']['set']} step {step}", gset]
                    + [count_cell(checks[k]) for k in ('R0_proxy_audit', 'R1_correct_beats_off',
                                                       'R2_correct_beats_wrong', 'R3_wrong_tracks_supplied')]
                    + [gate['primary_abs_rel']['proxy_verdict'], sec['proxy_verdict'], own_cell]
                    + blind_cells + [gate['verdict']])
    return rows


GATE_HEADER = ['Gate run', 'Images', 'R0 audit (real RGB)', 'R1 correct < off', 'R2 correct < wrong',
               'R3 wrong follows supplied', 'Proxy verdict (AbsRel)', 'Proxy verdict (Spearman, secondary)',
               'Self-rating (NOT blind)', 'Independent blind rating, seed 0 grid',
               'Independent blind rating, seed 1 grid', 'Declared gate verdict']


def gate_table(gate_root, blind_path, out_stem):
    blind = read_json(blind_path) if Path(blind_path).exists() else None
    rows = gate_rows(gate_root, blind)
    notes = [
        'Source-controller gate (P4), single seed pilot, SOURCE validation only; 8 conditions x 2 paired noise seeds, '
        'frozen DA3Metric proxy, median-scaled AbsRel. Cells are conditions in the expected direction / 8 (mean gain). '
        'Each check needs >= 7/8 and a positive mean gain. R3 is exactly at the threshold in the passing pilot steps.',
        '**Rating provenance.** "Self-rating" was written by the person running the gate after seeing the proxy '
        'results for that gate (not blind); the declared verdict combines the proxy rules with this rating. "Independent blind rating" '
        f'is by {(blind or {}).get("rater", "an independent rater")}; it did not see the proxy results and covers the grids '
        'listed. At pilot step 1500 the two ratings disagree (self-rating FAIL, blind 24/24 yes on both seeds); both are shown.']
    write_table(out_stem, 'Source-controller gate (P4) per step', GATE_HEADER, rows, notes)
    return rows


# ---------------------------------------------------------------- (b) reduced vs full matrices

def _blocks(reduced_dirs, full_analysis):
    """List of (label, seeds, n_conditions, primary_block, secondary_block, r0)."""
    blocks = []
    for label, directory in reduced_dirs:
        m = read_json(Path(directory) / 'matrix.json')
        n = m['primary_abs_rel']['per_adapter']['target']['checks'][CHECKS[0]]['conditions']
        blocks.append((label, str(m['render']['seed']), n, m['primary_abs_rel'], m['secondary_spearman'],
                       m['R0_audit_real_rgb']))
    a = read_json(full_analysis)
    n = a['conditions']
    for seed in a['seeds']:
        blocks.append((f'v2 full 32x2: seed {seed}', str(seed), n, a['primary_abs_rel']['per_seed'][str(seed)],
                       a['secondary_spearman']['per_seed'][str(seed)],
                       a['primary_abs_rel']['per_seed'][str(seed)]['R0_audit_real_rgb']))
    blocks.append(('v2 full 32x2: combined (condition mean over seeds)', '+'.join(str(s) for s in a['seeds']), n,
                   a['primary_abs_rel']['combined'], a['secondary_spearman']['combined'],
                   a['primary_abs_rel']['combined']['R0_audit_real_rgb']))
    return blocks, a['rules']


def matrix_tables(reduced_dirs, full_analysis, out_dir):
    out_dir = Path(out_dir)
    blocks, full_rules = _blocks(reduced_dirs, full_analysis)
    rows = []
    for label, seeds, n, primary, secondary, r0 in blocks:
        for adapter in ('none', 'source', 'target'):
            entry = primary['per_adapter'][adapter]
            sec = secondary['per_adapter'][adapter]
            rows.append([label, seeds, n, 'v2' if 'v2' in label else 'v1', adapter,
                         f"{r0['conditions_in_expected_direction']}/{r0['conditions']}"]
                        + [count_cell(entry['checks'][k]) for k in CHECKS]
                        + [fmt(entry['control_retained']), fmt(sec['control_retained'])])
    header = ['Matrix', 'Noise seed(s)', 'Conditions', 'Target LoRA', 'Adapter', 'R0 audit'] + [
        f'{s} ({k.split("_", 1)[1].replace("_", " ")})' for s, k in zip(SHORT, CHECKS)] + [
        'Control retained (primary AbsRel)', 'Control retained (Spearman, secondary)']
    notes = [
        'Swap diagnostic: {none = both LoRAs off, source, target} x {controller off, correct, wrong depth}, SOURCE-validation '
        'conditions, frozen DA3Metric proxy, median-scaled AbsRel. Cells: conditions in the expected direction (mean gain). '
        f'Reduced matrices need >= 7/8 per check; the 32-condition expansion uses the scaled rule >= {full_rules["min_conditions"]}/32 '
        '(ceil(0.875 x 32)), thresholds otherwise unchanged. Only within-adapter comparisons are decision-relevant; the '
        'proxy mixes shading and gloss with geometry. v1 = target LoRA 1000 steps; v2 = selected recipe (2000 steps, selected '
        'by RGB-appearance metrics, not by these scores). Single recipe, one proxy, controller trained under the source adapter only.']
    write_table(out_dir / 'swap_matrix_by_adapter', 'Swap matrix per adapter: reduced vs full', header, rows, notes)

    rrows = []
    for label, seeds, n, primary, secondary, _ in blocks:
        for metric, block in (('primary AbsRel', primary), ('secondary Spearman', secondary)):
            ret = block['retention_source_vs_target']
            rrows.append([label, n, metric,
                          block['swap_verdict'].split(' (')[0]]
                         + [f"{ret[k]['ratio_target_over_source']:.2f} ({ret[k]['conditions_target_gain_larger']}/{n})"
                            for k in CHECKS]
                         + [block['swap_verdict']])
    rheader = ['Matrix', 'Conditions', 'Metric', 'Declared target-adapter verdict'] + [
        f'{s} ratio target/source gain (cond. where target gain larger)' for s in SHORT] + ['Verdict detail']
    rnotes = ['Retention ratio = mean gain with the target adapter / mean gain with the source adapter. No retention '
              'threshold was declared, so none is applied; the declared verdict is the primary-AbsRel rule. The secondary '
              'metric is reported next to it, not instead of it. Reduced (8-condition) verdicts are NOT superseded by the '
              '32 x 2 expansion; they are different evidence levels.']
    write_table(out_dir / 'swap_matrix_retention', 'Swap matrix: verdicts and retention ratios', rheader, rrows, rnotes)
    return rows, rrows


# ---------------------------------------------------------------- (c) DA3 source validation

def da3_rows(runs_root, methods=('R', 'A', 'G', 'M'), seed=0):
    rows = []
    for method in methods:
        run = Path(runs_root) / f'da3_{method}_seed{seed}'
        status = read_json(run / 'status.json') if (run / 'status.json').exists() else None
        done, target = (status['completed_steps'], status['target_steps']) if status else (None, None)
        steps = sorted((run / 'validation').glob('step_*/summary.json')) if run.exists() else []
        seen = []
        for path in steps:
            step = int(path.parent.name.split('_')[1])
            s = read_json(path)['sequence_macro']
            seen.append((step, s))
        for step, s in seen:
            rows.append([method, step, 'pretrained start (step 0)' if step == 0 else 'trained', s['abs_rel'],
                         s['rmse_mm'], s['delta_1_25'], s['oracle_median_rmse_mm'],
                         'complete' if done is not None and done >= target else
                         f'training in progress ({done}/{target} steps)'])
        if not any(step > 0 for step, _ in seen):
            rows.append([method, 'final', 'trained', None, None, None, None,
                         'pending' + (f' (training {done}/{target} steps)' if status else ' (no run found)')])
    return rows


def da3_table(runs_root, out_stem):
    rows = da3_rows(runs_root)
    header = ['Method', 'Step', 'Head', 'AbsRel (sequence macro)', 'RMSE mm (sequence macro)', 'delta<1.25',
              'Oracle median-aligned RMSE mm (diagnostic)', 'Run status']
    notes = ['**SOURCE validation only** (SimCol3D held-out validation, 200 frames, 2 sequences); this is NOT real-phantom '
             '(C3VDv2 C2) performance, and no C2 evaluation is included here. Metric unaligned depth in mm; the oracle '
             'column is a ground-truth-aligned diagnostic. R = raw source, A = photometric augmentation, G = generated, '
             'M = mixture; seed 0, frozen N=2000 matched IDs, 1000 steps. "pending" = training not finished when this '
             'table was built (G and M cells are filled by re-running the build after their validation steps exist).']
    write_table(out_stem, 'DA3 source-validation results (R/A/G/M)', header, rows, notes)
    return rows


# ---------------------------------------------------------------- appearance metrics table (figure 4)

def appearance_rows(v1_ref, v2_ref):
    """(header, body) from the two saved review json files. Source LoRA numbers come from the v1 review."""
    def nn(ref, adapter):
        key = 'source_train' if adapter == 'source' else 'c1_train'
        entry = ref['nearest_neighbour'][key]
        return (entry['latent']['median_ratio_generated_over_heldout'], entry['pixel']['median_ratio_generated_over_heldout'])

    def row(name, ref, adapter):
        div = ref['diversity_mean_pairwise_l1']['generated_' + adapter]
        w1 = ref['w1_rgb_mean'][adapter]
        lat, pix = nn(ref, adapter)
        return [name, f'{div:.3f}', f"{w1['c1_train']:.3f}", f"{w1['c1_validation']:.3f}",
                f"{w1['source_train']:.3f}", f'{lat:.2f} / {pix:.2f}']
    div = v1_ref['diversity_mean_pairwise_l1']
    w1 = v1_ref['w1_rgb_mean']
    real_row = ['Real C1 train / validation', f"{div['real_c1_train']:.3f} / {div['real_c1_validation']:.3f}",
                f"(train vs val {w1['real_c1_train_vs_c1_validation']:.3f})", '', f"(C1 train vs source train {w1['real_c1_train_vs_source_train']:.3f})", '1.00 = real']
    header = ['Set (64 images, controller off)', 'Diversity (L1)', 'W1 to C1 train', 'W1 to C1 val.',
              'W1 to source train', 'NN ratio latent / pixel']
    body = [row('Source LoRA', v1_ref, 'source'), row('Target LoRA v1', v1_ref, 'target'),
            row('Target LoRA v2', v2_ref, 'target'), real_row]
    return header, body


def appearance_table(v1_ref, v2_ref, out_stem):
    header, body = appearance_rows(v1_ref, v2_ref)
    write_table(out_stem, 'Appearance metrics behind Figure 4', header, body,
                ['Read from `appearance_review/*/reference.json`. Single seed set, RGB statistics only; these do not establish '
                 'geometry or realism. The source-LoRA row is from the v1 review (reproduced exactly in the v2 review).'])
    return header, body
