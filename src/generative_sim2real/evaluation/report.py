"""Per-method reports and the comparison tables (any registered method list) for the locked C2 campaign.

Headline columns are raw (unaligned) metric predictions aggregated with equal
weight per sequence. The GT-median-scaled RMSE is an oracle diagnostic and is
written to its own table with an explicit label.
"""
from __future__ import annotations
import csv
import json
from collections import Counter
from pathlib import Path

from .depth import METRICS, aggregate_frames

RAW_METRICS = ('abs_rel', 'rmse_mm', 'delta_1_1', 'delta_1_25')
STAT_FIELDS = ('abs_rel_sum', 'squared_error_sum_m2', 'delta_1_1_count', 'delta_1_25_count',
               'oracle_squared_error_sum_m2')
FRAME_FIELDS = ('sample_id', 'sequence_id', 'status', 'failure', 'valid_pixels', 'bad_prediction_pixels',
                *METRICS, 'oracle_median_scale', 'median_gt_m', 'median_pred_m', *STAT_FIELDS)
ORDER = ('Z', 'R', 'A', 'G', 'M')      # the original campaign's methods; only used to list legacy absent methods
LABELS = {'abs_rel': 'AbsRel', 'rmse_mm': 'RMSE (mm)', 'delta_1_1': 'delta<1.1', 'delta_1_25': 'delta<1.25',
          'oracle_median_rmse_mm': 'ORACLE GT-median-scaled RMSE (mm)'}
ORACLE_NOTE = ('ORACLE DIAGNOSTIC: uses per-frame ground-truth median scaling; not a deployable metric and '
               'never used for selection.')


def failed_row(sample_id, sequence_id, reason, valid_pixels, bad_pixels):
    return {'sample_id': sample_id, 'sequence_id': sequence_id, 'status': 'failed', 'failure': reason,
            'valid_pixels': valid_pixels, 'bad_prediction_pixels': bad_pixels}


def no_valid_row(sample_id, sequence_id):
    """A frame with no valid GT pixel: a label-only property, so every method excludes the same rows."""
    return {'sample_id': sample_id, 'sequence_id': sequence_id, 'status': 'no_valid_gt',
            'failure': 'no_valid_gt_pixels', 'valid_pixels': 0, 'bad_prediction_pixels': 0}


def ok_row(metrics, sample_id, sequence_id):
    return {'sample_id': sample_id, 'sequence_id': sequence_id, 'status': 'ok', 'failure': '',
            'bad_prediction_pixels': 0, **metrics}


def method_summary(rows, expected_sequences, metadata) -> dict:
    """Aggregate one method; failed frames stay visible and make the method incomplete."""
    counts = dict(sorted(Counter(r['sequence_id'] for r in rows).items()))
    if counts != dict(sorted(expected_sequences.items())) or len({r['sample_id'] for r in rows}) != len(rows):
        raise ValueError('Method rows do not match the registered per-sequence counts or repeat sample IDs')
    ok = [r for r in rows if r['status'] == 'ok']
    aggregate = aggregate_frames(ok) if ok else None
    evaluated = {s['sequence_id']: s for s in aggregate['sequences']} if aggregate else {}
    sequences = []
    for sid in sorted(expected_sequences):
        mine = [r for r in rows if r['sequence_id'] == sid]
        entry = {'sequence_id': sid, 'total_frames': len(mine),
                 'ok_frames': sum(r['status'] == 'ok' for r in mine),
                 'failed_frames': sum(r['status'] == 'failed' for r in mine),
                 'no_valid_gt_frames': sum(r['status'] == 'no_valid_gt' for r in mine),
                 'gt_valid_pixels': sum(r['valid_pixels'] for r in mine),
                 'evaluated_valid_pixels': sum(r['valid_pixels'] for r in mine if r['status'] == 'ok')}
        entry.update({k: evaluated[sid][k] if sid in evaluated else None for k in METRICS})
        sequences.append(entry)
    failed = [r['sample_id'] for r in rows if r['status'] == 'failed']
    no_valid = [r['sample_id'] for r in rows if r['status'] == 'no_valid_gt']
    return {'metadata': metadata, 'complete': not failed, 'total_frames': len(rows), 'ok_frames': len(ok),
            'failed_frames': len(failed), 'failed_sample_ids': failed,
            'no_valid_gt_frames': len(no_valid), 'no_valid_gt_sample_ids': no_valid, 'sequence_count': len(expected_sequences),
            'sequences_evaluated': len(evaluated), 'gt_valid_pixels': sum(r['valid_pixels'] for r in rows),
            'evaluated_valid_pixels': aggregate['valid_pixels'] if aggregate else 0,
            'headline_definition': 'sequence_macro raw metric; oracle_median_rmse_mm is a labelled diagnostic',
            'sequence_macro': aggregate['sequence_macro'] if aggregate else None,
            'frame_macro': aggregate['frame_macro'] if aggregate else None,
            'pixel_global': aggregate['pixel_global'] if aggregate else None, 'sequences': sequences}


def _write_csv(path, records, fields):
    with Path(path).open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(fields), extrasaction='ignore')
        writer.writeheader()
        for record in records:
            writer.writerow({k: ('' if record.get(k) is None else record[k]) for k in fields})


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def write_method_report(directory, rows, summary):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    frame_records = [{k: r.get(k) for k in FRAME_FIELDS} for r in rows]
    write_json(directory / 'frames.json', frame_records)
    _write_csv(directory / 'frames.csv', frame_records, FRAME_FIELDS)
    write_json(directory / 'sequences.json', summary['sequences'])
    _write_csv(directory / 'sequences.csv', summary['sequences'], summary['sequences'][0])
    write_json(directory / 'summary.json', summary)


def _fmt(value, digits):
    return 'n/a' if value is None else f'{value:.{digits}f}'


def matched_subset(rows_by_method: dict, names=None) -> list:
    """Sequence-macro raw metrics over the rows that are ok for EVERY method (like-for-like rows)."""
    common = set.intersection(*({r['sample_id'] for r in rows if r['status'] == 'ok'} for rows in rows_by_method.values()))
    out = []
    for name in (list(rows_by_method) if names is None else names):
        subset = [r for r in rows_by_method[name] if r['sample_id'] in common]
        aggregate = aggregate_frames(subset) if subset else None
        out.append({'method': name, 'matched_frames': len(subset),
                    'matched_sequences': aggregate['sequence_count'] if aggregate else 0,
                    **{k: aggregate['sequence_macro'][k] if aggregate else None for k in RAW_METRICS}})
    return out


def comparison(summaries: dict, registered, expected_sequences, rows_by_method) -> dict:
    """Tables over the registered methods in registry (spec) order. The legacy Z/R/A/G/M names that a purely legacy
    campaign did not register are listed as 'not registered'; campaigns with other names list none."""
    names = list(registered)
    if set(summaries) != set(names) or len(set(names)) != len(names) or not names:
        raise ValueError('Comparison methods differ from the registry')
    if any(summaries[n]['no_valid_gt_sample_ids'] != summaries[names[0]]['no_valid_gt_sample_ids'] for n in names):
        raise ValueError('no_valid_gt frames differ between methods (must be a label-only property)')
    if set(rows_by_method) != set(summaries):
        raise ValueError('Rows must be supplied for every compared method')
    for name in names:
        counts = {s['sequence_id']: (s['total_frames'], s['gt_valid_pixels']) for s in summaries[name]['sequences']}
        reference = {s['sequence_id']: (s['total_frames'], s['gt_valid_pixels']) for s in summaries[names[0]]['sequences']}
        if counts != reference or summaries[name]['total_frames'] != summaries[names[0]]['total_frames']:
            raise ValueError(f'{name}: evaluated rows/valid-pixel counts differ from {names[0]}')

    def table(level, keys):
        return [{'method': m, 'complete': summaries[m]['complete'], 'frames': summaries[m]['total_frames'],
                 'failed_frames': summaries[m]['failed_frames'], 'no_valid_gt_frames': summaries[m]['no_valid_gt_frames'],
                 'sequences': summaries[m]['sequence_count'],
                 **{k: (summaries[m][level] or {}).get(k) for k in keys}} for m in names]
    per_sequence = []
    for sid in sorted(expected_sequences):
        for m in names:
            s = next(x for x in summaries[m]['sequences'] if x['sequence_id'] == sid)
            per_sequence.append({'sequence_id': sid, 'method': m, **{k: s[k] for k in (
                'total_frames', 'ok_frames', 'failed_frames', 'no_valid_gt_frames', 'gt_valid_pixels', 'evaluated_valid_pixels')},
                **{k: s[k] for k in METRICS}})
    legacy_only = set(names) <= set(ORDER)
    provenance = []
    for m in names:
        meta = summaries[m].get('metadata') or {}
        trained = meta.get('trainable') is not None
        provenance.append({'method': m, 'training_seed': meta.get('training_seed'),
                           'variant': meta.get('trainable') or 'official',
                           'head': ('last' if meta.get('head_choice') == 'last' else 'best') if trained else 'official',
                           'selected_step': meta.get('selected_step')})
    if any('inference_process_hw' in (summaries[m].get('metadata') or {}) or 'output_scale' in (summaries[m].get('metadata') or {})
           for m in names):
        # Diagnostic modifiers (explicit columns only when some arm uses them, so plain campaigns keep their shape).
        for row in provenance:
            meta = summaries[row['method']].get('metadata') or {}
            hw, scale = meta.get('inference_process_hw'), meta.get('output_scale')
            row.update({'inference_process_hw': 'native' if hw is None else f'{hw[0]}x{hw[1]}',
                        'output_scale': '' if scale is None else scale['value'],
                        'output_scale_source': '' if scale is None else scale['source'],
                        'output_scale_sha256': '' if scale is None else scale['sha256']})
    return {'methods': names, 'not_registered': [m for m in ORDER if m not in summaries] if legacy_only else [],
            'provenance': provenance,
            'headline_sequence_macro_raw': table('sequence_macro', RAW_METRICS),
            'secondary_frame_macro_raw': table('frame_macro', RAW_METRICS),
            'secondary_pixel_global_raw': table('pixel_global', RAW_METRICS),
            'matched_subset_sequence_macro_raw': matched_subset(rows_by_method, names),
            'oracle_diagnostic_median_scaled': [
                {'method': r['method'], 'complete': r['complete'], 'frames': r['frames'],
                 'failed_frames': r['failed_frames'], 'no_valid_gt_frames': r['no_valid_gt_frames'],
                 'sequences': r['sequences'], 'sequence_macro_oracle_rmse_mm': r['oracle_median_rmse_mm'], 'note': 'ORACLE DIAGNOSTIC'}
                for r in table('sequence_macro', ('oracle_median_rmse_mm',))],
            'per_sequence': per_sequence}


def _md_table(headers, rows):
    lines = ['| ' + ' | '.join(headers) + ' |', '|' + '|'.join(['---'] * len(headers)) + '|']
    lines += ['| ' + ' | '.join(str(c) for c in row) + ' |' for row in rows]
    return lines


def markdown(tables, banner=None, metadata=None) -> str:
    out = ['# C2 locked evaluation: method comparison', '']
    if banner:
        out += [f'> **{banner}**', '']
    if metadata:
        out += [f'Registry `{metadata["registry_sha256"]}`; evaluator `{metadata["evaluator_version"]}`; '
                f'{metadata["frames"]} frames in {metadata["sequences"]} sequences.', '']
    flag = lambda r: '' if r['complete'] else f' INCOMPLETE ({r["failed_frames"]} failed)'
    out += ['## Method provenance (training seed, model variant, head used)', '']
    provenance = tables.get('provenance', [])
    modified = bool(provenance) and 'inference_process_hw' in provenance[0]
    out += _md_table(['Method', 'Seed', 'Variant', 'Head', 'Step'] + (['Process HW', 'Output scale (source, sha256)'] if modified else []),
                     [[r['method'], 'n/a' if r['training_seed'] is None else r['training_seed'], r['variant'], r['head'],
                       'n/a' if r['selected_step'] is None else r['selected_step']]
                      + ([r['inference_process_hw'], 'none' if r['output_scale'] == '' else
                          f'{r["output_scale"]} ({r["output_scale_source"]}, {r["output_scale_sha256"][:12]})'] if modified else [])
                      for r in provenance])
    out.append('')
    if modified:
        out += ['_DIAGNOSTIC arms: entries with a non-native Process HW and/or an output scale apply an inference-time '
                'modifier to an already-registered head (training and selection unchanged). Process HW: the frame is '
                'resized bicubically to it, K is scaled per axis, and the prediction is bilinearly resized back to the '
                'native GT grid; output scale: the metric prediction is multiplied by the pinned value before every metric '
                'incl. the oracle diagnostic._', '']
    for key, title, note in (
            ('headline_sequence_macro_raw', 'Headline: raw (unaligned) metric depth, equal weight per sequence', None),
            ('secondary_frame_macro_raw', 'Secondary: raw metric, equal weight per frame', None),
            ('secondary_pixel_global_raw', 'Secondary: raw metric, pooled valid pixels', None)):
        out += [f'## {title}', '']
        out += _md_table(['Method', 'AbsRel', 'RMSE (mm)', 'delta<1.1', 'delta<1.25', 'Frames', 'Sequences', 'Failed', 'No valid GT'],
                         [[r['method'] + flag(r), _fmt(r['abs_rel'], 4), _fmt(r['rmse_mm'], 2), _fmt(r['delta_1_1'], 4),
                           _fmt(r['delta_1_25'], 4), r['frames'], r['sequences'], r['failed_frames'], r['no_valid_gt_frames']]
                          for r in tables[key]])
        out.append('')
    out += ['## Matched subset: raw metric, rows that are ok for every method (equal weight per sequence)', '']
    out += _md_table(['Method', 'AbsRel', 'RMSE (mm)', 'delta<1.1', 'delta<1.25', 'Matched frames', 'Matched sequences'],
                     [[r['method'], _fmt(r['abs_rel'], 4), _fmt(r['rmse_mm'], 2), _fmt(r['delta_1_1'], 4),
                       _fmt(r['delta_1_25'], 4), r['matched_frames'], r['matched_sequences']]
                      for r in tables['matched_subset_sequence_macro_raw']])
    out += ['', '## ORACLE diagnostic: GT-median-scaled RMSE (mm), equal weight per sequence', '', f'_{ORACLE_NOTE}_', '']
    out += _md_table(['Method', 'ORACLE scaled RMSE (mm)'],
                     [[r['method'] + flag(r), _fmt(r['sequence_macro_oracle_rmse_mm'], 2)]
                      for r in tables['oracle_diagnostic_median_scaled']])
    out.append('')
    for m in tables['not_registered']:
        out.append(f'- {m}: not registered in this campaign (no result).')
    out += ['', '## Per-sequence raw metrics and counts', '']
    out += _md_table(['Sequence', 'Method', 'Frames', 'Failed', 'No valid GT', 'GT valid px', 'AbsRel', 'RMSE (mm)', 'd<1.1', 'd<1.25',
                      'ORACLE RMSE (mm)'],
                     [[r['sequence_id'], r['method'], r['total_frames'], r['failed_frames'], r['no_valid_gt_frames'], r['gt_valid_pixels'],
                       _fmt(r['abs_rel'], 4), _fmt(r['rmse_mm'], 2), _fmt(r['delta_1_1'], 4), _fmt(r['delta_1_25'], 4),
                       _fmt(r['oracle_median_rmse_mm'], 2)] for r in tables['per_sequence']])
    return '\n'.join(out) + '\n'


def write_comparison(directory, tables, banner=None, metadata=None):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    write_json(directory / 'comparison.json', {'banner': banner, **tables})
    for key, name in (('provenance', 'comparison_provenance'),
                      ('headline_sequence_macro_raw', 'comparison_headline'),
                      ('secondary_frame_macro_raw', 'comparison_frame_macro'),
                      ('secondary_pixel_global_raw', 'comparison_pixel_global'),
                      ('matched_subset_sequence_macro_raw', 'comparison_matched_subset'),
                      ('oracle_diagnostic_median_scaled', 'comparison_oracle_diagnostic'),
                      ('per_sequence', 'comparison_per_sequence')):
        _write_csv(directory / f'{name}.csv', tables[key], tables[key][0])
    (directory / 'comparison.md').write_text(markdown(tables, banner, metadata))
