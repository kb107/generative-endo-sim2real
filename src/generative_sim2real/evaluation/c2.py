"""Locked C2 evaluation campaign: runs only from a frozen, still-matching registry.

``run`` refuses unless the registry is frozen, every hashed input still matches,
every trained head was selected by source validation only, and the one-time-use
ledger permits the start. Target pixels are first decoded after the ledger
records that C2 inference started. ``--dry-run`` validates everything without
loading a model or decoding any frame. ``source-check`` pushes native
source-validation frames through the identical evaluation function to reconcile
the evaluator with saved training-time reports; it never touches C2.
"""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import shutil
import time

import torch
import torch.nn.functional as F

from . import ledger, report
from .depth import frame_metrics
from .registry import (RegistryError, official_digests, read_json, registry_hash, sha256_file, verify_registry,
                       write_json)

TARGET_SPLIT, TARGET_GEOMETRY = 'target_test', 'target_rectified_full'
CAMPAIGN_DIR = 'outputs/c2_campaign'          # fixed; only run_campaign(campaign_dir=...) (a test hook) can move it


def prediction_failure(prediction, valid):
    """Reason and bad-pixel count when a valid pixel has a nonfinite or nonpositive prediction."""
    values = prediction[valid]
    bad = ~torch.isfinite(values) | (values <= 0)
    if bool(bad.any()):
        return 'nonfinite_or_nonpositive_prediction_at_valid_pixel', int(bad.sum())
    return None


def set_processing(model, hw):
    """Process frames at their native grid (no square squash): the wrapper then leaves RGB and K untouched."""
    from ..depth.model import DA3MetricDepth
    DA3MetricDepth._validate_hw(tuple(hw))
    model.process_hw = tuple(hw)


@torch.no_grad()
def evaluate_dataset(model, dataset, device='cpu', *, split, geometry, hw=None, indices=None, num_workers=0, log=None,
                     exact_grid=False):
    """One code path for C2 and the source-validation cross-check: raw metric depth against the GT grid.

    ``exact_grid`` (C2): the network output must already lie on the GT grid; any mismatch aborts, so no
    anisotropic resize can occur. Otherwise (source validation) the prediction is bilinearly resized to the GT grid.
    """
    if dataset.split != split or dataset.geometry != geometry:
        raise ValueError(f'Expected {split}/{geometry}, got {dataset.split}/{dataset.geometry}')
    indices = list(range(len(dataset))) if indices is None else list(indices)
    was_training = model.training
    model.eval()
    rows = []
    try:
        subset = torch.utils.data.Subset(dataset, indices)
        loader = torch.utils.data.DataLoader(subset, batch_size=None, shuffle=False, num_workers=num_workers,
                                             worker_init_fn=_worker_init if num_workers else None)
        for position, sample in zip(indices, loader):
            if sample['sample_id'] != dataset.rows[position]['sample_id']:
                raise ValueError('Dataset order differs from its manifest rows')
            target, valid = sample['targets']['depth_m'], sample['targets']['depth_valid']
            sid, seq = sample['sample_id'], sample['meta']['sequence_id']
            if hw is not None and (tuple(sample['rgb'].shape[-2:]) != tuple(hw) or tuple(target.shape[-2:]) != tuple(hw)):
                raise ValueError(f'Frame {sid} is not at the registered {hw} geometry')
            if not bool(valid.any()):
                rows.append(report.no_valid_row(sid, seq))      # label-only property: same rows for every method
                continue
            prediction = model(sample['rgb'][None].to(device), sample['camera']['K'][None].to(device))['depth_m']
            if exact_grid:
                if tuple(prediction.shape[-2:]) != tuple(target.shape[-2:]):
                    raise ValueError(f'Prediction grid {tuple(prediction.shape[-2:])} differs from the GT grid '
                                     f'{tuple(target.shape[-2:])}; resizing is not permitted')
                prediction = prediction[0].cpu()
            else:
                prediction = F.interpolate(prediction, size=target.shape[-2:], mode='bilinear',
                                           align_corners=False)[0].cpu()
            failure = prediction_failure(prediction, valid)
            if failure:
                rows.append(report.failed_row(sid, seq, failure[0], int(valid.sum()), failure[1]))
            else:
                rows.append(report.ok_row(frame_metrics(prediction, target, valid), sid, seq))
            if log and len(rows) % 250 == 0:
                log(f'  {len(rows)}/{len(indices)} frames')
    finally:
        model.train(was_training)
    return rows


def _worker_init(worker_id):
    from ..data.workers import configure_worker_cleanup
    configure_worker_cleanup(worker_id)


# ---------------------------------------------------------------- heads

def prepare_heads(model, registry, root):
    """Load and verify every registered head (and backbone LoRA) against the registry before any C2 pixel is touched.

    Returns ``{name: checkpoint-like dict}`` accepted by ``depth.checkpoints.apply_checkpoint``.
    """
    from ..depth.checkpoints import head_digest, read_final_training, read_head
    digests = official_digests(model)
    base = registry['base']
    if digests != {'official_head_sha256': base['official_head_sha256'], 'backbone_sha256': base['backbone_sha256']}:
        raise RegistryError('Loaded official head/backbone weights differ from the registry digests')
    states = {'Z': {'head_state_dict': {k: v.detach().cpu().clone() for k, v in model.head_state_dict().items()}}}
    for record in registry['methods']:
        if record['kind'] != 'trained':
            continue
        path = Path(root) / record['head']['path']
        if sha256_file(path) != record['head']['file_sha256']:
            raise RegistryError(f'{record["name"]}: head file changed')
        last = record.get('head_choice') == 'last'
        if path.name != ('last.pt' if last else 'best.pt'):
            raise RegistryError(f'{record["name"]}: registered head file {path.name} does not match its head choice')
        head = read_final_training(path) if last else read_head(path, allow_backbone_lora=True)
        if head['head_sha256'] != record['head']['head_sha256'] or head['selected_step'] != record['selected_step']:
            raise RegistryError(f'{record["name"]}: head content or selected step differs from the registry')
        if head_digest(head['head_state_dict']) != record['head']['head_sha256']:
            raise RegistryError(f'{record["name"]}: head state digest mismatch')
        recorded_lora = record.get('backbone_lora')
        if (recorded_lora is None) != ('backbone_lora_state_dict' not in head):
            raise RegistryError(f'{record["name"]}: backbone-LoRA presence differs from the registry')
        if recorded_lora is not None:
            if (head_digest(head['backbone_lora_state_dict']) != recorded_lora['sha256']
                    or head['backbone_lora_sha256'] != recorded_lora['sha256']
                    or head['backbone_lora_config'] != recorded_lora['config']):
                raise RegistryError(f'{record["name"]}: backbone-LoRA content or settings differ from the registry')
        states[record['name']] = {k: head[k] for k in ('head_state_dict', 'backbone_lora_state_dict', 'backbone_lora_config')
                                  if k in head}
    return states


# ---------------------------------------------------------------- campaign

def dataset_matches(dataset, target):
    from .registry import ids_digest
    ids = [r['sample_id'] for r in dataset.rows]
    counts = {}
    for r in dataset.rows:
        counts[r['sequence_id']] = counts.get(r['sequence_id'], 0) + 1
    if (len(ids) != target['count'] or ids_digest(ids) != target['ids_sha256']
            or dict(sorted(counts.items())) != target['per_sequence']):
        raise RegistryError('Loaded target_test dataset rows differ from the registry (IDs, count or per-sequence counts)')
    if dataset.split != TARGET_SPLIT or dataset.geometry != TARGET_GEOMETRY:
        raise RegistryError('Dataset is not target_test at the full rectified geometry')


def _real_dataset(registry, root):
    """Builds the real adapter (parses C2 pose files, lists depth/RGB files); decodes no pixel."""
    from ..data.manifests import ManifestDataset
    target = registry['data']['target_test']
    return ManifestDataset(Path(root) / target['path'], os.environ['DATA_ROOT'], purpose='evaluation',
                           geometry=TARGET_GEOMETRY)


def _real_model():
    from ..depth.model import DA3MetricDepth, verify_upstream_checkout
    verify_upstream_checkout()
    return DA3MetricDepth.from_pretrained()


def _device_name(device):
    try:
        return torch.cuda.get_device_name(torch.device(device)) if torch.device(device).type == 'cuda' else 'cpu'
    except Exception:
        return str(device)


def run_campaign(registry_path, *, root, device='cpu', dry_run=False, disclosure=None, expected_sha256=None,
                 code_root=None, num_workers=0, dataset_factory=_real_dataset, model_factory=_real_model,
                 official_digests_fn=None, campaign_dir=None, log=print):
    """``campaign_dir`` is a test hook; the CLI never passes it (ledger/seals/results live in CAMPAIGN_DIR)."""
    from . import registry as reg
    from ..depth.checkpoints import apply_checkpoint
    root = Path(root)
    campaign = Path(campaign_dir) if campaign_dir else root / CAMPAIGN_DIR
    registry = verify_registry(registry_path, root, code_root=code_root, require_frozen=True,
                               expected_sha256=expected_sha256, official_digests_fn=official_digests_fn)
    sha = registry_hash(registry)
    target = registry['data']['target_test']
    runtime = reg.check_runtime(registry)
    dataset = dataset_factory(registry, root)           # builds the adapter and validates manifests; decodes no frame
    dataset_matches(dataset, target)
    output = campaign / sha[:12]
    if dry_run:
        would_refuse = [f'runtime differs from the registry: {m}' for m in runtime]
        try:
            plan = ledger.plan_start(ledger.audit(campaign)['entries'], sha, disclosure)
        except ledger.LedgerError as error:
            plan = {}
            would_refuse.append(str(error))
        return {'dry_run': True, 'registry_sha256': sha, 'methods': registry['method_names'],
                'declared_absent': registry['declared_absent_methods'], 'frames': target['count'],
                'sequences': target['sequence_count'], 'per_sequence': target['per_sequence'],
                'process_hw': registry['protocol']['model_input']['process_hw'],
                'official_digests_recomputed_from_weights_file': True,
                'governing_files_rehashed_on_disk': target['governing_files_rehashed_on_disk'],
                'ledger': plan, 'would_refuse': would_refuse, 'output_dir': str(output), 'model_loaded': False,
                'pixels_decoded': False}
    if runtime:
        raise RegistryError('Runtime differs from the registry: ' + '; '.join(runtime))
    torch.use_deterministic_algorithms(True, warn_only=True)
    model = model_factory()
    set_processing(model, registry['protocol']['model_input']['process_hw'])
    states = prepare_heads(model, registry, root)       # fully verified before the ledger records a start
    model.to(device)
    with ledger.locked(campaign):                        # audit + plan + start are one critical section
        state = ledger.audit(campaign)
        plan = ledger.plan_start(state['entries'], sha, disclosure)
        for orphan in plan['orphaned']:
            ledger.append(campaign, {'event': 'abort', 'registry_sha256': orphan,
                                     'error': 'orphaned: no live process and no complete/abort event'}, held=True)
        if (output / 'COMPLETE').exists():
            raise ledger.LedgerError(f'{output} holds a COMPLETE marker without a ledger complete event; resolve manually')
        if output.exists():
            stale = output.with_name(f'{output.name}.incomplete_{plan["attempt"] - 1}')
            if stale.exists():
                raise ledger.LedgerError(f'{stale} already exists; resolve manually')
            shutil.move(output, stale)
            log(f'Moved an earlier incomplete attempt to {stale}')
        output.mkdir(parents=True, exist_ok=False)
        start = ledger.append(campaign, {'event': 'start', 'registry_sha256': sha, 'registry_path': str(registry_path),
                                         'output_dir': str(output), 'methods': registry['method_names'], **plan}, held=True)
        if not (ledger.seal_dir(campaign) / sha).exists():
            ledger.write_marker(campaign, sha, {'ledger_start_index': start['index'], 'utc': start['utc']})
    log(f'REGISTRY SHA256 {sha}  (copy into docs/implementation-plan.md)')
    log(f'C2 inference start recorded in ledger: experiment {plan["experiment_index"]}, attempt {plan["attempt"]}; '
        f'device {_device_name(device)}')
    try:
        write_json(output / 'registry.json', registry)
        summaries, rows_by_method = {}, {}
        for name in registry['method_names']:
            apply_checkpoint(model, states[name])
            log(f'Evaluating {name} on {target["count"]} C2 frames')
            started = time.time()
            rows = evaluate_dataset(model, dataset, device, split=TARGET_SPLIT, geometry=TARGET_GEOMETRY,
                                    hw=tuple(target['hw']), num_workers=num_workers, log=log, exact_grid=True)
            if [r['sample_id'] for r in rows] != [r['sample_id'] for r in dataset.rows]:
                raise RegistryError(f'{name}: evaluated row IDs differ from the registered test rows')
            summary = report.method_summary(rows, target['per_sequence'], {
                'method': name, 'split': TARGET_SPLIT, 'geometry': TARGET_GEOMETRY, 'registry_sha256': sha,
                'evaluator_version': registry['evaluator_version'],
                **{k: next(r.get(k) for r in registry['methods'] if r['name'] == name)
                   for k in ('selected_step', 'training_seed', 'trainable', 'head_choice')},
                'process_hw': registry['protocol']['model_input']['process_hw'],
                'seconds': round(time.time() - started, 1), 'device': str(device), 'device_name': _device_name(device)})
            report.write_method_report(output / 'methods' / name, rows, summary)
            summaries[name], rows_by_method[name] = summary, rows
        banner = None
        if plan['other_registries']:
            others = ', '.join(h[:12] for h in plan['other_registries'])
            if plan['new_disclosed_experiment']:
                banner = (f'NEW DISCLOSED EXPERIMENT #{plan["experiment_index"]}: C2 inference was previously started '
                          f'under registries {others}. Reason: {plan["disclosure"]}')
            else:
                banner = (f'DISCLOSURE: C2 inference was also started under other registries ({others}); this is '
                          f'experiment #{plan["experiment_index"]}, attempt {plan["attempt"]}.')
        tables = report.comparison(summaries, registry['method_names'], target['per_sequence'], rows_by_method)
        report.write_comparison(output, tables, banner, {'registry_sha256': sha, 'frames': target['count'],
                                                         'sequences': target['sequence_count'],
                                                         'evaluator_version': registry['evaluator_version']})
        digest_files = {str(p.relative_to(output)): sha256_file(p) for p in sorted(output.rglob('*')) if p.is_file()}
        write_json(output / 'campaign.json', {
            'registry_sha256': sha, 'ledger_start': start, 'banner': banner, 'methods': registry['method_names'],
            'complete': all(s['complete'] for s in summaries.values()), 'output_files_sha256': digest_files,
            'runtime': reg.runtime_requirements(), 'device': str(device), 'device_name': _device_name(device)})
        (output / 'COMPLETE').write_text(sha + '\n')
    except BaseException as error:
        ledger.append(campaign, {'event': 'abort', 'registry_sha256': sha, 'error': f'{type(error).__name__}: {error}'[:500]})
        raise
    with ledger.locked(campaign):
        done = ledger.append(campaign, {'event': 'complete', 'registry_sha256': sha, 'output_dir': str(output),
                                        'all_frames_finite': all(s['complete'] for s in summaries.values())}, held=True)
        ledger.write_campaign_seal(campaign, sha, done, ledger.line_hashes(campaign)[-1],
                                   sha256_file(output / 'campaign.json'))
    return {'registry_sha256': sha, 'output_dir': str(output), 'summaries': summaries}


# ---------------------------------------------------------------- source-validation cross-check

def source_check(model, dataset, runs, device, output, *, log=print):
    """Evaluate Z and trained heads on native source validation and compare with saved training-time reports."""
    from ..depth.checkpoints import apply_checkpoint, read_head
    results, official = {}, {'head_state_dict': {k: v.detach().cpu().clone() for k, v in model.head_state_dict().items()}}
    jobs = [('Z', None, None)] + [(Path(run).name, Path(run), read_head(Path(run) / 'best.pt', allow_backbone_lora=True))
                                  for run in runs]
    reference_run = Path(runs[0]) if runs else None
    for name, run, head in jobs:
        apply_checkpoint(model, official if head is None else head)
        rows = evaluate_dataset(model, dataset, device, split='source_validation', geometry='source_native', log=log)
        sequences = {r['sequence_id']: 0 for r in rows}
        for r in rows:
            sequences[r['sequence_id']] += 1
        summary = report.method_summary(rows, sequences, {'method': name, 'split': 'source_validation',
                                                          'purpose': 'evaluator cross-check, not a result'})
        report.write_method_report(Path(output) / name, rows, summary)
        step = 0 if head is None else head['selected_step']
        base = (reference_run if head is None else run) / 'validation' / f'step_{step:06d}'
        saved, saved_frames = read_json(base / 'summary.json'), read_json(base / 'frames.json')
        compare = {}
        for level in ('sequence_macro', 'frame_macro', 'pixel_global'):
            compare[level] = {k: {'new': summary[level][k], 'saved': saved[level][k],
                                  'abs_diff': abs(summary[level][k] - saved[level][k])} for k in summary[level]}
        saved_by_id = {f['sample_id']: f for f in saved_frames}
        if set(saved_by_id) != {r['sample_id'] for r in rows} or len(saved_by_id) != len(rows):
            raise ValueError(f'{name}: evaluated frame IDs differ from the saved report')
        frame_diff = max(abs(r[k] - saved_by_id[r['sample_id']][k]) for r in rows for k in ev_metrics())
        results[name] = {'saved_report_step': step, 'frame_count': len(rows), 'max_frame_metric_abs_diff': frame_diff,
                         'comparison': compare}
    write_json(Path(output) / 'crosscheck.json', results)
    return results


def ev_metrics():
    from .depth import METRICS
    return METRICS


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    run = sub.add_parser('run', help=f'locked C2 campaign (needs a frozen registry; state lives in {CAMPAIGN_DIR})')
    run.add_argument('--registry', required=True)
    run.add_argument('--device', default='cpu')
    run.add_argument('--num-workers', type=int, default=0)
    run.add_argument('--dry-run', action='store_true', help='validate registry and manifests; no model, no pixels')
    run.add_argument('--disclose-new-experiment', default='', help='reason, required when the ledger shows another registry')
    run.add_argument('--expected-registry-sha256', default='')
    check = sub.add_parser('source-check', help='native source-validation cross-check of the evaluator (no C2)')
    check.add_argument('--manifest', default='outputs/manifests/milestone_v1/source_validation.jsonl')
    check.add_argument('--runs', nargs='+', required=True)
    check.add_argument('--output', required=True)
    check.add_argument('--device', default='cpu')
    args = parser.parse_args(argv)
    if args.command == 'run':
        try:
            result = run_campaign(args.registry, root=Path('.').resolve(), device=args.device, dry_run=args.dry_run,
                                  disclosure=args.disclose_new_experiment or None,
                                  expected_sha256=args.expected_registry_sha256 or None, num_workers=args.num_workers)
        except (RegistryError, ledger.LedgerError) as error:
            raise SystemExit(f'REFUSED: {error}')
        if args.dry_run:
            print(json.dumps(result, indent=2))
            raise SystemExit(1 if result['would_refuse'] else 0)
        print(json.dumps({k: v for k, v in result.items() if k != 'summaries'}, indent=2))
        return
    from ..data.manifests import ManifestDataset
    sidecar = read_json(Path(args.manifest).with_suffix('.meta.json'))
    if sidecar.get('split') != 'source_validation':
        raise SystemExit('source-check evaluates only source_validation')
    dataset = ManifestDataset(args.manifest, os.environ['DATA_ROOT'], purpose='evaluation', geometry='source_native')
    torch.use_deterministic_algorithms(True, warn_only=True)
    model = _real_model().to(args.device)
    print(json.dumps(source_check(model, dataset, args.runs, args.device, args.output), indent=2))


if __name__ == '__main__':
    main()
