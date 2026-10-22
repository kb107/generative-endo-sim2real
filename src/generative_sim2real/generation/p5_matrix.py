"""P5 reduced swap-diagnostic matrix: {none, source, target} LoRA x {off, correct, wrong} depth.

  render   (generator venv, GPU)  9 arms x 8 fixed source-validation conditions x 1 seed = 72 images,
           paired initial noise, plus the off-invariance and reproduction tests and 8-row panels
  report   (any venv)             within-adapter direction checks on the DA3 proxy scores (scored by
           generation/p4_proxy.py in the DA3 venv between the two steps) and the failure panel
Rules are declared in configs/p5.yaml before the first run. Nothing is trained or tuned here.
"""
import argparse
import json
import os
import time
from pathlib import Path
from statistics import mean

import torch
import yaml

from . import p4_report as report
from .provenance import provenance


def set_adapter_mode(transformer, controller, adapter):
    """Inference with one named LoRA, or (adapter None) with every LoRA disabled = clean base weights."""
    from peft.tuners.tuners_utils import BaseTunerLayer

    from . import model as gm

    layers = [m for m in transformer.modules() if isinstance(m, BaseTunerLayer)]
    if not layers:
        raise ValueError('No adapter layers are attached')
    for layer in layers:
        layer.enable_adapters(True)
    gm.set_stage(transformer, controller, stage='inference', adapter=adapter or 'source')
    if adapter is None:
        for layer in layers:
            layer.enable_adapters(False)


def set_lora_scale(transformer, adapter, factor):
    """Multiply the named LoRA's alpha/r scaling by `factor` (1.0 restores the trained scale).

    Inference-time only: no weight changes (state hashes are unaffected)."""
    from peft.tuners.tuners_utils import BaseTunerLayer

    touched = 0
    for module in transformer.modules():
        if isinstance(module, BaseTunerLayer) and adapter in getattr(module, 'scaling', {}):
            base = module.__dict__.setdefault('_p5_base_scaling', {})
            base.setdefault(adapter, module.scaling[adapter])
            module.scaling[adapter] = base[adapter] * factor
            touched += 1
    if not touched:
        raise ValueError(f'No LoRA layers found for adapter {adapter!r}')
    return touched


def arm_name(adapter, control):
    return f'{adapter or "none"}_{control}'


def render_images(ctx, adapter, seeds, *, scale, batch, control, guidance=None):
    """control: None (controller off) or control latents [B,4,h,w] (scale 0 = off in disguise)."""
    from . import model as gm

    set_adapter_mode(ctx.transformer, ctx.controller, adapter)
    inference = ctx.config['inference']
    outputs = []
    for i in range(0, len(seeds), batch):
        outputs.append(gm.generate(
            ctx.model, ctx.vae, ctx.scheduler, ctx.prompt, seeds=seeds[i:i + batch],
            steps=inference['steps'],
            guidance=inference['guidance_scale'] if guidance is None else guidance,
            control=None if control is None else control[i:i + batch], control_scale=scale,
            size=ctx.config['resolution'], dtype=ctx.dtype).cpu())
    return torch.cat(outputs)


def matrix_checks(rows, partner, rules, adapters):
    """Within-adapter M1-M4 of configs/p5.yaml from proxy rows (lower error is better).

    Rows carry `<metric>_own` (score against the condition's own depth) and `<metric>_partner`
    (against the wrong-depth partner's depth)."""
    count, need, pos = len(partner), rules['min_conditions'], rules['require_positive_mean_gain']

    def check(gains):
        return report.direction_check(gains, min_conditions=need, require_positive_mean=pos)

    result = {}
    for adapter in adapters:
        def mean_of(control, key):
            return report.condition_means(rows, f'{adapter}_{control}', f'abs_rel_{key}', count)
        off_own, off_partner = mean_of('off', 'own'), mean_of('off', 'partner')
        cor_own = mean_of('correct', 'own')
        wrong_own, wrong_partner = mean_of('wrong', 'own'), mean_of('wrong', 'partner')
        sub = lambda a, b: [x - y for x, y in zip(a, b)]  # noqa: E731  (gain = error saved)
        checks = {
            'M1_correct_beats_off': check(sub(off_own, cor_own)),
            'M2_wrong_beats_off_against_supplied': check(sub(off_partner, wrong_partner)),
            'M3_wrong_tracks_supplied_not_original': check(sub(wrong_own, wrong_partner)),
            'M4_correct_beats_wrong_against_original': check(sub(wrong_own, cor_own))}
        result[adapter] = {
            'checks': checks,
            'control_retained': all(checks[k]['passed'] for k in (
                'M1_correct_beats_off', 'M3_wrong_tracks_supplied_not_original',
                'M4_correct_beats_wrong_against_original')),
            'mean_error': {'off_vs_original': mean(off_own), 'correct_vs_original': mean(cor_own),
                           'wrong_vs_original_depth': mean(wrong_own),
                           'wrong_vs_supplied_depth': mean(wrong_partner),
                           'off_vs_partner_depth': mean(off_partner)}}
    return result


def retention_comparison(per_adapter):
    """Descriptive source-vs-target comparison of per-condition gains (no threshold)."""
    out = {}
    for name in per_adapter['source']['checks']:
        s = per_adapter['source']['checks'][name]
        t = per_adapter['target']['checks'][name]
        out[name] = {
            'mean_gain_source': s['mean_gain'], 'mean_gain_target': t['mean_gain'],
            'ratio_target_over_source': t['mean_gain'] / s['mean_gain'] if s['mean_gain'] else None,
            'conditions_target_gain_larger': sum(a > b for a, b in zip(t['gains'], s['gains'])),
            'paired_difference_source_minus_target': [a - b for a, b in zip(s['gains'], t['gains'])]}
    return out


def swap_verdict(per_adapter, audit_passed):
    if not audit_passed:
        return 'UNINTERPRETABLE (proxy fails its real-RGB audit)'
    target = per_adapter['target']
    if target['control_retained']:
        return 'RETAINED (target adapter satisfies M1, M3, M4)'
    failed = [k for k in ('M1_correct_beats_off', 'M3_wrong_tracks_supplied_not_original',
                          'M4_correct_beats_wrong_against_original')
              if not target['checks'][k]['passed']]
    return 'NOT RETAINED (target adapter fails ' + ', '.join(failed) + ')'


def failure_cells(per_adapter, adapters, count):
    """The `count` (adapter, condition) cells with the lowest M1 gain, regardless of sign."""
    cells = [(per_adapter[a]['checks']['M1_correct_beats_off']['gains'][i], a, i)
             for a in adapters for i in range(len(per_adapter[a]['checks']['M1_correct_beats_off']['gains']))]
    return sorted(cells)[:count]


def make_report(directory, config, *, failure_rows=6):
    directory = Path(directory)
    render = json.loads((directory / 'render.json').read_text())
    proxy = json.loads((directory / 'proxy.json').read_text())
    rules = config['analysis']['rules']
    adapters = [a if a != 'none' else 'none' for a in config['adapters']]
    partner = render['partner']
    rows = proxy['rows']
    own = report.condition_means(rows, 'real', 'abs_rel_own', len(partner))
    other = report.condition_means(rows, 'real', 'abs_rel_partner', len(partner))
    audit = report.direction_check([b - a for a, b in zip(own, other)],
                                   min_conditions=rules['min_conditions'], require_positive_mean=True)
    out = {'render': {k: render[k] for k in ('ids', 'partner', 'seed', 'control_scale', 'tests')},
           'proxy_model': proxy['model'], 'R0_audit_real_rgb': audit}
    for metric, label in (('abs_rel', 'primary_abs_rel'), ('spearman', 'secondary_spearman')):
        if metric == 'abs_rel':
            use = rows
        else:  # higher correlation is better: negate to share the lower-is-better logic
            use = [{**r, 'spearman_own': -r['spearman_own'], 'spearman_partner': -r['spearman_partner']}
                   for r in rows]
        renamed = [{**r, 'abs_rel_own': r[f'{metric}_own'], 'abs_rel_partner': r[f'{metric}_partner']}
                   for r in use] if metric != 'abs_rel' else use
        per_adapter = matrix_checks(renamed, partner, rules, adapters)
        out[label] = {'per_adapter': per_adapter,
                      'retention_source_vs_target': retention_comparison(per_adapter),
                      'swap_verdict': swap_verdict(per_adapter, audit['passed'])}
    cells = failure_cells(out['primary_abs_rel']['per_adapter'], adapters, failure_rows)
    out['failure_cells_lowest_M1_gain'] = [{'adapter': a, 'condition': i, 'M1_gain': g}
                                           for g, a, i in cells]
    out['code_provenance'] = provenance()
    (directory / 'matrix.json').write_text(json.dumps(out, indent=2, allow_nan=False) + '\n')
    make_failure_panel(directory, cells)
    return out


def make_failure_panel(directory, cells):
    from . import gate

    directory = Path(directory)
    load = lambda arm, i: report.png_tensor(directory / 'images' / arm / f'c{i}.png')  # noqa: E731
    render = json.loads((directory / 'render.json').read_text())
    partner = render['partner']
    rows = [[load('real', i), load('control', i), load(arm_name(a, 'off'), i),
             load(arm_name(a, 'correct'), i), load(arm_name(a, 'wrong'), i),
             load('control', partner[i])] for _, a, i in cells]
    gate.save_panel(directory / 'failure_panel.png', rows,
                    ['real rgb', 'control', 'off', 'correct', 'wrong', 'wrong: supplied control'])
    (directory / 'failure_panel_rows.json').write_text(json.dumps(
        [{'row': r, 'adapter': a, 'condition': i, 'M1_gain': g} for r, (g, a, i) in enumerate(cells)],
        indent=1) + '\n')


def run_render(config, args):
    from . import gate
    from . import model as gm
    from .inputs import control_image, encode_image
    from .p4_eval import gate_conditions, load_weights
    from .train import load_p4

    p4, generator = load_p4(config['p4_config'])
    out = Path(config['output_dir'])
    started = time.time()
    ctx = gate.start(generator, Path.cwd())
    inputs = config['inputs']
    loaded = load_weights(ctx, p4, generator, source_lora=inputs['source_lora'],
                          target_lora=inputs['target_lora'], controller=inputs['controller'])
    manifest, ids, partner = gate_conditions(p4, 'validation')
    index = {row['sample_id']: i for i, row in enumerate(manifest.rows)}
    samples = [manifest[index[i]] for i in ids]
    images = torch.stack([control_image(s, manifest.sidecar) for s in samples])
    latents = encode_image(ctx.vae, images, control=True)
    hashes_before = gate.snapshot_hashes(ctx.model, ctx.controller)
    seeds = [config['seed'] + i for i in range(len(ids))]
    scale, batch = config['control_scale'], generator['smoke']['inference_batch']
    lora_scale, guidance = config.get('lora_scale', 1.0), config.get('guidance')
    if lora_scale != 1.0:  # v2: the same LoRA scale for BOTH adapters (matched comparison)
        for adapter in ('source', 'target'):
            set_lora_scale(ctx.transformer, adapter, lora_scale)
    generated, tests = {}, {'off_invariance': {}}
    for adapter in (None if a == 'none' else a for a in config['adapters']):
        generated[arm_name(adapter, 'off')] = render_images(
            ctx, adapter, seeds, scale=scale, batch=batch, control=None, guidance=guidance)
        generated[arm_name(adapter, 'correct')] = render_images(
            ctx, adapter, seeds, scale=scale, batch=batch, control=latents, guidance=guidance)
        generated[arm_name(adapter, 'wrong')] = render_images(
            ctx, adapter, seeds, scale=scale, batch=batch, control=latents[partner], guidance=guidance)
        off = generated[arm_name(adapter, 'off')]
        zero_correct = render_images(ctx, adapter, seeds, scale=0.0, batch=batch, control=latents, guidance=guidance)
        zero_wrong = render_images(ctx, adapter, seeds, scale=0.0, batch=batch,
                                   control=latents[partner], guidance=guidance)
        tests['off_invariance'][adapter or 'none'] = {
            'off_equals_scale0_correct_latents': bool(torch.equal(off, zero_correct)),
            'off_equals_scale0_wrong_latents': bool(torch.equal(off, zero_wrong)),
            'controller_on_differs_from_off_mean_abs': (
                generated[arm_name(adapter, 'correct')] - off).abs().mean().item()}
    tests['adapter_effect_mean_abs'] = {
        'source_vs_none_correct': (generated['source_correct'] - generated['none_correct']).abs().mean().item(),
        'target_vs_none_correct': (generated['target_correct'] - generated['none_correct']).abs().mean().item(),
        'target_vs_source_correct': (generated['target_correct'] - generated['source_correct']).abs().mean().item()}
    tests['weights_unchanged'] = hashes_before == gate.snapshot_hashes(ctx.model, ctx.controller)
    files = []
    for arm, result in generated.items():
        for i in range(len(ids)):
            path = out / 'images' / arm / f'c{i}.png'
            report.save_png(result[i], path)
            files.append({'arm': arm, 'cond': i, 'seed_index': 0, 'seed': seeds[i],
                          'file': str(path.relative_to(out))})
    for i, sample in enumerate(samples):
        report.save_png(sample['rgb'], out / 'images' / 'real' / f'c{i}.png')
        report.save_png(images[i], out / 'images' / 'control' / f'c{i}.png')
        files.append({'arm': 'real', 'cond': i, 'seed_index': None, 'seed': None,
                      'file': f'images/real/c{i}.png'})
    torch.save({'depth_m': torch.stack([s['targets']['depth_m'] for s in samples]),
                'valid': torch.stack([s['targets']['depth_valid'] for s in samples]),
                'K': torch.stack([s['camera']['K'] for s in samples])}, out / 'refs.pt')
    tests['inference_settings'] = {'lora_scale': lora_scale,
                                   'guidance': generator['inference']['guidance_scale'] if guidance is None else guidance}
    tests['reproduction_of_p4'] = (
        reproduction(out, files) if lora_scale == 1.0 and guidance in (None, 4.5)
        else 'not applicable: LoRA scale or guidance differ from the P4 gate renders')
    columns = ['real rgb', 'control', 'wrong: supplied control']
    order = [arm_name(None if a == 'none' else a, c) for a in config['adapters']
             for c in config['controls']]
    panel = [[samples[i]['rgb'], images[i], images[partner[i]]] + [generated[a][i] for a in order]
             for i in range(len(ids))]
    gate.save_panel(out / 'panel_all.png', panel, columns + order)
    for adapter in config['adapters']:
        names = [arm_name(None if adapter == 'none' else adapter, c) for c in config['controls']]
        gate.save_panel(out / f'panel_{adapter}.png',
                        [[samples[i]['rgb'], images[i], images[partner[i]]] + [generated[n][i] for n in names]
                         for i in range(len(ids))], columns + names)
    info = {'ids': ids, 'partner': partner, 'seed': config['seed'], 'seeds_per_condition': seeds,
            'control_scale': scale, 'arms': order, 'files': files, 'weights': loaded, 'tests': tests,
            'inference': generator['inference'], 'wall_seconds': time.time() - started,
            'code_provenance': provenance()}
    gate.write_json(out / 'render.json', info)
    return info


def reproduction(out, files):
    """Compare this run's source arms with seed 0 of the P4 pilot step-2000 gate (8-bit PNGs)."""
    p4 = Path('outputs/p4-appearance/gate_runs/controller_pilot_step2000_validation/images')
    result = {}
    for control in ('off', 'correct', 'wrong'):
        identical, worst = 0, 0.0
        for i in range(8):
            a = report.png_tensor(out / 'images' / f'source_{control}' / f'c{i}.png')
            b = report.png_tensor(p4 / control / f'c{i}_s0.png')
            diff = (a - b).abs().max().item()
            identical += diff == 0
            worst = max(worst, diff)
        result[f'source_{control}'] = {'identical_images': identical, 'of': 8, 'max_abs_diff': worst}
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['render', 'report'])
    parser.add_argument('--config', default='configs/p5.yaml')
    args = parser.parse_args()
    config = yaml.safe_load(Path(args.config).read_text())
    if args.command == 'render':
        result = run_render(config, args)
        print(json.dumps(result['tests'], indent=1))
    else:
        result = make_report(config['output_dir'], config)
        for label in ('primary_abs_rel', 'secondary_spearman'):
            print(label, result[label]['swap_verdict'])
            for adapter, entry in result[label]['per_adapter'].items():
                print('  ', adapter, 'retained' if entry['control_retained'] else 'NOT retained', {
                    k: (v['conditions_in_expected_direction'], round(v['mean_gain'], 3))
                    for k, v in entry['checks'].items()})


if __name__ == '__main__':
    main()
