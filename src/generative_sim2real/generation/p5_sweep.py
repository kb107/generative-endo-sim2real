"""Step A of the v2 follow-up: inference-only sweep of the target LoRA's scale and guidance.

Declared in configs/p5_sweep.yaml before it ran. Unconditional images (controller off) from the same 64
seeds in every cell are scored with the `appearance_review` metrics against C1 TRAIN and C1 validation RGB
(RGB-only loader). The selection rule is applied mechanically by `select_setting`. No C2, no depth, no
training."""
import argparse
import json
import os
import time
from pathlib import Path

import torch
import yaml

from . import gate
from . import p4_report as report
from .inputs import generation_dataset
from .p4_eval import load_weights, nn_stats, pool_features, real_features, summarize
from .p5_matrix import render_images, set_adapter_mode, set_lora_scale
from .provenance import provenance
from .train import load_p4


def select_setting(cells, rule, real_c1_train_diversity, reference=None):
    """Apply the declared rule: constraints against the v1 cell, highest diversity, tie toward v1.

    `reference` is the v1 LoRA's metrics at the v1 cell (step B review of a different LoRA); by default
    it is that cell of `cells` itself (the sweep of the v1 LoRA)."""
    v1 = reference or next(c for c in cells if [c['lora_scale'], c['guidance']] == list(rule['v1_cell']))
    margin = rule['w1_margin']
    for c in cells:
        c['constraints'] = {
            'w1_train_not_worse_than_v1_by_more_than_margin': c['w1_c1_train'] <= v1['w1_c1_train'] + margin,
            'w1_validation_not_worse_than_v1_by_more_than_margin':
                c['w1_c1_validation'] <= v1['w1_c1_validation'] + margin,
            'nn_ratio_latent_at_least_min': c['nn_ratio_latent'] >= rule['nn_ratio_min'],
            'nn_ratio_pixel_at_least_min': c['nn_ratio_pixel'] >= rule['nn_ratio_min'],
            'diversity_at_least_fraction_of_real_c1_train':
                c['diversity'] >= rule['diversity_min_fraction_of_real_c1_train'] * real_c1_train_diversity}
        c['qualifies'] = all(c['constraints'].values())
    qualifying = [c for c in cells if c['qualifies']]
    if not qualifying:
        return None
    best = max(c['diversity'] for c in qualifying)
    tied = [c for c in qualifying if best - c['diversity'] <= 0.005]
    v1_scale, v1_guidance = rule['v1_cell']
    return min(tied, key=lambda c: (abs(c['lora_scale'] - v1_scale), abs(c['guidance'] - v1_guidance)))


def combine_reviews(reviews, tie=0.005):
    """Step B: across several exports (ordered by step), the qualifying cell with the highest diversity
    wins; cells within `tie` of the best are tied and the later step, then the guidance closest to 4.5,
    is taken. `reviews`: [{'step': int, 'cells': [...]}] with `qualifies` set by select_setting."""
    flat = [dict(c, step=r['step']) for r in reviews for c in r['cells'] if c['qualifies']]
    if not flat:
        return None
    best = max(c['diversity'] for c in flat)
    tied = [c for c in flat if best - c['diversity'] <= tie]
    return min(tied, key=lambda c: (-c['step'], abs(c['guidance'] - 4.5)))


def metrics_for(images, ctx, c1_train, held_out, quantiles, count):
    parts = [pool_features(images[i:i + 16], ctx) for i in range(0, len(images), 16)]
    pixel = torch.cat([p[1] for p in parts]).flatten(1)
    latent = torch.cat([p[2] for p in parts])
    strided = torch.cat([p[3] for p in parts])
    result = {'diversity': report.pairwise_diversity(images),
              'w1_c1_train': report.rgb_w1(strided, c1_train['strided'], quantiles=quantiles),
              'w1_c1_validation': report.rgb_w1(strided, held_out['strided'], quantiles=quantiles)}
    for space, features in (('latent', latent), ('pixel', pixel)):
        own, _ = nn_stats(features, c1_train[space])
        real, _ = nn_stats(held_out[space], c1_train[space])
        result[f'nn_median_generated_{space}'] = summarize(own)['median']
        result[f'nn_median_heldout_real_{space}'] = summarize(real)['median']
        result[f'nn_ratio_{space}'] = summarize(own)['median'] / summarize(real)['median']
    return result


def run_sweep(config, out_dir=None, lora_path=None, label='v1'):
    p4, generator = load_p4(config['p4_config'])
    out = Path(out_dir or config['output_dir'])
    started = time.time()
    ctx = gate.start(generator, Path.cwd())
    lora = lora_path or config['target_lora']
    loaded = load_weights(ctx, p4, generator, target_lora=lora)
    hashes_before = gate.snapshot_hashes(ctx.model, ctx.controller)
    data_root = os.environ['DATA_ROOT']
    quantiles = p4['appearance_review']['w1']['quantiles']
    train = real_features(ctx, generation_dataset(p4['manifests']['target_appearance_train'], data_root,
                                                  stage='target_lora'), p4['num_workers'])
    held = real_features(ctx, generation_dataset(p4['manifests']['target_appearance_validation'],
                                                 data_root, stage='validation'), p4['num_workers'])
    from .train import subset_indices
    count = config['images']
    pick = subset_indices(len(train['tile']), count)
    real_div = report.pairwise_diversity(train['tile'][pick].float() / 255)
    pick_val = subset_indices(len(held['tile']), count)
    real_div_val = report.pairwise_diversity(held['tile'][pick_val].float() / 255)
    seeds = [config['seed_base'] + j for j in range(count)]
    batch = generator['smoke']['inference_batch']
    cells, grid_rows = [], []
    for scale in config['settings']['lora_scale']:
        for guidance in config['settings']['guidance']:
            set_lora_scale(ctx.transformer, 'target', scale)
            images = render_images(ctx, 'target', seeds, scale=1.0, batch=batch, control=None,
                                   guidance=guidance)
            cell = {'lora_scale': scale, 'guidance': guidance,
                    **metrics_for(images, ctx, train, held, quantiles, count)}
            cells.append(cell)
            for j in range(16):
                report.save_png(images[j], out / 'images' / f's{scale}_g{guidance}' / f's{j}.png')
            grid_rows.append((scale, guidance, [images[j] for j in range(8)]))
    set_lora_scale(ctx.transformer, 'target', 1.0)
    v1_dir = Path('outputs/p4-appearance/appearance_review/lora1000/images/target_off')
    reproduces = None
    if label == 'v1' and v1_dir.exists():
        same = [torch.equal(report.png_tensor(out / 'images' / 's1.0_g4.5' / f's{j}.png'),
                            report.png_tensor(v1_dir / f's{j}.png')) for j in range(16)]
        reproduces = {'identical_images': sum(same), 'of': 16}
    reference = None
    if config.get('reference_from'):  # step B: constraints are measured against the v1 LoRA's v1 cell
        previous = json.loads(Path(config['reference_from']).read_text())
        reference = next(c for c in previous['cells'] if [c['lora_scale'], c['guidance']]
                         == list(config['selection_rule']['v1_cell']))
    decision = select_setting(cells, config['selection_rule'], real_div, reference)
    gate.save_panel(out / 'sweep_grid.png', [row for _, _, row in grid_rows],
                    [f's{i}' for i in range(8)])
    result = {
        'label': label, 'lora': loaded, 'cells': cells, 'reference_cell': reference,
        'real_diversity': {'c1_train_64': real_div, 'c1_validation_64': real_div_val},
        'selection_rule': config['selection_rule'],
        'selected': None if decision is None else {k: decision[k] for k in ('lora_scale', 'guidance')},
        'rows_in_sweep_grid': [{'lora_scale': s, 'guidance': g} for s, g, _ in grid_rows],
        'v1_cell_reproduces_appearance_review_images': reproduces,
        'weights_unchanged': hashes_before == gate.snapshot_hashes(ctx.model, ctx.controller),
        'wall_seconds': time.time() - started, 'code_provenance': provenance()}
    gate.write_json(out / 'sweep.json', result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='configs/p5_sweep.yaml')
    parser.add_argument('--out')
    parser.add_argument('--lora', help='override the LoRA file (step B review)')
    parser.add_argument('--label', default='v1')
    parser.add_argument('--combine', help='directory holding step_<N>/sweep.json reviews: write decision.json')
    args = parser.parse_args()
    config = yaml.safe_load(Path(args.config).read_text())
    if args.combine:
        reviews = [{'step': int(p.parent.name.split('_')[1]), 'cells': json.loads(p.read_text())['cells']}
                   for p in sorted(Path(args.combine).glob('step_*/sweep.json'))]
        winner = combine_reviews(reviews)
        Path(args.combine, 'decision.json').write_text(json.dumps(
            {'selected': winner, 'reviews': [r['step'] for r in reviews],
             'rule': config['selection_rule']}, indent=1) + '\n')
        print(json.dumps(winner, indent=1))
        return
    result = run_sweep(config, args.out, args.lora, args.label)
    print(json.dumps({k: result[k] for k in ('selected', 'real_diversity', 'v1_cell_reproduces_appearance_review_images',
                                            'weights_unchanged')}, indent=1))
    for c in result['cells']:
        print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in c.items() if k != 'constraints'})


if __name__ == '__main__':
    main()
