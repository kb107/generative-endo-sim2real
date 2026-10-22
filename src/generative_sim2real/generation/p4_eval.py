"""P4 gate renders and appearance checks (generator venv; GPU).

  render      fixed conditions x {correct, off, wrong} x paired seeds, source LoRA active, written
              as 8-bit PNGs + render.json + refs.pt (reference depth for the DA3 proxy step)
  appearance  unconditional (controller off) images per adapter with fixed shared seeds, pixel-
              distribution distances to the real source-validation and C1-validation frames,
              descriptive diversity, and fixed visual grids (including every seed, no cherry-picking)
The rules applied to these outputs are declared in configs/p4.yaml and implemented in
p4_report.py. Nothing here trains; adapters and controller are loaded read-only.
"""
import argparse
import json
import os
import time
from pathlib import Path

import torch

from . import checkpoints as ck
from . import gate
from . import lowrank_controller as lrc
from . import model as gm
from . import p4_report as report
from .provenance import provenance
from .inputs import FrameTensors, control_image, encode_image, generation_dataset
from .train import load_p4, subset_indices


def load_weights(ctx, config, generator, *, source_lora=None, target_lora=None, controller=None):
    rank, alpha, revision = (generator['lora']['rank'], generator['lora']['alpha'],
                             generator['revision'])
    loaded = {}
    for name, path in (('source', source_lora), ('target', target_lora)):
        if path:
            gm.load_adapter(ctx.transformer, path, name, rank=rank, alpha=alpha, revision=revision)
            loaded[f'lora_{name}'] = {'path': str(path), 'sha256': ck.file_sha256(path)}
    if controller:
        # Dispatches on the file's own format: full controller.safetensors or a low-rank file.
        metadata = lrc.load_any(ctx.controller, controller, revision=revision)
        loaded['controller'] = {'path': str(controller), 'sha256': ck.file_sha256(controller)}
        if metadata.get('kind') == 'lowrank':  # full-controller records stay unchanged
            loaded['controller'].update(lrc.spec_from_metadata(metadata))
    return loaded


def gate_conditions(config, which, runs='runs'):
    """(manifest, ids, partner) for the 8 fixed conditions of the validation or overfit gate."""
    data_root = os.environ['DATA_ROOT']
    if which == 'validation':
        diagnostics = json.loads(Path(config['diagnostics']).read_text())
        manifest = generation_dataset(config['manifests']['source_validation'], data_root,
                                      stage='generation')
        ids = diagnostics[config['gate']['conditions']]
        partner = diagnostics[config['gate']['wrong_depth']]
    elif which == 'overfit':
        pairs = json.loads((Path(config['output_root']) / runs / 'controller_overfit'
                            / 'overfit_pairs.json').read_text())
        manifest = generation_dataset(config['manifests']['source_train'], data_root,
                                      stage='generation')
        ids, partner = pairs['ids'], pairs['wrong_depth_permutation']
    else:
        raise ValueError('set must be validation or overfit')
    if sorted(partner) != list(range(len(ids))) or any(p == i for i, p in enumerate(partner)):
        raise ValueError('Wrong-depth pairing must be a derangement bijection')
    return manifest, ids, partner


def run_render(config, generator, args):
    gcfg = config['gate']
    out = Path(args.out)
    start_time = time.time()
    ctx = gate.start(generator, Path.cwd())
    loaded = load_weights(ctx, config, generator, source_lora=args.source_lora,
                          controller=args.controller)
    manifest, ids, partner = gate_conditions(config, args.set, args.runs)
    index = {row['sample_id']: i for i, row in enumerate(manifest.rows)}
    samples = [manifest[index[i]] for i in ids]
    images = torch.stack([control_image(s, manifest.sidecar) for s in samples])
    latents = encode_image(ctx.vae, images, control=True)
    hashes_before = gate.snapshot_hashes(ctx.model, ctx.controller)
    scale = gcfg['control_scale']
    batch = generator['smoke']['inference_batch']
    generated = {}
    for k, base in enumerate(gcfg['seeds']):
        seeds = [base + i for i in range(len(ids))]
        for arm in gcfg['arms']:
            if arm == 'correct':
                result = gate.render(ctx, 'source', latents, seeds, scale=scale, batch=batch)
            elif arm == 'off':
                result = gate.render(ctx, 'source', latents, seeds, scale=scale, batch=batch,
                                     control=False)
            else:
                result = gate.render(ctx, 'source', latents[partner], seeds, scale=scale,
                                     batch=batch)
            generated[(arm, k)] = result
    files = []
    for (arm, k), result in generated.items():
        for i in range(len(ids)):
            path = out / 'images' / arm / f'c{i}_s{k}.png'
            report.save_png(result[i], path)
            files.append({'arm': arm, 'cond': i, 'seed_index': k, 'seed': gcfg['seeds'][k] + i,
                          'file': str(path.relative_to(out))})
    for i, sample in enumerate(samples):
        report.save_png(sample['rgb'], out / 'images' / 'real' / f'c{i}.png')
        report.save_png(images[i], out / 'images' / 'control' / f'c{i}.png')
        files.append({'arm': 'real', 'cond': i, 'seed_index': None, 'seed': None,
                      'file': f'images/real/c{i}.png'})
    torch.save({'depth_m': torch.stack([s['targets']['depth_m'] for s in samples]),
                'valid': torch.stack([s['targets']['depth_valid'] for s in samples]),
                'K': torch.stack([s['camera']['K'] for s in samples])}, out / 'refs.pt')
    rows = [[samples[i]['rgb'], images[i]] + [generated[(a, 0)][i] for a in gcfg['arms']]
            + [images[partner[i]]] for i in range(len(ids))]
    gate.save_panel(out / 'grid_seed0.png', rows,
                    ['real rgb', 'control'] + [f'{a} (seed0)' for a in gcfg['arms']]
                    + ['wrong: supplied control'])
    gate.save_panel(out / 'grid_seed1.png', [
        [samples[i]['rgb'], images[i]] + [generated[(a, 1)][i] for a in gcfg['arms']]
        + [images[partner[i]]] for i in range(len(ids))],
        ['real rgb', 'control'] + [f'{a} (seed1)' for a in gcfg['arms']]
        + ['wrong: supplied control'])
    gate_info = {
        'set': args.set, 'step': args.step, 'controller': loaded.get('controller'),
        'source_lora': loaded.get('lora_source'), 'ids': ids, 'partner': partner,
        'seeds': gcfg['seeds'], 'control_scale': scale, 'arms': gcfg['arms'],
        'files': files, 'hashes_unchanged_by_rendering': hashes_before == gate.snapshot_hashes(
            ctx.model, ctx.controller), 'wall_seconds': time.time() - start_time,
        'inference': generator['inference'], 'image_format': gcfg['image_format'],
        'code_provenance': provenance()}
    gate.write_json(out / 'render.json', gate_info)
    return gate_info


def real_pixels(manifest, indices, stride):
    from torch.utils.data import DataLoader

    dataset = FrameTensors(manifest, with_control=False, indices=indices)
    chunks = [b['rgb'][..., ::stride, ::stride] for b in DataLoader(
        dataset, batch_size=16, shuffle=False, num_workers=4)]
    return torch.cat(chunks)


def run_hashes(path):
    """Hashes recorded at the end of a training run (final.json) for reload comparison."""
    return json.loads((Path(path) / 'final.json').read_text())['hashes_after']


def swap_recovery(ctx, config, generator, args, latents, seeds, batch, first_source):
    """source -> target -> source on the TRAINED adapters (controller on, fixed seeds): the
    second source render must equal the first within the declared tolerance, and neither the
    controller nor either adapter may change. Also compares the freshly loaded files' weight
    hashes with those recorded when each training run ended (fresh-process reload evidence)."""
    smoke = generator['smoke']
    before = gate.snapshot_hashes(ctx.model, ctx.controller)
    scale = config['gate']['control_scale']
    target = gate.render(ctx, 'target', latents, seeds, scale=scale, batch=batch)
    second = gate.render(ctx, 'source', latents, seeds, scale=scale, batch=batch)
    after = gate.snapshot_hashes(ctx.model, ctx.controller)
    root = Path(config['output_root']) / 'runs'
    recorded = {'lora_source': run_hashes(root / 'source_lora')['lora_source'],
                'lora_target': run_hashes(root / 'target_lora')['lora_target'],
                'controller': run_hashes(root / 'controller_pilot')['controller']}
    return {
        'source_second_vs_first': gate.compare(second, first_source, smoke['recovery_atol'],
                                               smoke['recovery_rtol']),
        'mean_abs_source_vs_target': (first_source - target).abs().mean().item(),
        'hashes_unchanged_by_swaps': before == after,
        'loaded_hashes_equal_training_run_end_hashes': {k: before[k] == v for k, v in recorded.items()},
        'controller_hash': before['controller']}


def run_appearance(config, generator, args):
    acfg = config['appearance']
    out = Path(args.out)
    start_time = time.time()
    ctx = gate.start(generator, Path.cwd())
    loaded = load_weights(ctx, config, generator, source_lora=args.source_lora,
                          target_lora=args.target_lora, controller=args.controller)
    stride, count = acfg['pixel_stride'], acfg['unconditional_images_per_adapter']
    seeds = [acfg['seed_base'] + j for j in range(count)]
    batch = generator['smoke']['inference_batch']
    scale = config['gate']['control_scale']
    unconditional = {a: gate.render(ctx, a, None, seeds, scale=scale, batch=batch, control=False)
                     for a in ('source', 'target')}
    for adapter, images in unconditional.items():
        for j in range(count):
            report.save_png(images[j], out / 'images' / f'{adapter}_off' / f's{j}.png')
    manifest, ids, partner = gate_conditions(config, 'validation')
    index = {row['sample_id']: i for i, row in enumerate(manifest.rows)}
    samples = [manifest[index[i]] for i in ids[:acfg['conditioned_images']]]
    controls = torch.stack([control_image(s, manifest.sidecar) for s in samples])
    latents = encode_image(ctx.vae, controls, control=True)
    conditioned = {a: gate.render(ctx, a, latents, seeds[:len(samples)], scale=scale, batch=batch)
                   for a in ('source', 'target')}
    for adapter, images in conditioned.items():
        for j in range(len(samples)):
            report.save_png(images[j], out / 'images' / f'{adapter}_controller' / f'c{j}.png')

    recovery = swap_recovery(ctx, config, generator, args, latents, seeds[:len(samples)], batch,
                             conditioned['source'])
    data_root = os.environ['DATA_ROOT']
    source_val = real_pixels(manifest, None, stride)
    c1_manifest = generation_dataset(config['manifests']['target_appearance_validation'],
                                     data_root, stage='validation')
    c1_val = real_pixels(c1_manifest, None, stride)
    generated = {a: v[..., ::stride, ::stride] for a, v in unconditional.items()}
    quantiles = acfg['quantiles']
    distances = {
        'source_vs_c1': report.rgb_w1(generated['source'], c1_val, quantiles=quantiles),
        'target_vs_c1': report.rgb_w1(generated['target'], c1_val, quantiles=quantiles),
        'source_vs_source': report.rgb_w1(generated['source'], source_val, quantiles=quantiles),
        'target_vs_source': report.rgb_w1(generated['target'], source_val, quantiles=quantiles),
        'real_c1_vs_real_source': report.rgb_w1(c1_val, source_val, quantiles=quantiles)}
    c1_indices = subset_indices(len(c1_manifest), acfg['c1_validation_panel_frames'])
    c1_frames = [c1_manifest[i]['rgb'] for i in c1_indices]
    spread = subset_indices(len(c1_manifest), count)
    diversity = {
        'source_lora_unconditional': report.pairwise_diversity(unconditional['source']),
        'target_lora_unconditional': report.pairwise_diversity(unconditional['target']),
        'real_c1_validation': report.pairwise_diversity(
            torch.stack([c1_manifest[i]['rgb'] for i in spread[:count]])),
        'real_source_validation': report.pairwise_diversity(torch.stack([
            manifest[i]['rgb'] for i in subset_indices(len(manifest), count)]))}
    # Grids: every row uses a fixed index/seed; nothing is selected after seeing the images.
    gate.save_panel(out / 'grid_source_validation.png', [
        [samples[j]['rgb'], controls[j], conditioned['source'][j], conditioned['target'][j],
         unconditional['source'][j], unconditional['target'][j]] for j in range(len(samples))],
        ['source-val rgb', 'control', 'source LoRA + ctrl', 'target LoRA + ctrl',
         'source LoRA, ctrl off', 'target LoRA, ctrl off'])
    gate.save_panel(out / 'grid_c1_validation.png', [
        [c1_frames[j], unconditional['target'][j], unconditional['source'][j],
         unconditional['target'][len(c1_frames) + j], unconditional['source'][len(c1_frames) + j]]
        for j in range(len(c1_frames))],
        ['C1 validation (real)', 'target LoRA s%d' % 0, 'source LoRA s%d' % 0,
         'target LoRA s%d' % len(c1_frames), 'source LoRA s%d' % len(c1_frames)])
    result = {
        'weights': loaded, 'swap_recovery_trained_adapters': recovery, 'seeds': seeds, 'pixel_stride': stride, 'quantiles': quantiles,
        'w1_rgb_mean': distances, 'checks': report.appearance_checks(distances),
        'diversity_mean_pairwise_l1': diversity,
        'c1_validation_frame_ids': [c1_manifest.rows[i]['sample_id'] for i in c1_indices],
        'source_validation_condition_ids': ids[:len(samples)],
        'real_set_sizes': {'source_validation': len(manifest), 'c1_validation': len(c1_manifest)},
        'wall_seconds': time.time() - start_time, 'code_provenance': provenance()}
    gate.write_json(out / 'appearance.json', result)
    return result


def pool_features(batch, ctx):
    """Frames [B,3,512,512] -> (tile128 uint8, pixel64 float, latent pooled float [B,1024])."""
    import torch.nn.functional as F

    rgb = batch.to(ctx.device)
    tile = F.adaptive_avg_pool2d(rgb, 128)
    latent = encode_image(ctx.vae, rgb, control=True)  # posterior mode, scaled
    return ((tile.clamp(0, 1) * 255).round().byte().cpu(), F.adaptive_avg_pool2d(tile, 64).cpu(),
            F.adaptive_avg_pool2d(latent, 16).flatten(1).cpu(), rgb[..., ::8, ::8].cpu())


def real_features(ctx, manifest, workers):
    from torch.utils.data import DataLoader

    tiles, pixels, latents, strided = [], [], [], []
    for batch in DataLoader(FrameTensors(manifest, with_control=False), batch_size=16,
                            shuffle=False, num_workers=workers):
        tile, pixel, latent, stride = pool_features(batch['rgb'], ctx)
        tiles.append(tile); pixels.append(pixel); latents.append(latent); strided.append(stride)
    return {'tile': torch.cat(tiles), 'pixel': torch.cat(pixels).flatten(1),
            'latent': torch.cat(latents), 'strided': torch.cat(strided),
            'ids': [r['sample_id'] for r in manifest.rows]}


def nn_stats(queries, references, *, exclude_self=False):
    """Per-query nearest-neighbour mean-absolute-difference distance and index."""
    distance = torch.cdist(queries.cuda(), references.cuda(), p=1) / queries.shape[1]
    if exclude_self:
        distance.fill_diagonal_(float('inf'))
    values, index = distance.min(dim=1)
    return values.cpu(), index.cpu()


def summarize(values):
    ordered = values.sort().values
    quantile = lambda q: ordered[min(len(ordered) - 1, int(q * len(ordered)))].item()
    return {'n': len(values), 'mean': values.mean().item(), 'median': ordered[len(ordered) // 2].item(),
            'min': ordered[0].item(), 'p05': quantile(0.05), 'p95': quantile(0.95)}


def run_reference(config, generator, args):
    review = config['appearance_review']
    out = Path(args.out)
    start_time = time.time()
    ctx = gate.start(generator, Path.cwd())
    loaded = load_weights(ctx, config, generator, source_lora=args.source_lora,
                          target_lora=args.target_lora)
    count = review['images_per_adapter']
    seeds = [config['appearance']['seed_base'] + j for j in range(count)]
    batch = generator['smoke']['inference_batch']
    generated = {a: gate.render(ctx, a, None, seeds, scale=config['gate']['control_scale'],
                                batch=batch, control=False) for a in ('source', 'target')}
    for adapter, images in generated.items():
        for j in range(count):
            report.save_png(images[j], out / 'images' / f'{adapter}_off' / f's{j}.png')
    gen_feats = {}
    for adapter, images in generated.items():
        parts = [pool_features(images[i:i + 16], ctx) for i in range(0, count, 16)]
        gen_feats[adapter] = {'tile': torch.cat([p[0] for p in parts]),
                              'pixel': torch.cat([p[1] for p in parts]).flatten(1),
                              'latent': torch.cat([p[2] for p in parts]),
                              'strided': torch.cat([p[3] for p in parts])}
    data_root = os.environ['DATA_ROOT']
    real = {}
    for name, key in review['reference_sets'].items():
        manifest = generation_dataset(config['manifests'][key], data_root,
                                      stage='validation' if 'validation' in name else (
                                          'source_lora' if name == 'source_train' else 'target_lora'))
        real[name] = real_features(ctx, manifest, config['num_workers'])
    quantiles = review['w1']['quantiles']
    w1 = {a: {name: report.rgb_w1(gen_feats[a]['strided'], real[name]['strided'], quantiles=quantiles)
              for name in real} for a in gen_feats}
    w1['real_c1_train_vs_c1_validation'] = report.rgb_w1(
        real['c1_train']['strided'], real['c1_validation']['strided'], quantiles=quantiles)
    w1['real_source_train_vs_source_validation'] = report.rgb_w1(
        real['source_train']['strided'], real['source_validation']['strided'], quantiles=quantiles)
    w1['real_c1_train_vs_source_train'] = report.rgb_w1(
        real['c1_train']['strided'], real['source_train']['strided'], quantiles=quantiles)

    def spread(features):
        pick = subset_indices(len(features), count)
        return report.pairwise_diversity(features[pick].float() / 255)

    diversity = {f'generated_{a}': report.pairwise_diversity(generated[a]) for a in generated}
    diversity.update({f'real_{name}': spread(real[name]['tile']) for name in real})
    pairs = {'source_train': ('source_validation', 'source'), 'c1_train': ('c1_validation', 'target')}
    neighbours, panel_rows = {}, {}
    for train_name, (held_name, adapter) in pairs.items():
        train = real[train_name]
        entry = {}
        for space in ('latent', 'pixel'):
            own, own_index = nn_stats(gen_feats[adapter][space], train[space])
            other = 'target' if adapter == 'source' else 'source'
            cross, _ = nn_stats(gen_feats[other][space], train[space])
            held, _ = nn_stats(real[held_name][space], train[space])
            loo_pick = subset_indices(len(train[space]), 256)
            subset = train[space][loo_pick]
            loo, _ = nn_stats(subset, train[space][loo_pick], exclude_self=True)
            entry[space] = {
                f'{adapter}_lora_generated_vs_own_training': summarize(own),
                f'{other}_lora_generated_vs_this_training_set': summarize(cross),
                f'real_{held_name}_vs_training': summarize(held),
                'leave_one_out_within_256_training_frames_(coarse: subset only)': summarize(loo),
                'median_ratio_generated_over_heldout': summarize(own)['median'] / summarize(held)['median']}
            if space == 'latent':
                panel_rows[adapter] = (own_index, train['tile'])
        neighbours[train_name] = entry
    for adapter, (index, tiles) in panel_rows.items():
        rows = [[generated[adapter][j], tiles[index[j]].float() / 255] for j in range(8)]
        gate.save_panel(out / f'nn_{adapter}_lora.png', rows,
                        [f'{adapter} LoRA, controller off (seed {seeds[0]}+row)',
                         'nearest own-training frame (latent)'])
    result = {'weights': loaded, 'images_per_adapter': count, 'seeds': [seeds[0], seeds[-1]],
              'w1_rgb_mean': w1, 'diversity_mean_pairwise_l1': diversity,
              'nearest_neighbour': neighbours,
              'reference_set_sizes': {n: len(r['ids']) for n, r in real.items()},
              'wall_seconds': time.time() - start_time, 'code_provenance': provenance()}
    gate.write_json(out / 'reference.json', result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    render = sub.add_parser('render')
    render.add_argument('--set', required=True, choices=['validation', 'overfit'])
    render.add_argument('--controller', required=True)
    render.add_argument('--source-lora', required=True)
    render.add_argument('--step', type=int, required=True, help='controller step (label only)')
    render.add_argument('--out', required=True)
    render.add_argument('--runs', default='runs', choices=['runs', 'smoke'],
                        help='where the overfit pairs live (labeled smoke runs use smoke)')
    appearance = sub.add_parser('appearance')
    appearance.add_argument('--source-lora', required=True)
    appearance.add_argument('--target-lora', required=True)
    appearance.add_argument('--controller', required=True)
    appearance.add_argument('--out', required=True)
    reference = sub.add_parser('reference')
    reference.add_argument('--source-lora', required=True)
    reference.add_argument('--target-lora', required=True)
    reference.add_argument('--out', required=True)
    for p in (render, appearance, reference):
        p.add_argument('--config', default='configs/p4.yaml')
    args = parser.parse_args()
    config, generator = load_p4(args.config)
    runner = {'render': run_render, 'appearance': run_appearance, 'reference': run_reference}[args.command]
    result = runner(config, generator, args)
    print(json.dumps({k: v for k, v in result.items() if k not in ('files',)}, indent=1,
                     default=float)[:4000])


if __name__ == '__main__':
    main()
