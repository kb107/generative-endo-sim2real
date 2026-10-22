"""Compare a generated image set with a real reference set (and a real-vs-real reference floor).

  PYTHONPATH=src external/generator/.venv/bin/python -m generative_sim2real.generation_metrics.compare \
      --generated outputs/p4-appearance/appearance_review/lora1000/images/target_off \
      --reference outputs/manifests/milestone_v1/target_appearance_validation.jsonl \
      --out outputs/sweeps/gm/target_v1.json

Reports, all on the same two image sets:
  * DINOv2-feature KID (unbiased cubic-polynomial MMD^2) and Frechet distance (labelled DINOv2
    variants: NOT standard Inception FID/KID; Inception weights are not available offline here)
  * per-channel RGB Wasserstein-1 on stride-8 pooled pixels (as appearance_review/reference.json)
  * pixel statistics of each set: pairwise-L1 diversity at 64x64, HSV saturation mean/p95, brightness
  * a reference floor: random generated-sized subsets of real TRAIN frames scored against the same
    validation reference with the same metrics, so a score can be read against the achievable value.
Spread: B random subsamples (default 80%, without replacement) of both sets; with replacement the
unbiased MMD and the covariance would see duplicates. The point estimate uses the full sets.
Features are computed on images resized (bilinear, antialiased) to --image-size; pixel statistics
on the native 512 size. Only RGB is read; C2 / target_test manifests are refused.
"""
import argparse
import json
import os
import time
from pathlib import Path

import torch

from . import images as im
from .distance import channel_w1, frechet_distance, kid
from .features import DEFAULT_MODEL, Dinov2Extractor, describe, describe_cached
from .pixel import summarize_pixels

METRIC_NOTE = ('kid/frechet are DINOv2-feature variants (CLS embedding), not Inception FID/KID. '
               'Frechet is biased upward for n < feature dim and depends on n: compare at equal n.')


def take(desc, index):
    return {k: ([v[i] for i in index] if k == 'ids' else v[index]) for k, v in desc.items()}


def distances(a, b, *, kid_subsets=50, kid_subset_size=100, seed=0, quantiles=1000):
    """All set-to-set distances between two descriptor dicts."""
    return {'kid': kid(a['features'], b['features'], subsets=kid_subsets,
                       subset_size=kid_subset_size, seed=seed),
            'frechet': frechet_distance(a['features'], b['features']),
            'w1_rgb': channel_w1(a['strided'], b['strided'], quantiles=quantiles)}


def scalar_view(d):
    """The headline scalars of a distances() result (for aggregation over draws)."""
    return {'kid': d['kid']['full'], 'frechet': d['frechet'], 'w1_rgb_mean': d['w1_rgb']['mean'],
            'w1_R': d['w1_rgb']['R'], 'w1_G': d['w1_rgb']['G'], 'w1_B': d['w1_rgb']['B']}


def pixel_view(desc):
    s = summarize_pixels(desc)
    return {k: s[k] for k in ('diversity_l1_64', 'sat_mean', 'sat_p95', 'brightness')}


def spread(rows):
    """mean/std/2.5%/97.5% of each scalar over a list of dicts."""
    out = {}
    for key in rows[0]:
        v = torch.tensor([r[key] for r in rows], dtype=torch.float64)
        out[key] = {'mean': v.mean().item(), 'std': v.std().item() if len(v) > 1 else 0.0,
                    'p2.5': v.quantile(0.025).item(), 'p97.5': v.quantile(0.975).item()}
    return out


def subsample(n, fraction, generator):
    return torch.randperm(n, generator=generator)[:max(2, round(n * fraction))]


def compare_sets(gen, ref, floor_pool=None, *, draws=30, fraction=0.8, kid_subsets=50,
                 kid_subset_size=100, seed=0):
    """Pure computation on descriptor dicts (features, pix64, strided, sat_*, brightness)."""
    kw = dict(kid_subsets=kid_subsets, kid_subset_size=kid_subset_size)
    generator = torch.Generator().manual_seed(seed)
    result = {
        'n_generated': len(gen['ids']), 'n_reference': len(ref['ids']),
        'pixel_stats': {'generated': summarize_pixels(gen), 'reference': summarize_pixels(ref)},
        'generated_vs_reference': distances(gen, ref, seed=seed, **kw),
        'metric_note': METRIC_NOTE}
    rows, pix = [], []
    for _ in range(draws):
        gi, ri = subsample(len(gen['ids']), fraction, generator), subsample(
            len(ref['ids']), fraction, generator)
        g, r = take(gen, gi), take(ref, ri)
        rows.append(scalar_view(distances(g, r, seed=seed, **{**kw, 'kid_subsets': 1})))
        pix.append(pixel_view(g))
    result['generated_vs_reference_spread'] = {
        'method': f'{draws} subsamples without replacement, fraction {fraction} of each set',
        'distances': spread(rows), 'generated_pixel_stats': spread(pix)}
    if floor_pool is not None:
        size = min(len(gen['ids']), len(floor_pool['ids']))
        rows, pix = [], []
        for _ in range(draws):
            fi = torch.randperm(len(floor_pool['ids']), generator=generator)[:size]
            f = take(floor_pool, fi)
            rows.append(scalar_view(distances(f, ref, seed=seed, **{**kw, 'kid_subsets': 1})))
            pix.append(pixel_view(f))
        full = distances(take(floor_pool, torch.randperm(
            len(floor_pool['ids']), generator=generator)[:size]), ref, seed=seed, **kw)
        result['floor_train_vs_reference'] = {
            'n_train_per_draw': size, 'n_train_pool': len(floor_pool['ids']),
            'n_reference': len(ref['ids']), 'draws': draws,
            'method': f'{draws} random train subsets of size {size} (without replacement) from the '
                      'evenly spaced train pool vs the FULL reference set',
            'one_draw_full_detail': full, 'distances': spread(rows),
            'train_pixel_stats': spread(pix), 'train_pool_pixel_stats': summarize_pixels(floor_pool)}
        g = scalar_view(result['generated_vs_reference'])
        result['generated_over_floor_ratio'] = {
            k: g[k] / result['floor_train_vs_reference']['distances'][k]['mean']
            for k in g if result['floor_train_vs_reference']['distances'][k]['mean'] > 0}
        if size < len(gen['ids']):
            result['floor_train_vs_reference']['warning'] = (
                'train pool smaller than the generated set; Frechet/KID not exactly size-matched')
    return result


def run(args, extractor=None):
    start = time.time()
    torch.set_num_threads(args.threads)
    data_root = args.data_root or os.environ.get('DATA_ROOT')
    ref_set, split = im.reference_set(args.reference, data_root=data_root,
                                      limit=args.max_reference)
    gen_set = im.generated_set(args.generated, root=args.generated_root, limit=args.max_generated)
    if extractor is None:
        device = 'cuda' if torch.cuda.is_available() and not args.cpu else 'cpu'
        extractor = Dinov2Extractor(args.model, image_size=args.image_size, device=device)
    common = dict(batch_size=args.batch_size, workers=args.workers)
    ref = describe_cached(ref_set, extractor, args.cache_dir, tag=f'ref{args.max_reference}', **common)
    gen = describe(gen_set, extractor, **common)
    floor_pool, floor_manifest, floor_split = None, None, None
    if not args.no_floor:
        floor_manifest = args.floor_train or im.floor_manifest_for(args.reference)
        if floor_manifest is not None:
            pool_set, floor_split = im.reference_set(floor_manifest, data_root=data_root,
                                                     limit=args.floor_pool)
            floor_pool = describe_cached(pool_set, extractor, args.cache_dir,
                                         tag=f'pool{args.floor_pool}', **common)
    result = compare_sets(gen, ref, floor_pool, draws=args.draws, fraction=args.fraction,
                          kid_subsets=args.kid_subsets, kid_subset_size=args.kid_subset_size,
                          seed=args.seed)
    result['setup'] = {
        'generated': gen_set.source, 'generated_n_used': len(gen_set),
        'generated_fingerprint': gen_set.fingerprint(), 'reference': ref_set.source,
        'reference_split': split, 'reference_fingerprint': ref_set.fingerprint(),
        'floor_train_manifest': None if floor_manifest is None else str(floor_manifest),
        'floor_split': floor_split, 'feature_extractor': extractor.name,
        'feature_revision': getattr(extractor, 'revision', None), 'image_size': args.image_size,
        'pixel_size_for_pixel_stats': 512, 'seed': args.seed, 'draws': args.draws,
        'fraction': args.fraction, 'wall_seconds': time.time() - start,
        'inception_fid': 'not computed: Inception weights are not cached/obtainable offline here'}
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(out.suffix + '.tmp')
    tmp.write_text(json.dumps(result, indent=2, sort_keys=True, default=float) + '\n')
    tmp.replace(out)
    return result


def build_parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--generated', required=True, help='directory of PNGs, or a .jsonl manifest')
    p.add_argument('--generated-root', default='outputs', help='root for relative manifest paths')
    p.add_argument('--max-generated', type=int, default=None, help='evenly spaced subset size')
    p.add_argument('--reference', required=True,
                   help='source_validation / target_appearance_validation (or train) manifest .jsonl')
    p.add_argument('--max-reference', type=int, default=None)
    p.add_argument('--floor-train', default=None,
                   help='train manifest for the floor (default: the train manifest beside --reference)')
    p.add_argument('--no-floor', action='store_true')
    p.add_argument('--floor-pool', type=int, default=512, help='evenly spaced train frames in the pool')
    p.add_argument('--out', required=True)
    p.add_argument('--model', default=DEFAULT_MODEL)
    p.add_argument('--image-size', type=int, default=224)
    p.add_argument('--draws', type=int, default=30)
    p.add_argument('--fraction', type=float, default=0.8)
    p.add_argument('--kid-subsets', type=int, default=50)
    p.add_argument('--kid-subset-size', type=int, default=100)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--batch-size', type=int, default=32)
    p.add_argument('--workers', type=int, default=4)
    p.add_argument('--data-root', default=None)
    p.add_argument('--cache-dir', default='outputs/sweeps/gm_cache')
    p.add_argument('--cpu', action='store_true')
    p.add_argument('--threads', type=int, default=8, help='torch CPU threads (shared machine)')
    return p


def main():
    result = run(build_parser().parse_args())
    print(json.dumps({k: result[k] for k in ('n_generated', 'n_reference', 'generated_vs_reference',
                                             'generated_over_floor_ratio') if k in result},
                     indent=1, default=float))


if __name__ == '__main__':
    main()
