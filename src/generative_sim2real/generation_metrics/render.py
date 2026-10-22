"""Render N unconditional images (controller off) from one appearance-LoRA checkpoint (GPU).

  CUDA_VISIBLE_DEVICES=3 PYTHONPATH=src external/generator/.venv/bin/python -m \
      generative_sim2real.generation_metrics.render \
      --lora outputs/p4-appearance/runs/target_lora_v2/checkpoints/step_002000 \
      --n 256 --seed-base 77000 --out outputs/sweeps/images/my_variant

`--lora` is a checkpoint directory (containing lora_source.safetensors or lora_target.safetensors)
or the file itself. The adapter kind, rank and alpha are read from the file's own metadata, so
rank/alpha sweeps work without editing configs. Image j uses seed `seed-base + j` (the review sets
use seed-base 77000, so the first 64 images of a run reproduce appearance_review/*/images/*_off),
the project's neutral prompt, 20 DPM steps, CFG 4.5, bf16, batch 4 (same settings as
generation.p4_eval reference). Writes <out>/s<j>.png (8-bit PNG) and <out>/render.json."""
import argparse
import hashlib
import json
import time
from pathlib import Path

import torch


def resolve_lora(path):
    path = Path(path)
    if path.is_dir():
        found = [p for p in (path / 'lora_source.safetensors', path / 'lora_target.safetensors')
                 if p.exists()]
        if len(found) != 1:
            raise ValueError(f'{path} must hold exactly one of lora_source/lora_target.safetensors')
        path = found[0]
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def read_lora_metadata(path):
    from safetensors import safe_open

    with safe_open(str(path), framework='pt') as handle:
        meta = handle.metadata() or {}
    return {'adapter': meta['adapter'], 'rank': int(meta['rank']), 'alpha': float(meta['alpha'])
            if '.' in meta['alpha'] else int(meta['alpha']), 'raw': meta}


def render_set(lora, *, n, seed_base, out, config_path='configs/p4.yaml', batch=None, guidance=None):
    from generative_sim2real.generation import gate
    from generative_sim2real.generation import model as gm
    from generative_sim2real.generation.checkpoints import file_sha256
    from generative_sim2real.generation.p4_eval import load_weights
    from generative_sim2real.generation.p4_report import save_png
    from generative_sim2real.generation.train import load_p4

    start = time.time()
    path = resolve_lora(lora)
    meta = read_lora_metadata(path)
    if meta['adapter'] not in gm.ADAPTERS:
        raise ValueError(f"Unknown adapter {meta['adapter']!r}")
    config, generator = load_p4(config_path)
    generator = {**generator, 'lora': {**generator['lora'], 'rank': meta['rank'],
                                       'alpha': meta['alpha']}}
    if guidance is not None:
        generator = {**generator, 'inference': {**generator['inference'], 'guidance_scale': guidance}}
    ctx = gate.start(generator, Path.cwd())
    kwargs = {f"{meta['adapter']}_lora": path}
    loaded = load_weights(ctx, config, generator, **kwargs)
    seeds = [seed_base + j for j in range(n)]
    batch = batch or generator['smoke']['inference_batch']
    images = gate.render(ctx, meta['adapter'], None, seeds, scale=config['gate']['control_scale'],
                         batch=batch, control=False)
    out = Path(out)
    for j in range(n):
        save_png(images[j], out / f's{j}.png')
    info = {'lora': loaded, 'adapter': meta['adapter'], 'rank': meta['rank'], 'alpha': meta['alpha'],
            'n': n, 'seed_base': seed_base, 'seeds': [seeds[0], seeds[-1]], 'batch': batch,
            'controller': 'off', 'steps': generator['inference']['steps'],
            'guidance': generator['inference']['guidance_scale'], 'wall_seconds': time.time() - start,
            'peak_gib': torch.cuda.max_memory_allocated() / 2**30,
            'files_sha256': hashlib.sha256(b''.join(
                (out / f's{j}.png').read_bytes() for j in range(n))).hexdigest()}
    gate.write_json(out / 'render.json', info)
    return info


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--lora', required=True)
    p.add_argument('--n', type=int, default=256)
    p.add_argument('--seed-base', type=int, default=77000)
    p.add_argument('--out', required=True)
    p.add_argument('--config', default='configs/p4.yaml')
    p.add_argument('--batch', type=int, default=None)
    p.add_argument('--guidance', type=float, default=None, help='override inference guidance_scale (default: generator.yaml)')
    args = p.parse_args()
    print(json.dumps(render_set(args.lora, n=args.n, seed_base=args.seed_base, out=args.out,
                                config_path=args.config, batch=args.batch, guidance=args.guidance), indent=1))


if __name__ == '__main__':
    main()
