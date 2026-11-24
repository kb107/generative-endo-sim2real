"""Qwen-Image-Edit-2511 appearance editing of SimCol frames toward two C1 reference frames.

Image 1 = SimCol frame, images 2/3 = C1 references (prompt.PROMPT). Runs in external/foundation-venv.
Output is resized back to 512x512 when the model returns another size; that is recorded per image.
Exploratory open-editor client: not part of the reported results.
  CUDA_VISIBLE_DEVICES=2 external/foundation-venv/bin/python -m generative_sim2real.foundation.qwen_edit \
      --set fixed8 --out outputs/foundation/qwen/fixed8
"""
import argparse
import json
import time
from pathlib import Path

from PIL import Image

from .items import SIZE, load_items
from .prompt import NEGATIVE_PROMPT, PROMPT, PROMPT_SOURCE_LAST

MODEL_ID = 'Qwen/Qwen-Image-Edit-2511'
MODEL_REVISION = '6f3ccc0b56e431dc6a0c2b2039706d7d26f22cb9'


def fit_to_size(image, size=SIZE):
    """Return (image at size x size, record). Non-square outputs are an error, not cropped."""
    record = {'model_output_size': list(image.size), 'resized_back': False, 'resample': None}
    if image.size[0] != image.size[1]:
        raise ValueError(f'non-square model output {image.size}; refusing to crop/pad')
    if image.size != (size, size):
        image = image.resize((size, size), Image.LANCZOS)
        record.update(resized_back=True, resample='LANCZOS')
    return image, record


def load_pipeline(offload='model'):
    import torch
    from diffusers import QwenImageEditPlusPipeline
    pipe = QwenImageEditPlusPipeline.from_pretrained(MODEL_ID, revision=MODEL_REVISION,
                                                     torch_dtype=torch.bfloat16)
    if offload == 'model':
        pipe.enable_model_cpu_offload()
    elif offload == 'none':
        pipe.to('cuda')
    else:
        raise ValueError(offload)
    pipe.set_progress_bar_config(disable=True)
    return pipe


def edit_one(pipe, images, *, seed, steps, cfg, working, prompt=PROMPT):
    """working=None uses the pipeline default (~1 MP, so 1024x1024 for square inputs); an int sets
    the output side and the VAE conditioning area to working^2 (cheaper)."""
    import torch
    from diffusers.pipelines.qwenimage import pipeline_qwenimage_edit_plus as module
    module.VAE_IMAGE_SIZE = (working or 1024) ** 2
    kwargs = {} if working is None else {'height': working, 'width': working}
    with torch.inference_mode():
        out = pipe(image=images, prompt=prompt, negative_prompt=NEGATIVE_PROMPT, true_cfg_scale=cfg,
                   guidance_scale=1.0, num_inference_steps=steps, num_images_per_prompt=1,
                   generator=torch.Generator('cpu').manual_seed(seed), **kwargs)
    return out.images[0]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--set', required=True, choices=['fixed8', 'subset300'])
    parser.add_argument('--inputs', default='outputs/foundation/inputs/inputs.json')
    parser.add_argument('--out', required=True)
    parser.add_argument('--seed', type=int, default=17)
    parser.add_argument('--steps', type=int, default=40)
    parser.add_argument('--cfg', type=float, default=4.0)
    parser.add_argument('--working', type=int, default=None, help='working resolution (default: pipeline ~1 MP)')
    parser.add_argument('--limit', type=int, default=None, help='first N items only (timing probes)')
    parser.add_argument('--order', default='source_first', choices=['source_first', 'source_last'],
                        help='position of the SimCol frame among the input images')
    parser.add_argument('--offload', default='model', choices=['model', 'none'])
    args = parser.parse_args()
    items, refs = load_items(args.inputs, args.set)
    items = items[:args.limit]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    pipe = load_pipeline(args.offload)
    reference_images = [Image.open(p).convert('RGB') for p in refs]
    seconds = []
    for item in items:
        target = out / f'{item["name"]}.png'
        if target.exists() and target.with_suffix('.json').exists():
            continue
        source = Image.open(item['real']).convert('RGB')
        if source.size != (SIZE, SIZE):
            raise ValueError(f'{item["real"]} is {source.size}, expected 512x512')
        start = time.time()
        last = args.order == 'source_last'
        image = edit_one(pipe, [*reference_images, source] if last else [source, *reference_images],
                         seed=args.seed, steps=args.steps, cfg=args.cfg, working=args.working,
                         prompt=PROMPT_SOURCE_LAST if last else PROMPT)
        elapsed = time.time() - start
        image, record = fit_to_size(image)
        image.save(target)
        seconds.append(elapsed)
        target.with_suffix('.json').write_text(json.dumps({
            'key': item['key'], 'sample_id': item['sample_id'], 'input': item['real'], 'refs': refs,
            'model': MODEL_ID, 'revision': MODEL_REVISION, 'seed': args.seed, 'steps': args.steps,
            'true_cfg_scale': args.cfg, 'order': args.order, 'working_resolution': args.working or 'pipeline-default',
            'seconds': round(elapsed, 2), **record}, indent=1) + '\n')
        print(item['name'], f'{elapsed:.1f}s', record, flush=True)
    if seconds:
        (out / 'timing.json').write_text(json.dumps({'n': len(seconds), 'mean_s': sum(seconds) / len(seconds),
                                                     'seconds': seconds}) + '\n')


if __name__ == '__main__':
    main()
