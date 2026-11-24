"""SimuScope-style baseline: SD1.5 img2img of SimCol RGB with a C1 style LoRA and SoftEdge+Depth ControlNets.

A diffusers re-implementation of the pipeline of Martyniak et al., WACV 2025 (arXiv 2412.02332,
github SanoScience/SimuScope). It is NOT the official AUTOMATIC1111 run; settings that differ from
the official ones are noted in configs/simuscope.yaml. The output is one 512x512 image per frozen downstream ID, paired with the
unchanged SimCol depth, using the bulk-generation bookkeeping (shards, atomic writes, manifest) and the
rgb_modes G/M validation unchanged.

  select      deterministic family-diverse choice of the style-LoRA frames (RGB manifest rows only)
  train-lora  export the frames as a kohya dataset and run the official kohya command (kohya venv)
  run         render missing/invalid images (generator venv, one GPU)
  manifest    build + validate generated_manifest.jsonl (any venv with the source data)
  verify      re-check every image without rendering
"""
import argparse
import hashlib
import json
import os
import subprocess
from pathlib import Path

import yaml

from generative_sim2real.data import rgb_modes as rm
from generative_sim2real.generation import bulk

METHOD = 'SimuScope-style'
FORBIDDEN_LABEL_KEYS = ('depth', 'pose')   # C1 adaptation rows must carry no label references
SELECTION_FILE = 'style_frames.json'
NO_SOURCE_ADAPTER = 'none (SimuScope-style img2img of SimCol RGB; there is no source adapter)'


def load_config(path):
    return yaml.safe_load(Path(path).read_text())


# ------------------------------------------------------------ frame choice ----

def family(sequence_id):
    """c1_ascending_t3_v2 -> ascending_t3 (all versions of one segment/texture family)."""
    return '_'.join(sequence_id.split('_')[1:-1])


def select_style_frames(rows, count=85):
    """Deterministic family-diverse choice: equal quota per family (remainder to evenly spaced families),
    midpoints of an even stride through each family's frames ordered by sample id. Uses manifest rows only."""
    if not rows:
        raise ValueError('No C1 rows to select from')
    for row in rows:
        if any(k in key for key in row for k in FORBIDDEN_LABEL_KEYS if key != 'rgb_path'):
            raise ValueError(f'C1 adaptation row carries label fields: {sorted(row)}')
    groups = {}
    for row in sorted(rows, key=lambda r: r['sample_id']):
        groups.setdefault(family(row['sequence_id']), []).append(row)
    names = sorted(groups)
    count = min(count, sum(len(v) for v in groups.values()))
    base, extra = divmod(count, len(names))
    lucky = {names[int(i * len(names) / extra)] for i in range(extra)} if extra else set()
    quota = {name: base + (name in lucky) for name in names}
    chosen = []
    for name in names:
        members, q = groups[name], min(quota[name], len(groups[name]))
        chosen += [members[(2 * j + 1) * len(members) // (2 * q)] for j in range(q)] if q else []
    return chosen[:count]


def selection_record(rows, manifest_path, count):
    chosen = select_style_frames(rows, count)
    return {'schema_version': 1, 'method': METHOD,
            'source_manifest': str(manifest_path), 'source_manifest_sha256': bulk.file_sha256(manifest_path),
            'rule': 'equal quota per family, remainder to evenly spaced families, midpoints of an even stride '
                    'through each family ordered by sample_id; manifest rows only, no depth/pose touched',
            'requested': count, 'count': len(chosen),
            'families': {name: sum(family(r['sequence_id']) == name for r in chosen)
                         for name in sorted({family(r['sequence_id']) for r in chosen})},
            'frames': [{'sample_id': r['sample_id'], 'sequence_id': r['sequence_id'],
                        'family': family(r['sequence_id']), 'frame_index': r['frame_index'],
                        'rgb_path': r['rgb_path']} for r in chosen]}


def write_selection(config, root='.'):
    style = config['style_lora']
    manifest = Path(root) / style['manifest']
    rows = [json.loads(line) for line in manifest.read_text().splitlines()]
    record = selection_record(rows, style['manifest'], style['frames'])
    path = Path(root) / style['work_dir'] / SELECTION_FILE
    bulk.atomic_json(path, record)
    return path, record


# ------------------------------------------------------------ LoRA training ----

KOHYA_FLAGS = [   # the official README command for CholectG45/CholectL45, flag for flag
    '--network_train_unet_only', '--bucket_no_upscale', '--bucket_reso_steps=64', '--cache_latents',
    '--cache_latents_to_disk', '--enable_bucket', '--min_bucket_reso=256', '--max_bucket_reso=2048',
    '--learning_rate=0.0012', '--lr_scheduler=constant_with_warmup',
    '--lr_scheduler_num_cycles=20', '--max_data_loader_n_workers=0', '--max_grad_norm=1',
    '--resolution=512,512', '--min_snr_gamma=5', '--mixed_precision=fp16', '--network_alpha=1',
    '--network_dim=128', '--network_module=networks.lora', '--optimizer_type=Adafactor',
    '--save_model_as=safetensors', '--save_precision=fp16', '--text_encoder_lr=0.0012',
    '--train_batch_size=5', '--unet_lr=0.0012']
# Replaced or dropped: --xformers (not installed; --sdpa is the same attention), --gradient_checkpointing
# (recompute-only memory saver, dropped for speed; 48 GB suffices), --logging_dir (no tensorboard),
# --save_every_n_epochs=1 (replaced by --save_every_n_steps=2000; the final LoRA is the one used).


def export_kohya_dataset(config, selection, data_root, root='.'):
    """Write the chosen C1 frames as PNGs into <work_dir>/img/<repeats>_<caption>/ through the C1 RGB loader
    (generation_dataset 'target_lora' role + FrameTensors without control: no depth/pose is indexed or decoded)."""
    import numpy as np
    from PIL import Image
    from generative_sim2real.generation.inputs import FrameTensors, generation_dataset

    style = config['style_lora']
    manifest_path = Path(root) / style['manifest']
    data = generation_dataset(manifest_path, data_root, stage='target_lora')
    lookup = {r['sample_id']: i for i, r in enumerate(data.rows)}
    wanted = [f['sample_id'] for f in selection['frames']]
    frames = FrameTensors(data, with_control=False, indices=[lookup[s] for s in wanted])
    # kohya DreamBooth folders "<repeats>_<instance token>" caption every image with the folder's token text when no
    # caption files exist; that is how the official "cholect45 style" prompt arises. Same mechanism, our caption.
    out = Path(root) / style['work_dir'] / 'img' / f'{style["repeats"]}_{style["caption"]}'
    out.mkdir(parents=True, exist_ok=True)
    for k, sid in enumerate(wanted):
        item = frames[k]
        assert item['sample_id'] == sid
        array = (item['rgb'].clamp(0, 1) * 255).round().byte().permute(1, 2, 0).numpy()
        assert array.shape == (512, 512, 3)
        name = f'{k:03d}'
        Image.fromarray(array).save(out / f'{name}.png')
    return out.parent


def kohya_command(config, img_dir, base_snapshot, root='.'):
    style, root = config['style_lora'], Path(root).resolve()
    work = root / style['work_dir']
    return [str(root / style['kohya_python']), '-m', 'accelerate.commands.launch',
            '--num_cpu_threads_per_process=8', '--mixed_precision=fp16', '--num_processes=1',
            str(root / style['kohya_root'] / 'train_network.py'), *KOHYA_FLAGS, '--sdpa',
            '--save_every_n_steps=2000',
            f'--max_train_steps={style["max_train_steps"]}', f'--seed={style["seed"]}',
            f'--output_dir={work / "model"}', f'--output_name={style["output_name"]}',
            f'--pretrained_model_name_or_path={base_snapshot}', f'--train_data_dir={img_dir}']


def base_snapshot(config):
    from huggingface_hub import snapshot_download
    base = config['base']
    return snapshot_download(base['model_id'], revision=base['revision'], local_files_only=True)


def train_lora(config, data_root, root='.'):
    style = config['style_lora']
    selection = json.loads((Path(root) / style['work_dir'] / SELECTION_FILE).read_text())
    img_dir = export_kohya_dataset(config, selection, data_root, root).resolve()
    command = kohya_command(config, img_dir, base_snapshot(config), root)
    work = Path(root).resolve() / style['work_dir']
    (work / 'kohya_command.txt').write_text(' '.join(command) + '\n')
    kohya = Path(root).resolve() / style['kohya_root']
    subprocess.run(command, check=True, cwd=kohya, env={**os.environ, 'PYTHONPATH': str(kohya)})
    return work / 'model' / f'{style["output_name"]}.safetensors'


# ------------------------------------------------------- recipe / bookkeeping ----

def official_seed(recipe_seed, sample_id):
    """The official notebook renders every image with one fixed seed; bulk would derive one per ID."""
    return recipe_seed


def locate_files(config, root='.'):
    """Hash every weight file that determines a pixel (needs the HF cache and the trained LoRA)."""
    from huggingface_hub import hf_hub_download
    style, nets = config['style_lora'], config['controlnets']
    work = Path(root) / style['work_dir']
    paths = {'target_adapter': work / 'model' / f'{style["output_name"]}.safetensors',
             'style_frames': work / SELECTION_FILE}
    for key in ('softedge', 'depth'):
        paths[f'controlnet_{key}'] = Path(hf_hub_download(
            nets['repo'], nets[f'{key}_file'], revision=nets['revision'], local_files_only=True))
    files = {}
    for key, path in paths.items():
        if not path.is_file():
            raise bulk.BulkError(f'Missing file for {key}: {path}')
        files[key] = {'path': str(path), 'sha256': bulk.file_sha256(path)}
    return files


def resolve(config, files, *, purpose):
    """The `resolved` object bulk.generate/build_manifest/verify consume (bulk.load_recipe is bypassed because
    it is specific to the PixArt LoRA + controller layout)."""
    inf, nets, base = config['inference'], config['controlnets'], config['base']
    label = lambda key: f'{Path(files[key]["path"]).name}#sha256={files[key]["sha256"]}'
    provenance = {
        'base_model': base['model_id'], 'base_revision': base['revision'],
        'source_adapter': NO_SOURCE_ADAPTER, 'target_adapter': label('target_adapter'),
        'adapter_weight': float(inf['lora_weight']),
        'controller': (f'{nets["repo"]}@{nets["revision"]}: softedge {label("controlnet_softedge")} (HED), '
                       f'depth {label("controlnet_depth")} (MiDaS) of the SimCol RGB, not SimCol depth'),
        'controller_scale': [float(inf['softedge_weight']), float(inf['depth_weight'])],
        'prompt': inf['prompt'],
        'scheduler': f'DPMSolverMultistepScheduler(dpmsolver++, solver_order=2, karras_sigmas) = {inf["scheduler"]}',
        'inference_steps': int(inf['steps']), 'precision': inf['precision'], 'size_hw': [bulk.SIZE, bulk.SIZE],
        # Extra, method-specific fields (rgb_modes only checks GENERATED_PROVENANCE):
        'method': METHOD, 'denoising_strength': float(inf['denoising_strength']),
        'negative_prompt': inf['negative_prompt'], 'style_frames_sha256': files['style_frames']['sha256'],
        'annotators': f'{config["annotators"]["repo"]}@{config["annotators"]["revision"]}',
        'hed_safe': bool(config['annotators']['hed_safe']),
        'reference_controlnet': 'dropped (no diffusers support)'}
    if not set(rm.GENERATED_PROVENANCE) - {'seed'} <= set(provenance):
        raise bulk.BulkError('Provenance misses rgb_modes.GENERATED_PROVENANCE fields')
    seed, guidance = int(inf['seed']), float(inf['cfg_scale'])
    defining = {'seed': seed, 'guidance': guidance, 'batch': 1, 'provenance': provenance,
                'fingerprint_version': bulk.FINGERPRINT_VERSION}
    return {'recipe': {'method': METHOD, 'config': config}, 'files': files, 'provenance': provenance,
            'defining': defining, 'guidance': guidance, 'seed': seed, 'batch': 1,
            'recipe_hash': rm.digest(defining)[:16], 'purpose': purpose}


class SimuScopeLayout(bulk.Layout):
    """<out_root>/<subdir>/{recipe.json, shards/..., generated_manifest.jsonl}; manifest image paths are
    relative to out_root (= outputs/), which is what RGBModeDataset(outputs_root='outputs') requires."""

    def __init__(self, out_root, subdir):
        self.out_root = Path(out_root)
        self.directory = self.out_root / subdir


def environment():
    env = bulk.environment_fingerprint()
    from importlib import metadata
    for package in ('controlnet_aux', 'safetensors'):
        try:
            env['versions'][package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            env['versions'][package] = None
    code = hashlib.sha256(env['code_sha256'].encode() + Path(__file__).read_bytes()).hexdigest()
    return {**env, 'code_sha256': code, 'code_files': env['code_files'] + [str(Path(__file__).name)]}


# ----------------------------------------------------------------- GPU path ----

def to_pil(rgb):
    from PIL import Image
    return Image.fromarray((rgb.detach().cpu().float().clamp(0, 1) * 255).round().byte().permute(1, 2, 0).numpy())


def jpeg_roundtrip(pil):
    """The official notebook sends the init image to the WebUI as cv2.imencode('.jpg') bytes (default quality 95);
    the init image and the ControlNet input image are that decoded JPEG. Reproduced here."""
    import cv2
    import numpy as np
    from PIL import Image
    ok, buffer = cv2.imencode('.jpg', cv2.cvtColor(np.asarray(pil), cv2.COLOR_RGB2BGR))
    if not ok:
        raise bulk.TechnicalFailure('JPEG encoding of the init image failed')
    return Image.fromarray(cv2.cvtColor(cv2.imdecode(buffer, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB))


def load_renderer(config, files, device='cuda'):
    """Closure (samples, seeds) -> [B,3,512,512] float in [0,1]: HED(safe) and MiDaS of the SimCol RGB drive the
    two ControlNets; img2img from the SimCol RGB itself; the style LoRA at lora_weight."""
    import torch
    from controlnet_aux import HEDdetector, MidasDetector
    from diffusers import ControlNetModel, DPMSolverMultistepScheduler, StableDiffusionControlNetImg2ImgPipeline
    from huggingface_hub import hf_hub_download
    from safetensors.torch import load_file

    inf, nets, ann = config['inference'], config['controlnets'], config['annotators']
    dtype = getattr(torch, inf['precision'])
    arch = nets['architecture_config']
    architecture = hf_hub_download(arch['repo'], 'config.json', revision=arch['revision'], local_files_only=True)
    controlnets = []
    for key in ('softedge', 'depth'):
        net = ControlNetModel.from_config(json.loads(Path(architecture).read_text()))
        net.load_state_dict(load_file(files[f'controlnet_{key}']['path']))
        controlnets.append(net.to(dtype))
    base = base_snapshot(config)
    pipe = StableDiffusionControlNetImg2ImgPipeline.from_pretrained(
        base, controlnet=controlnets, torch_dtype=dtype, safety_checker=None, requires_safety_checker=False,
        local_files_only=True)
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(
        pipe.scheduler.config, algorithm_type='dpmsolver++', solver_order=2, use_karras_sigmas=True)
    lora = Path(files['target_adapter']['path'])
    pipe.load_lora_weights(str(lora.parent), weight_name=lora.name, adapter_name='style')
    pipe.set_adapters(['style'], [float(inf['lora_weight'])])
    pipe.to(device)
    pipe.set_progress_bar_config(disable=True)
    hed = HEDdetector.from_pretrained(ann['repo'], local_files_only=True).to(device)
    midas = MidasDetector.from_pretrained(ann['repo'], local_files_only=True).to(device)
    scales = [float(inf['softedge_weight']), float(inf['depth_weight'])]

    def render(samples, seeds):
        images = []
        for sample, seed in zip(samples, seeds):
            pil = jpeg_roundtrip(to_pil(sample['rgb']))
            edges = hed(pil, detect_resolution=512, image_resolution=512, safe=bool(ann['hed_safe']))
            depth = midas(pil, detect_resolution=512, image_resolution=512)
            generator = torch.Generator(device).manual_seed(int(seed))
            out = pipe(prompt=inf['prompt'], negative_prompt=inf['negative_prompt'] or None, image=pil,
                       control_image=[edges, depth], height=512, width=512, strength=float(inf['denoising_strength']),
                       num_inference_steps=int(inf['steps']), guidance_scale=float(inf['cfg_scale']),
                       controlnet_conditioning_scale=scales, generator=generator, output_type='pt')
            images.append(out.images[0].float().cpu())
        return torch.stack(images)
    render.pipe, render.hed, render.midas = pipe, hed, midas
    return render


# ------------------------------------------------------------------- CLI ----

def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('command', choices=['select', 'train-lora', 'run', 'manifest', 'verify'])
    parser.add_argument('--config', default='configs/simuscope.yaml')
    parser.add_argument('--ids', default='outputs/manifests/milestone_v1/downstream_ids.json')
    parser.add_argument('--source', default='outputs/manifests/milestone_v1/source_train.jsonl')
    parser.add_argument('--smoke', action='store_true', help='write below the smoke directory')
    parser.add_argument('--limit', type=int, default=None, help='first N IDs; with --smoke only')
    parser.add_argument('--allow-environment-change', action='store_true')
    parser.add_argument('--no-manifest', action='store_true')
    args = parser.parse_args()
    config = load_config(args.config)
    if args.command == 'select':
        path, record = write_selection(config)
        print(json.dumps({'file': str(path), 'count': record['count'], 'families': len(record['families'])}))
        return
    if args.command == 'train-lora':
        print(train_lora(config, os.environ['DATA_ROOT']))
        return
    purpose = 'smoke' if args.smoke else 'production'
    out = config['output']
    layout = SimuScopeLayout(out['out_root'], out['smoke_dir' if args.smoke else 'production_dir'])
    bulk.check_destination(purpose, layout.directory)
    bulk.per_id_seed = official_seed          # fixed seed for every image, as in the official notebook
    resolved = resolve(config, locate_files(config), purpose=purpose)
    source = bulk.open_source(args.source, os.environ['DATA_ROOT'])
    ids = bulk.select_ids(args.ids, source, args.limit, purpose)
    if args.command == 'run':
        import torch
        render = load_renderer(config, resolved['files'])
        summary = bulk.generate(resolved, source, ids, layout, render, environment=environment(),
                                allow_environment_change=args.allow_environment_change)
        summary.update(recipe_hash=resolved['recipe_hash'], purpose=purpose,
                       vram_max_gib=torch.cuda.max_memory_allocated() / 2**30)
        bulk.atomic_json(layout.directory / 'run_summary.json', summary)
        print(json.dumps(summary, indent=1))
        if summary['failed'] or args.no_manifest:
            raise SystemExit(2 if summary['failed'] else 0)
    if args.command in ('run', 'manifest'):
        print(json.dumps(bulk.build_manifest(resolved, source, ids, layout), indent=1))
    if args.command == 'verify':
        result = bulk.verify(resolved, source, ids, layout)
        print(json.dumps(result, indent=1))
        raise SystemExit(0 if not result['invalid_count'] else 1)


if __name__ == '__main__':
    main()
