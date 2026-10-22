"""P3 integration gate: real-model smoke, fresh-process reload, and throughput profile.

Run through scripts/{smoke,profile}_generator.sh. Stages:
  smoke   assemble, train a few updates per stage, audit, swap, save, trace (writes references)
  reload  fresh process: rebuild, load the three saved files, compare with smoke references
  profile warmed ~100 updates per stage and 16 inference images; VRAM, timing, ETA
Results are smoke/profile evidence, not validation of the scientific claim."""
import argparse
import gc
import hashlib
import json
import os
import platform
import statistics
import subprocess
import time
from pathlib import Path

import torch
import yaml

from generative_sim2real.data.preprocessing import InverseDepthEncoding

from . import model as gm
from .inputs import (FrameTensors, conditioning_trace, encode_image, encode_prompts, endless,
                     generation_dataset, load_prompt_cache, save_prompt_cache, tensor_trace,
                     training_loader)
from .training import train_updates

DTYPES = {'bfloat16': torch.bfloat16, 'float32': torch.float32}


def validate_config(config):
    """Every key either has an effect or must equal the one supported value."""
    def need(condition, message):
        if not condition:
            raise ValueError(f'Unsupported generator config: {message}')
    need(config['backend'] == 'pixart_alpha', 'backend must be pixart_alpha')
    need(config['resolution'] == 512, 'resolution must be 512')
    need(config['precision'] in DTYPES, 'precision must be bfloat16 or float32')
    need(config['lora']['target_modules'] == gm.LORA_MODULES,
         f'lora.target_modules must be {gm.LORA_MODULES}')
    need(config['controller']['initialization'] == 'clean_base_first_blocks',
         'controller.initialization')
    need(config['controller']['latent_policy'] == 'posterior_mode', 'controller.latent_policy')
    need(config['training']['rgb_latent_policy'] == 'posterior_sample',
         'training.rgb_latent_policy')
    need(config['inference']['scheduler'] == 'DPMSolverMultistepScheduler',
         'inference.scheduler')
    accumulation_for(config['training'], config['training']['micro_batch'])
    seeds = config['training']['seeds']
    need(set(seeds['adapter_init']) == set(gm.ADAPTERS), 'seeds.adapter_init')
    roles = {'source_lora', 'target_lora', 'controller'}
    need(set(seeds['loader']) == roles and set(seeds['noise']) == roles, 'seeds per stage')
    values = list(seeds['loader'].values()) + list(seeds['noise'].values())
    need(len(set(values)) == len(values), 'loader/noise seeds must all differ')


def load_config(path):
    config = yaml.safe_load(Path(path).read_text())
    validate_config(config)
    return config


def accumulation_for(training, micro):
    effective = training['effective_batch']
    if micro < 1 or effective % micro:
        raise ValueError(f'effective_batch {effective} is not divisible by micro batch {micro}')
    return effective // micro


def now():
    return time.perf_counter()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True, default=float) + '\n')
    tmp.replace(path)


def environment_info(config):
    import diffusers
    import peft
    import transformers

    smi = subprocess.run(
        ['nvidia-smi', '--query-gpu=name,driver_version,memory.total', '--format=csv,noheader'],
        capture_output=True, text=True).stdout.strip().splitlines()
    code = Path(__file__).parent
    return {
        'python': platform.python_version(), 'torch': torch.__version__,
        'cuda_runtime': torch.version.cuda, 'diffusers': diffusers.__version__,
        'transformers': transformers.__version__, 'peft': peft.__version__,
        'visible_gpus': os.environ.get('CUDA_VISIBLE_DEVICES'), 'nvidia_smi': smi,
        'host': platform.node(), 'model_id': config['model_id'], 'revision': config['revision'],
        'precision': config['precision'],
        'generation_code_sha256': {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                                   for p in sorted(code.glob('*.py'))},
    }


def snapshot_hashes(model, controller):
    return {'backbone_base': gm.state_hash(model.transformer, exclude_lora=True),
            'lora_source': gm.state_hash(model.transformer, only_lora='source'),
            'lora_target': gm.state_hash(model.transformer, only_lora='target'),
            'controller': gm.state_hash(controller)}


def vram(device):
    return {'allocated_gib': torch.cuda.memory_allocated(device) / 2**30,
            'peak_allocated_gib': torch.cuda.max_memory_allocated(device) / 2**30,
            'peak_reserved_gib': torch.cuda.max_memory_reserved(device) / 2**30}


class Context:
    """Everything a stage needs, built once with startup timings."""


def snapshot_path(config):
    """Local directory of the pinned revision (downloaded by scripts/setup_generator.sh)."""
    from huggingface_hub import snapshot_download

    path = Path(snapshot_download(config['model_id'], revision=config['revision'],
                                  local_files_only=True))
    if path.name != config['revision']:
        raise RuntimeError(f'Cached snapshot {path.name} is not the pinned revision')
    return path


def build_prompt_cache(config, device, timings):
    smoke = config['smoke']
    path = Path(smoke['prompt_cache'])
    if not path.exists():
        from transformers import T5EncoderModel, T5Tokenizer

        start = now()
        snapshot = snapshot_path(config)
        tokenizer = T5Tokenizer.from_pretrained(snapshot / 'tokenizer')
        encoder = T5EncoderModel.from_pretrained(
            snapshot / 'text_encoder', variant='fp16', torch_dtype=torch.float16
        ).to(device).eval()
        timings['text_encoder_load_s'] = now() - start
        before = torch.cuda.memory_allocated(device)
        positive = encode_prompts(tokenizer, encoder, [config['prompt']], device)
        negative = encode_prompts(tokenizer, encoder, [''], device)
        timings['text_encoder_resident_gib'] = before / 2**30
        timings['text_encoder_peak_gib'] = torch.cuda.max_memory_allocated(device) / 2**30
        save_prompt_cache(path, positive, negative, prompt=config['prompt'],
                          revision=config['revision'])
        del encoder
        gc.collect()
        torch.cuda.empty_cache()
        timings['gpu_gib_after_text_encoder_released'] = torch.cuda.memory_allocated(device) / 2**30
        timings['text_encoder_built_and_released_s'] = now() - start
    start = now()
    prompt = load_prompt_cache(path, prompt=config['prompt'], revision=config['revision'],
                               device=device)
    timings['prompt_cache_load_s'] = now() - start
    return prompt


def start(config, root, *, device='cuda'):
    from diffusers import (AutoencoderKL, DDPMScheduler, DPMSolverMultistepScheduler,
                           PixArtTransformer2DModel)

    ctx = Context()
    ctx.config, ctx.device, ctx.timings = config, torch.device(device), {}
    ctx.dtype = DTYPES[config['precision']]
    ctx.root = Path(root)
    snapshot = snapshot_path(config)
    t0 = now()
    ctx.prompt = build_prompt_cache(config, ctx.device, ctx.timings)
    start_load = now()
    transformer = PixArtTransformer2DModel.from_pretrained(
        snapshot / 'transformer', torch_dtype=torch.float32)
    ctx.timings['transformer_load_s'] = now() - start_load
    start_vae = now()
    ctx.vae = AutoencoderKL.from_pretrained(
        snapshot / 'vae', torch_dtype=torch.float32).to(ctx.device).eval().requires_grad_(False)
    ctx.scheduler = DPMSolverMultistepScheduler.from_pretrained(
        snapshot / 'scheduler')
    ctx.train_scheduler = DDPMScheduler.from_config(ctx.scheduler.config)  # training noise only
    ctx.timings['vae_scheduler_load_s'] = now() - start_vae
    start_assemble = now()
    ctx.base_sha256 = gm.state_hash(transformer)
    ctx.model, ctx.controller, ctx.provenance = gm.assemble(
        transformer, controller_layers=config['controller']['num_layers'],
        rank=config['lora']['rank'], alpha=config['lora']['alpha'],
        seeds=config['training']['seeds']['adapter_init'], dtype=ctx.dtype)
    ctx.model.to(ctx.device)
    ctx.transformer = ctx.model.transformer
    ctx.timings['assemble_to_device_s'] = now() - start_assemble
    ctx.timings['startup_total_s'] = now() - t0
    ctx.timings['gpu_gib_after_startup'] = torch.cuda.memory_allocated(ctx.device) / 2**30
    return ctx


def datasets(ctx, stage_role):
    root = os.environ['DATA_ROOT']
    smoke = ctx.config['smoke']
    paths = {'source_lora': smoke['source_manifest'], 'controller': smoke['source_manifest'],
             'target_lora': smoke['target_manifest']}
    manifest = generation_dataset(paths[stage_role], root, stage=stage_role)
    return FrameTensors(manifest, with_control=stage_role == 'controller')


def conditions(ctx, count, *, reduced):
    """Fixed source-validation conditions chosen by the P1 diagnostics (no generations seen)."""
    config = ctx.config
    diagnostics = json.loads(Path(config['smoke']['diagnostics']).read_text())
    if reduced:
        ids = diagnostics['reduced_gate_sample_ids'][:count]
    else:
        ids = diagnostics['sample_ids'][::2][:count]
    manifest = generation_dataset(
        str(Path(config['smoke']['source_manifest']).with_name('source_validation.jsonl')),
        os.environ['DATA_ROOT'], stage='generation')
    index = {row['sample_id']: i for i, row in enumerate(manifest.rows)}
    samples = [manifest[index[i]] for i in ids]
    ctx.encoding = InverseDepthEncoding.from_dict(
        manifest.sidecar['depth_control_encoding']['constants'])
    images = torch.stack([ctx.encoding.encode(s['targets']['depth_m'], s['targets']['depth_valid'])
                          for s in samples])
    start_t = now()
    latents = encode_image(ctx.vae, images, control=True)
    torch.cuda.synchronize()
    return {'ids': ids, 'samples': samples, 'control_images': images, 'latents': latents,
            'encode_seconds': now() - start_t, 'sidecar': manifest.sidecar}


def render(ctx, adapter, latents, seeds, *, scale=None, batch=None, control=True):
    gm.set_stage(ctx.transformer, ctx.controller, stage='inference', adapter=adapter)
    inference = ctx.config['inference']
    if scale is None:
        scale = ctx.config['controller']['conditioning_scale']
    batch = batch or len(seeds)
    outputs = []
    for i in range(0, len(seeds), batch):
        outputs.append(gm.generate(
            ctx.model, ctx.vae, ctx.scheduler, ctx.prompt, seeds=seeds[i:i + batch],
            steps=inference['steps'], guidance=inference['guidance_scale'],
            control=latents[i:i + batch] if control else None, control_scale=scale,
            size=ctx.config['resolution'], dtype=ctx.dtype).cpu())
    return torch.cat(outputs)


def compare(a, b, atol, rtol):
    """Elementwise |a-b| <= atol + rtol*|b|; thresholds were declared in the config."""
    diff = (a - b).abs()
    bound = atol + rtol * b.abs()
    return {'max_abs_error': diff.max().item(), 'mean_abs_error': diff.mean().item(),
            'violations': int((diff > bound).sum()), 'elements': diff.numel(),
            'atol': atol, 'rtol': rtol, 'pass': bool((diff <= bound).all())}


def save_panel(path, rows, columns):
    from PIL import Image, ImageDraw

    size = 256
    sheet = Image.new('RGB', (size * len(columns), (size + 14) * len(rows)), '#202020')
    draw = ImageDraw.Draw(sheet)
    for r, row in enumerate(rows):
        for c, tensor in enumerate(row):
            array = (tensor.permute(1, 2, 0).clamp(0, 1) * 255).round().byte().numpy()
            tile = Image.fromarray(array).resize((size, size))
            sheet.paste(tile, (c * size, r * (size + 14) + 14))
            if r == 0:
                draw.text((c * size + 3, 1), columns[c], fill='white')
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    sheet.save(path)


def diagnostic_saturation(config):
    """Near/far clip rates and median depth over the 32 fixed diagnostic frames."""
    ids = json.loads(Path(config['smoke']['diagnostics']).read_text())['sample_ids']
    manifest = generation_dataset(
        str(Path(config['smoke']['source_manifest']).with_name('source_validation.jsonl')),
        os.environ['DATA_ROOT'], stage='generation')
    encoding = InverseDepthEncoding.from_dict(
        manifest.sidecar['depth_control_encoding']['constants'])
    index = {row['sample_id']: i for i, row in enumerate(manifest.rows)}
    near, far, median = [], [], []
    for sample_id in ids:
        target = manifest[index[sample_id]]['targets']
        depth = target['depth_m'][target['depth_valid'] & (target['depth_m'] > 0)]
        near.append((depth <= encoding.near_m).float().mean().item())
        far.append((depth >= encoding.far_m).float().mean().item())
        median.append(depth.median().item())
    return {'frames': len(ids), 'near_clip_fraction': stats(near), 'far_clip_fraction': stats(far),
            'median_depth_m': stats(median)}


def loader_control_equality(ctx):
    """A source_train frame through FrameTensors + DataLoader (workers) must equal the
    inference-path frame/control image, and give the same VAE control latent."""
    from torch.utils.data import DataLoader

    from generative_sim2real.data.workers import configure_worker_cleanup

    from .inputs import control_image

    dataset = datasets(ctx, 'controller')
    batch = next(iter(DataLoader(dataset, batch_size=2, shuffle=False, num_workers=2,
                                 worker_init_fn=configure_worker_cleanup)))
    reference = generation_dataset(ctx.config['smoke']['source_manifest'],
                                   os.environ['DATA_ROOT'], stage='generation')
    samples = [reference[i] for i in range(2)]
    images = torch.stack([control_image(s, reference.sidecar) for s in samples])
    latent_loader = encode_image(ctx.vae, batch['control'].to(ctx.device), control=True)
    latent_reference = encode_image(ctx.vae, images.to(ctx.device), control=True)
    return {'sample_ids': list(batch['sample_id']),
            'ids_match_reference': list(batch['sample_id']) == [s['sample_id'] for s in samples],
            'control_image_equal': bool(torch.equal(batch['control'], images)),
            'rgb_equal': bool(torch.equal(batch['rgb'], torch.stack([s['rgb'] for s in samples]))),
            'control_latent_equal': bool(torch.equal(latent_loader, latent_reference))}


def stage_run(ctx, role, adapter, stage, updates, *, micro, accumulation, lr, use_control,
              probe):
    """Select the stage, train, and report trainable audit plus which tensors changed."""
    training = ctx.config['training']
    named = gm.set_stage(ctx.transformer, ctx.controller, stage=stage, adapter=adapter)
    audit = gm.parameter_audit(named)
    before_hashes = snapshot_hashes(ctx.model, ctx.controller)
    before = gm.snapshot(named)
    seeds = training['seeds']
    loader = training_loader(datasets(ctx, role), batch_size=micro,
                             seed=seeds['loader'][role],
                             workers=ctx.config['smoke']['num_workers'])
    torch.cuda.reset_peak_memory_stats(ctx.device)
    records, evidence = train_updates(
        ctx.model, ctx.vae, ctx.train_scheduler, ctx.prompt, named, endless(loader),
        updates=updates,
        accumulation=accumulation, lr=lr, weight_decay=training['weight_decay'],
        max_grad_norm=training['max_grad_norm'], use_control=use_control,
        seed=seeds['noise'][role], device=ctx.device, dtype=ctx.dtype, probe_gradients=probe)
    after_hashes = snapshot_hashes(ctx.model, ctx.controller)
    changed = gm.changed_names(before, named)
    del loader
    return {'role': role, 'stage': stage, 'adapter': adapter, 'updates': updates,
            'trainable': audit, 'trainable_changed_tensors': len(changed),
            'hash_changed': {k: before_hashes[k] != after_hashes[k] for k in before_hashes},
            'hashes_after': after_hashes, 'loss_first': records[0]['loss'],
            'loss_last': records[-1]['loss'], 'gradient_evidence': evidence,
            'mean_update_seconds': statistics.mean(r['seconds'] for r in records),
            'vram': vram(ctx.device)}, records


def run_smoke(config, root):
    smoke = config['smoke']
    out = Path(smoke['output_dir'])
    out.mkdir(parents=True, exist_ok=True)
    results = {'stage': 'smoke', 'environment': environment_info(config)}
    ctx = start(config, root)
    results['startup'] = ctx.timings
    results['controller_provenance'] = ctx.provenance
    expected, modules = gm.expected_lora_elements(ctx.transformer, config['lora']['rank'])
    results['lora_expected'] = {'elements_per_adapter': expected, 'target_linears': modules,
                                'tensors_per_adapter': 2 * modules}
    results['parameters'] = {
        'backbone_elements': sum(p.numel() for n, p in ctx.transformer.named_parameters()
                                 if 'lora_' not in n),
        'controller_elements': sum(p.numel() for p in ctx.controller.parameters()),
        'controller_dtype': str(next(ctx.controller.parameters()).dtype),
        'backbone_dtype': str(next(p for n, p in ctx.transformer.named_parameters()
                                   if 'lora_' not in n).dtype)}
    results['hashes_initial'] = snapshot_hashes(ctx.model, ctx.controller)
    micro, workers = smoke['smoke_micro_batch'], smoke['num_workers']
    training = config['training']
    stages = {}
    kwargs = dict(micro=micro, accumulation=1)
    stages['source_lora'], _ = stage_run(
        ctx, 'source_lora', 'source', 'lora', smoke['lora_updates'], lr=training['lora_lr'],
        use_control=False, probe=True, **kwargs)
    stages['target_lora'], _ = stage_run(
        ctx, 'target_lora', 'target', 'lora', smoke['lora_updates'], lr=training['lora_lr'],
        use_control=False, probe=True, **kwargs)
    stages['controller'], _ = stage_run(
        ctx, 'controller', 'source', 'controller', smoke['controller_updates'],
        lr=training['controller_lr'], use_control=True, probe=True, **kwargs)
    results['stages'] = stages
    # Stage isolation, stated as checks against the hashes before each stage.
    results['isolation'] = {
        'source_stage_changed_only_source_lora': (
            stages['source_lora']['hash_changed'] ==
            {'backbone_base': False, 'lora_source': True, 'lora_target': False,
             'controller': False}),
        'target_stage_changed_only_target_lora': (
            stages['target_lora']['hash_changed'] ==
            {'backbone_base': False, 'lora_source': False, 'lora_target': True,
             'controller': False}),
        'controller_stage_changed_only_controller': (
            stages['controller']['hash_changed'] ==
            {'backbone_base': False, 'lora_source': False, 'lora_target': False,
             'controller': True}),
        'initial_base_hash_unchanged_at_end': (
            results['hashes_initial']['backbone_base'] == stages['controller']['hashes_after'][
                'backbone_base'])}

    # Conditions, trace, inference, swap.
    cond = conditions(ctx, 2, reduced=True)
    seeds = [config['inference']['seed'] + i for i in range(len(cond['ids']))]
    results['conditions'] = {'ids': cond['ids'], 'seeds': seeds}
    results['range_trace'] = conditioning_trace(
        ctx.vae, ctx.encoding, cond['samples'][0], scratch=out / 'control_trace.pt')
    pos = ctx.transformer.pos_embed(cond['latents'][:1].to(ctx.device, ctx.dtype))
    results['range_trace']['patch_embedding_of_control_latent'] = tensor_trace(pos)
    rgb = encode_image(ctx.vae, cond['samples'][0]['rgb'], control=False,
                       generator=torch.Generator(device=ctx.device).manual_seed(0))
    results['range_trace']['rgb_training_latent_for_same_frame'] = tensor_trace(rgb)
    results['range_trace']['source_rgb_input'] = tensor_trace(cond['samples'][0]['rgb'])
    results['range_trace']['encode_seconds_for_conditions'] = cond['encode_seconds']
    results['diagnostic_frames_control_saturation'] = diagnostic_saturation(config)
    results['loader_vs_inference_control'] = loader_control_equality(ctx)

    atol, rtol = smoke['recovery_atol'], smoke['recovery_rtol']
    hashes_before_swap = snapshot_hashes(ctx.model, ctx.controller)
    source_1 = render(ctx, 'source', cond['latents'], seeds)
    source_repeat = render(ctx, 'source', cond['latents'], seeds)
    target = render(ctx, 'target', cond['latents'], seeds)
    source_2 = render(ctx, 'source', cond['latents'], seeds)
    controller_off = render(ctx, 'source', cond['latents'], seeds, control=False)
    shuffled = render(ctx, 'source', cond['latents'].flip(0), seeds)
    zero_scale_a = render(ctx, 'source', cond['latents'], seeds, scale=0.0)
    zero_scale_b = render(ctx, 'source', cond['latents'].flip(0), seeds, scale=0.0)
    hashes_after_swap = snapshot_hashes(ctx.model, ctx.controller)
    effect = smoke['min_adapter_effect']
    results['inference'] = {
        'fixed_seed_repeat': compare(source_repeat, source_1, atol, rtol),
        'recovery_source_target_source': compare(source_2, source_1, atol, rtol),
        'adapter_effect_mean_abs_source_vs_target': (source_1 - target).abs().mean().item(),
        'adapter_effect_threshold': effect,
        'controller_effect_mean_abs_on_vs_off': (source_1 - controller_off).abs().mean().item(),
        'wrong_depth_effect_mean_abs': (source_1 - shuffled).abs().mean().item(),
        'controller_off_scale0_ignores_depth': bool(torch.equal(zero_scale_a, zero_scale_b)),
        'controller_off_scale0_equals_no_controller': bool(
            torch.equal(zero_scale_a, controller_off)),
        'hashes_unchanged_across_swaps': hashes_before_swap == hashes_after_swap,
        'controller_hash_before': hashes_before_swap['controller'],
        'controller_hash_after': hashes_after_swap['controller'],
        'image_range': {'min': source_1.min().item(), 'max': source_1.max().item(),
                        'finite': bool(torch.isfinite(source_1).all())},
        'tolerance_note': 'recovery_atol/rtol were declared in configs/generator.yaml '
                          'before any test of this code and were not changed',
    }
    results['inference']['adapter_effect_ok'] = (
        results['inference']['adapter_effect_mean_abs_source_vs_target'] > effect)

    # Save separate files and references for the fresh-process reload stage.
    for adapter in gm.ADAPTERS:
        gm.save_adapter(ctx.transformer, out / f'lora_{adapter}.safetensors', adapter,
                        rank=config['lora']['rank'], alpha=config['lora']['alpha'],
                        revision=config['revision'])
    gm.save_controller(ctx.controller, out / 'controller.safetensors', revision=config['revision'])
    torch.save({'source': source_1, 'target': target, 'ids': cond['ids'], 'seeds': seeds,
                'hashes': hashes_after_swap}, out / 'references.pt')
    results['saved_files'] = {p.name: {'bytes': p.stat().st_size,
                                       'sha256': hashlib.sha256(p.read_bytes()).hexdigest()}
                              for p in sorted(out.glob('*.safetensors'))}
    rows = [[cond['samples'][i]['rgb'], cond['control_images'][i], source_1[i], target[i],
             controller_off[i], shuffled[i]] for i in range(len(seeds))]
    save_panel(out / 'smoke_panel.png', rows,
               ['source rgb', 'control', 'source LoRA+ctrl', 'target LoRA+ctrl', 'ctrl off',
                'wrong depth'])
    results['peak_vram'] = vram(ctx.device)
    write_json(out / 'smoke.json', results)
    return results


def run_reload(config, root):
    smoke = config['smoke']
    out = Path(smoke['output_dir'])
    reference = torch.load(out / 'references.pt', weights_only=True)
    ctx = start(config, root)
    for adapter in gm.ADAPTERS:
        gm.load_adapter(ctx.transformer, out / f'lora_{adapter}.safetensors', adapter,
                        rank=config['lora']['rank'], alpha=config['lora']['alpha'],
                        revision=config['revision'])
    gm.load_controller(ctx.controller, out / 'controller.safetensors',
                       revision=config['revision'])
    hashes = snapshot_hashes(ctx.model, ctx.controller)
    cond = conditions(ctx, len(reference['ids']), reduced=True)
    assert cond['ids'] == reference['ids']
    seeds = reference['seeds']
    atol, rtol = smoke['recovery_atol'], smoke['recovery_rtol']
    source = render(ctx, 'source', cond['latents'], seeds)
    target = render(ctx, 'target', cond['latents'], seeds)
    results = {
        'stage': 'reload', 'environment': environment_info(config), 'startup': ctx.timings,
        'fresh_process_hashes_match_saved_run': {
            k: hashes[k] == reference['hashes'][k]
            for k in ('lora_source', 'lora_target', 'controller')},
        'source_reload_error': compare(source, reference['source'], atol, rtol),
        'target_reload_error': compare(target, reference['target'], atol, rtol),
        'rejects_wrong_file_kind': True}
    try:
        gm.load_controller(ctx.controller, out / 'lora_source.safetensors',
                           revision=config['revision'])
        results['rejects_wrong_file_kind'] = False
    except ValueError:
        pass
    write_json(out / 'reload.json', results)
    return results


def stats(values):
    ordered = sorted(values)
    return {'mean': statistics.mean(values), 'median': statistics.median(values),
            'stdev': statistics.pstdev(values), 'min': ordered[0], 'max': ordered[-1],
            'p95': ordered[min(len(ordered) - 1, int(0.95 * len(ordered)))]}


def run_profile(config, root):
    smoke, training = config['smoke'], config['training']
    out = Path(smoke['output_dir'])
    results = {'stage': 'profile', 'environment': environment_info(config)}
    log = Path('outputs/generator-integration/logs/download.log')
    if log.exists():
        text = log.read_text()
        results['download_seconds_logged'] = float(
            text.rsplit('download_seconds=', 1)[1].split()[0]) if 'download_seconds=' in text else None
    ctx = start(config, root)
    results['startup'] = ctx.timings
    updates, warm = smoke['profile_updates'], smoke['warmup_updates']
    micro = training['micro_batch']
    accumulation = accumulation_for(training, micro)
    profile = {}
    for name, role, adapter, stage, lr, control in (
            ('lora', 'source_lora', 'source', 'lora', training['lora_lr'], False),
            ('controller', 'controller', 'source', 'controller', training['controller_lr'], True)):
        torch.cuda.empty_cache()
        info, records = stage_run(
            ctx, role, adapter, stage, warm + updates, micro=micro, accumulation=accumulation,
            lr=lr, use_control=control, probe=False)
        timed = records[warm:]
        profile[name] = {
            'effective_batch': micro * accumulation, 'micro_batch': micro,
            'accumulation': accumulation, 'warmup_updates_excluded': warm,
            'timed_updates': len(timed), 'trainable': info['trainable'],
            'seconds_per_update': stats([r['seconds'] for r in timed]),
            'vae_seconds_per_update_mean': statistics.mean(r['vae_seconds'] for r in timed),
            'data_wait_seconds_per_update_mean': statistics.mean(
                r['data_wait_seconds'] for r in timed),
            'loss_first5_mean': statistics.mean(r['loss'] for r in records[:5]),
            'loss_last10_mean': statistics.mean(r['loss'] for r in records[-10:]),
            'vram': info['vram'], 'first_update_seconds_incl_compile_warmup': records[0]['seconds'],
        }
    cond = conditions(ctx, smoke['profile_images'], reduced=False)
    seeds = [config['inference']['seed'] + i for i in range(smoke['profile_images'])]
    profile['inference'] = {'images': len(seeds), 'steps': config['inference']['steps'],
                            'guidance': config['inference']['guidance_scale'],
                            'control_encode_seconds_for_images': cond['encode_seconds']}
    for batch in (1, smoke['inference_batch']):
        render(ctx, 'source', cond['latents'][:batch], seeds[:batch], batch=batch)  # warm
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats(ctx.device)
        start_t = now()
        render(ctx, 'target', cond['latents'], seeds, batch=batch)
        torch.cuda.synchronize()
        seconds = now() - start_t
        profile['inference'][f'batch{batch}'] = {
            'seconds_total': seconds, 'images_per_second': len(seeds) / seconds,
            'vram': vram(ctx.device)}
    results['profile'] = profile
    lora = profile['lora']['seconds_per_update']['mean']
    controller = profile['controller']['seconds_per_update']['mean']
    best = max(profile['inference'][f'batch{b}']['images_per_second']
               for b in (1, smoke['inference_batch']))
    results['eta_hours_single_gpu'] = {
        'lora_1000_updates': lora * 1000 / 3600, 'controller_2000_updates': controller * 2000 / 3600,
        'generate_1000_images': 1000 / best / 3600, 'generate_2000_images': 2000 / best / 3600}
    write_json(out / 'profile.json', results)
    return results


def run_probe(config, root):
    """Loop-mechanics probe: controller on one fixed batch, fixed-noise loss at checkpoints.

    Not a gate and not evidence that depth is followed."""
    from .model import compute_context, diffusion_loss

    cfg, training = config['smoke']['probe'], config['training']
    ctx = start(config, root)
    dataset = datasets(ctx, 'controller')
    items = [dataset[i * cfg['item_stride']] for i in range(cfg['items'])]
    batch = {'rgb': torch.stack([i['rgb'] for i in items]),
             'control': torch.stack([i['control'] for i in items])}
    device = ctx.device
    latents = encode_image(ctx.vae, batch['rgb'].to(device), control=False,
                           generator=torch.Generator(device=device).manual_seed(0))
    control = encode_image(ctx.vae, batch['control'].to(device), control=True)
    embeds = ctx.prompt['positive'].to(device)
    mask = ctx.prompt['positive_mask'].to(device)
    size = len(items)

    def evaluate():
        values = []
        for s in range(cfg['eval_seeds']):
            generator = torch.Generator(device=device).manual_seed(100 + s)
            with torch.no_grad(), compute_context(device, ctx.dtype):
                values.append(diffusion_loss(
                    ctx.model, ctx.train_scheduler, latents, embeds.expand(size, -1, -1),
                    mask.expand(size, -1), generator=generator, control=control).item())
        return statistics.mean(values)

    named = gm.set_stage(ctx.transformer, ctx.controller, stage='controller', adapter='source')
    curve = {0: evaluate()}

    def checkpoint(update, record):
        if update in cfg['eval_at']:
            curve[update] = evaluate()

    train_updates(
        ctx.model, ctx.vae, ctx.train_scheduler, ctx.prompt, named, iter(lambda: batch, None),
        updates=max(cfg['eval_at']), accumulation=1, lr=cfg['lr'],
        weight_decay=training['weight_decay'], max_grad_norm=training['max_grad_norm'],
        use_control=True, seed=cfg['noise_seed'], device=device, dtype=ctx.dtype,
        warmup=cfg['warmup'], callback=checkpoint)
    results = {'stage': 'probe', 'config': cfg, 'fixed_noise_eval_loss_by_update': curve,
               'environment': environment_info(config)}
    write_json(Path(config['smoke']['output_dir']) / 'probe.json', results)
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage', required=True, choices=['smoke', 'reload', 'profile', 'probe'])
    parser.add_argument('--config', default='configs/generator.yaml')
    args = parser.parse_args()
    config = load_config(args.config)
    runner = {'smoke': run_smoke, 'reload': run_reload, 'profile': run_profile,
              'probe': run_probe}[args.stage]
    result = runner(config, Path.cwd())
    print(json.dumps({k: v for k, v in result.items() if k in (
        'inference', 'isolation', 'source_reload_error', 'target_reload_error', 'profile',
        'eta_hours_single_gpu', 'startup', 'fixed_noise_eval_loss_by_update')}, indent=1, default=float))


if __name__ == '__main__':
    main()
