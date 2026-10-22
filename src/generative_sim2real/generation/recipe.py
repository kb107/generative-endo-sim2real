"""Machine-readable generation recipe: everything a bulk generator must reproduce, with file hashes.

Written by the P5 follow-up (outputs/p5-diagnostics/recipe_v{1,2}.json). It records the settings, not a
decision to run: bulk generation needs a separate parent approval. File hashes bind the adapter and
controller weights; `code_provenance` binds the code/configs present when it was written."""
import argparse
import json
import re
from pathlib import Path

import yaml

from . import lowrank_controller as lrc
from .checkpoints import file_sha256
from .provenance import provenance


def run_and_step(path):
    """(run directory, step) of a `.../runs/<run>/checkpoints/step_NNNNNN/<file>` export."""
    path = Path(path)
    match = re.fullmatch(r'step_(\d+)', path.parent.name)
    if not match or path.parent.parent.name != 'checkpoints':
        raise ValueError(f'Not a checkpoint export path: {path}')
    return str(path.parent.parent.parent), int(match.group(1))


def build_recipe(version, *, target_lora, controller, lora_scale, guidance, control_scale,
                 generator_config, evidence, caveats, source_lora=None):
    generator = yaml.safe_load(Path(generator_config).read_text())
    for path in (target_lora, controller):
        if not Path(path).exists():
            raise FileNotFoundError(path)
    run, step = run_and_step(target_lora)
    controller_run, controller_step = run_and_step(controller)
    lowrank = lrc.read_metadata(controller) if Path(controller).name == lrc.FILE_NAMES['lowrank'] else None
    inference = generator['inference']
    return {
        'recipe_version': version, 'status': 'recommended settings only; bulk generation is not approved by this file',
        'base_model': {'id': generator['model_id'], 'revision': generator['revision'],
                       'precision': generator['precision'], 'resolution': generator['resolution']},
        'prompt': generator['prompt'], 'negative_prompt': '',
        'target_adapter': {'run_dir': run, 'step': step, 'file': str(target_lora),
                           'sha256': file_sha256(target_lora), 'lora_scale': lora_scale,
                           'rank': generator['lora']['rank'], 'alpha': generator['lora']['alpha']},
        'matched_comparison_adapter': (
            None if source_lora is None else {
                'file': str(source_lora), 'sha256': file_sha256(source_lora), 'lora_scale': lora_scale,
                'note': 'compare adapters at the same LoRA scale and guidance'}),
        'controller': {'run_dir': controller_run, 'step': controller_step, 'file': str(controller),
                       'sha256': file_sha256(controller), 'control_scale': control_scale,
                       'latent_policy': generator['controller']['latent_policy'],
                       'depth_encoding': 'outputs/manifests/milestone_v1/source_train.meta.json depth_control_encoding',
                       **({} if lowrank is None else lrc.spec_from_metadata(lowrank))},
        'sampler': {'scheduler': inference['scheduler'], 'steps': inference['steps'],
                    'guidance_scale': guidance, 'noise': 'per-sample torch.Generator seeds (generate()); '
                                                         'bf16 autocast, fp32 LoRA/controller masters'},
        'evidence': evidence, 'caveats': caveats,
        'code_provenance': provenance()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--version', required=True)
    parser.add_argument('--target-lora', required=True)
    parser.add_argument('--source-lora')
    parser.add_argument('--controller', required=True)
    parser.add_argument('--lora-scale', type=float, required=True)
    parser.add_argument('--guidance', type=float, required=True)
    parser.add_argument('--control-scale', type=float, default=1.0)
    parser.add_argument('--generator-config', default='configs/generator.yaml')
    parser.add_argument('--evidence', required=True, help='JSON file with the evidence summary')
    parser.add_argument('--caveats', required=True, help='JSON list file')
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    recipe = build_recipe(
        args.version, target_lora=args.target_lora, controller=args.controller,
        lora_scale=args.lora_scale, guidance=args.guidance, control_scale=args.control_scale,
        generator_config=args.generator_config, source_lora=args.source_lora,
        evidence=json.loads(Path(args.evidence).read_text()),
        caveats=json.loads(Path(args.caveats).read_text()))
    Path(args.out).write_text(json.dumps(recipe, indent=2, allow_nan=False) + '\n')


if __name__ == '__main__':
    main()
