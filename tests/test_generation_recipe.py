"""Recipe file builder: hashes bind the exported weights and the settings are the declared ones."""
import hashlib
import json
from pathlib import Path

import pytest

from generative_sim2real.generation import recipe

ROOT = Path(__file__).parents[1]


def exports(tmp_path):
    lora = tmp_path / 'runs' / 'target_lora' / 'checkpoints' / 'step_001000' / 'lora_target.safetensors'
    ctrl = tmp_path / 'runs' / 'controller_pilot' / 'checkpoints' / 'step_002000' / 'controller.safetensors'
    for path, data in ((lora, b'lora'), (ctrl, b'controller')):
        path.parent.mkdir(parents=True)
        path.write_bytes(data)
    return lora, ctrl


def test_recipe_binds_files_and_settings(tmp_path):
    lora, ctrl = exports(tmp_path)
    built = recipe.build_recipe(
        'vtest', target_lora=lora, controller=ctrl, lora_scale=0.75, guidance=3.0, control_scale=1.0,
        generator_config=ROOT / 'configs' / 'generator.yaml', evidence={'a': 1}, caveats=['c'])
    assert built['target_adapter']['sha256'] == hashlib.sha256(b'lora').hexdigest()
    assert built['target_adapter']['step'] == 1000 and built['controller']['step'] == 2000
    assert built['target_adapter']['lora_scale'] == 0.75 and built['sampler']['guidance_scale'] == 3.0
    assert built['sampler']['steps'] == 20 and built['prompt'] == 'an endoscopic view of the colon lumen'
    assert built['base_model']['revision'] == '50f702106901db6d0f8b67eb88e814c56ded2692'
    assert 'not approved' in built['status']
    assert 'configs/p4.yaml' in built['code_provenance']
    json.dumps(built, allow_nan=False)  # machine-readable


def test_recipe_refuses_missing_files_and_non_export_paths(tmp_path):
    lora, ctrl = exports(tmp_path)
    with pytest.raises(FileNotFoundError):
        recipe.build_recipe('v', target_lora=tmp_path / 'nope', controller=ctrl, lora_scale=1.0,
                            guidance=4.5, control_scale=1.0, generator_config=ROOT / 'configs' / 'generator.yaml',
                            evidence={}, caveats=[])
    with pytest.raises(ValueError, match='checkpoint export'):
        recipe.run_and_step(tmp_path / 'lora.safetensors')
