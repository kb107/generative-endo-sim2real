"""P5 matrix fixtures: no-adapter inference, off invariance, and the declared within-adapter rules."""
from pathlib import Path

import pytest
import torch
import yaml

pytest.importorskip('diffusers')
pytest.importorskip('peft')
import generation_tiny as tiny  # noqa: E402

from generative_sim2real.generation import model as gm  # noqa: E402
from generative_sim2real.generation import p5_matrix as p5  # noqa: E402

CONFIG = yaml.safe_load((Path(__file__).parents[1] / 'configs' / 'p5.yaml').read_text())
EMBED = tiny.tiny_prompt()


def forward(transformer, latents):
    kwargs = dict(encoder_hidden_states=EMBED['positive'].expand(2, -1, -1),
                  encoder_attention_mask=EMBED['positive_mask'].expand(2, -1),
                  timestep=torch.tensor([5, 500]), return_dict=False,
                  added_cond_kwargs={'resolution': None, 'aspect_ratio': None})
    with torch.no_grad():
        return transformer(latents, **kwargs)[0]


def perturbed_model():
    model, controller, _ = tiny.tiny_model()
    with torch.no_grad():
        for name, p in model.transformer.named_parameters():
            if 'lora_B' in name:
                p.normal_(0, 0.5)  # non-trivial adapters (B starts at zero)
        for name, p in controller.named_parameters():
            if name.split('.')[-2] in ('before_proj', 'after_proj'):
                p.normal_(0, 0.5)
    return model, controller


def test_none_mode_is_the_clean_base_and_adapters_come_back():
    clean = tiny.tiny_transformer()  # same seed => same base weights the model was built from
    latents, _ = tiny.batch()
    base = forward(clean, latents)
    model, controller = perturbed_model()
    p5.set_adapter_mode(model.transformer, controller, 'source')
    source = forward(model.transformer, latents)
    p5.set_adapter_mode(model.transformer, controller, 'target')
    target = forward(model.transformer, latents)
    p5.set_adapter_mode(model.transformer, controller, None)
    none = forward(model.transformer, latents)
    assert torch.allclose(none, base, atol=1e-6)
    assert not torch.allclose(source, base, atol=1e-3) and not torch.allclose(target, source, atol=1e-3)
    p5.set_adapter_mode(model.transformer, controller, 'source')  # re-enabled after a none render
    assert torch.equal(forward(model.transformer, latents), source)
    assert not any(p.requires_grad for p in model.parameters())  # inference never trains anything


def test_off_does_not_depend_on_the_nominal_depth_for_any_adapter():
    model, controller = perturbed_model()
    vae, scheduler = tiny.tiny_vae(), tiny.tiny_scheduler()
    g = torch.Generator().manual_seed(4)
    latents = torch.randn(2, 4, 8, 8, generator=g)
    other = torch.randn(2, 4, 8, 8, generator=g)
    kwargs = dict(model=model, vae=vae, scheduler=scheduler, prompt=EMBED, seeds=[3, 4], steps=2,
                  guidance=4.5, size=64)
    for adapter in (None, 'source', 'target'):
        p5.set_adapter_mode(model.transformer, controller, adapter)
        off = gm.generate(**kwargs, control=None)
        assert torch.equal(off, gm.generate(**kwargs, control=latents, control_scale=0.0))
        assert torch.equal(off, gm.generate(**kwargs, control=other, control_scale=0.0))
        on = gm.generate(**kwargs, control=latents)
        assert not torch.equal(on, off)  # the same fixture proves the controller does act when on
        assert not torch.equal(on, gm.generate(**kwargs, control=other))  # and depends on the depth


def rows_for(adapter_gains):
    """Synthetic proxy rows: per adapter (retained, ...) -> errors that realise a given behaviour."""
    out = []
    for i in range(8):
        out.append({'arm': 'real', 'cond': i, 'abs_rel_own': 0.1, 'abs_rel_partner': 0.5})
        for adapter, kept in adapter_gains.items():
            good = kept or i < 4  # 'half' adapters follow depth in only 4/8 conditions
            out += [
                {'arm': f'{adapter}_off', 'cond': i, 'abs_rel_own': 0.5, 'abs_rel_partner': 0.5},
                {'arm': f'{adapter}_correct', 'cond': i, 'abs_rel_own': 0.2 if good else 0.5,
                 'abs_rel_partner': 0.5},
                {'arm': f'{adapter}_wrong', 'cond': i, 'abs_rel_own': 0.5,
                 'abs_rel_partner': 0.2 if good else 0.5}]
    return out


def test_declared_within_adapter_rules_and_swap_verdict():
    rules = CONFIG['analysis']['rules']
    partner = list(range(7, -1, -1))
    result = p5.matrix_checks(rows_for({'none': False, 'source': True, 'target': True}), partner, rules,
                              ['none', 'source', 'target'])
    assert result['source']['control_retained'] and result['target']['control_retained']
    assert not result['none']['control_retained']  # 4/8 conditions only
    assert p5.swap_verdict(result, True).startswith('RETAINED')
    result = p5.matrix_checks(rows_for({'none': True, 'source': True, 'target': False}), partner, rules,
                              ['none', 'source', 'target'])
    verdict = p5.swap_verdict(result, True)
    assert verdict.startswith('NOT RETAINED') and 'M1_correct_beats_off' in verdict
    assert p5.swap_verdict(result, False).startswith('UNINTERPRETABLE')
    comparison = p5.retention_comparison(result)
    assert comparison['M1_correct_beats_off']['mean_gain_target'] < comparison['M1_correct_beats_off']['mean_gain_source']
    cells = p5.failure_cells(result, ['none', 'source', 'target'], 6)
    assert len(cells) == 6 and cells == sorted(cells)
    assert [(a, i) for _, a, i in cells[:4]] == [('target', i) for i in range(4, 8)]  # lowest M1 gains first
    with pytest.raises(ValueError, match='Missing proxy rows'):
        p5.matrix_checks([r for r in rows_for({'source': True}) if r['arm'] != 'source_off'], partner,
                         rules, ['source'])


def test_p5_config_declares_the_reduced_matrix():
    assert CONFIG['controls'] == ['off', 'correct', 'wrong']  # bare YAML off would parse as False
    assert CONFIG['adapters'] == ['none', 'source', 'target']
    assert len(CONFIG['adapters']) * len(CONFIG['controls']) * 8 == 72
    assert CONFIG['inputs']['controller'].endswith('controller_pilot/checkpoints/step_002000/controller.safetensors')
    assert CONFIG['inputs']['source_lora'].endswith('step_001000/lora_source.safetensors')
    assert CONFIG['analysis']['rules']['min_conditions'] == 7
    assert 'no bulk generation' in CONFIG['scope_boundary']


def test_lora_scale_is_an_inference_only_multiplier():
    clean = tiny.tiny_transformer()
    latents, _ = tiny.batch()
    base = forward(clean, latents)
    model, controller = perturbed_model()
    p5.set_adapter_mode(model.transformer, controller, 'target')
    full = forward(model.transformer, latents)
    before = gm.state_hash(model.transformer, only_lora='target')
    assert p5.set_lora_scale(model.transformer, 'target', 0.0) == 4 * 2 * 4
    assert torch.allclose(forward(model.transformer, latents), base, atol=1e-6)  # scale 0 = no LoRA
    p5.set_lora_scale(model.transformer, 'target', 0.5)
    half = forward(model.transformer, latents)
    assert not torch.allclose(half, full, atol=1e-4) and not torch.allclose(half, base, atol=1e-4)
    p5.set_lora_scale(model.transformer, 'target', 1.0)  # restores the trained scale exactly
    assert torch.equal(forward(model.transformer, latents), full)
    assert gm.state_hash(model.transformer, only_lora='target') == before  # weights never change
    with pytest.raises(ValueError, match='No LoRA layers'):
        p5.set_lora_scale(model.transformer, 'missing', 0.5)


SWEEP = yaml.safe_load((Path(__file__).parents[1] / 'configs' / 'p5_sweep.yaml').read_text())


def cell(scale, guidance, diversity, w1=0.086, w1v=0.107, nn_l=1.0, nn_p=1.0):
    return {'lora_scale': scale, 'guidance': guidance, 'diversity': diversity, 'w1_c1_train': w1,
            'w1_c1_validation': w1v, 'nn_ratio_latent': nn_l, 'nn_ratio_pixel': nn_p}


def test_sweep_selection_rule_is_mechanical():
    from generative_sim2real.generation import p5_sweep

    rule = SWEEP['selection_rule']
    real = 0.208  # needs diversity >= 0.156
    cells = [cell(1.0, 4.5, 0.113), cell(1.0, 3.0, 0.17), cell(0.75, 4.5, 0.172),
             cell(0.5, 3.0, 0.20, w1=0.15), cell(0.5, 4.5, 0.21, nn_l=0.5), cell(0.75, 3.0, 0.14)]
    chosen = p5_sweep.select_setting(cells, rule, real)
    assert (chosen['lora_scale'], chosen['guidance']) == (1.0, 3.0)  # 0.17 vs 0.172: tied, closer to v1
    by = {(c['lora_scale'], c['guidance']): c for c in cells}
    assert not by[(1.0, 4.5)]['qualifies']  # v1 itself is below the diversity floor
    assert not by[(0.5, 3.0)]['constraints']['w1_train_not_worse_than_v1_by_more_than_margin']
    assert not by[(0.5, 4.5)]['constraints']['nn_ratio_latent_at_least_min']  # memorisation-like
    assert not by[(0.75, 3.0)]['constraints']['diversity_at_least_fraction_of_real_c1_train']
    assert p5_sweep.select_setting([cell(1.0, 4.5, 0.113), cell(0.5, 3.0, 0.12)], rule, real) is None
    # a clearly more diverse setting beats the tie-break
    assert p5_sweep.select_setting(cells + [cell(0.75, 3.5, 0.19)], rule, real)['guidance'] == 3.5


def test_sweep_config_declares_the_grid_and_the_rule():
    assert SWEEP['settings'] == {'lora_scale': [0.5, 0.75, 1.0], 'guidance': [3.0, 4.5]}
    assert SWEEP['selection_rule']['v1_cell'] == [1.0, 4.5] and SWEEP['images'] == 64
    assert SWEEP['step_B_retrain']['steps'] == 2000 and SWEEP['step_B_retrain']['lr'] == 5e-5


def test_review_config_keeps_the_declared_rule_and_combines_exports():
    from generative_sim2real.generation import p5_sweep

    review = yaml.safe_load((Path(__file__).parents[1] / 'configs' / 'p5_review_v2.yaml').read_text())
    for key in ('w1_margin', 'nn_ratio_min', 'diversity_min_fraction_of_real_c1_train', 'v1_cell'):
        assert review['selection_rule'][key] == SWEEP['selection_rule'][key]
    assert review['settings'] == {'lora_scale': [1.0], 'guidance': [3.0, 4.5]}
    # constraints are measured against an explicit reference (the v1 LoRA), not the new LoRA's own cell
    rule, reference = SWEEP['selection_rule'], cell(1.0, 4.5, 0.113)
    new = [cell(1.0, 4.5, 0.17, w1=0.0862), cell(1.0, 3.0, 0.18, w1=0.12)]  # second breaks the W1 margin
    assert p5_sweep.select_setting(new, rule, 0.208, reference)['guidance'] == 4.5
    first = {'step': 1000, 'cells': [dict(cell(1.0, 4.5, 0.172), qualifies=True)]}
    second = {'step': 2000, 'cells': [dict(cell(1.0, 4.5, 0.170), qualifies=True), dict(cell(1.0, 3.0, 0.169), qualifies=True)]}
    assert p5_sweep.combine_reviews([first, second])['step'] == 2000  # tie within 0.005 -> later step
    assert p5_sweep.combine_reviews([{'step': 1000, 'cells': [dict(cell(1.0, 4.5, 0.12), qualifies=False)]}]) is None


def test_v2_matrix_config_differs_from_v1_only_in_adapter_scale_and_guidance():
    v2 = yaml.safe_load((Path(__file__).parents[1] / 'configs' / 'p5_v2.yaml').read_text())
    v1 = yaml.safe_load((Path(__file__).parents[1] / 'configs' / 'p5.yaml').read_text())
    assert v2['inputs']['target_lora'].endswith('target_lora_v2/checkpoints/step_002000/lora_target.safetensors')
    assert v2['lora_scale'] == 1.0 and v2['guidance'] == 4.5 and v2['output_dir'].endswith('matrix_reduced_v2')
    for key in ('lora_scale', 'guidance'):
        v2.pop(key)
    v2['output_dir'], v2['inputs']['target_lora'] = v1['output_dir'], v1['inputs']['target_lora']
    assert v2 == v1  # every rule, threshold, condition, seed and test is identical
