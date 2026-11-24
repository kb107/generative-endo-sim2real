import copy
from pathlib import Path

import pytest
import torch
import yaml

pytest.importorskip('diffusers')
pytest.importorskip('peft')
import generation_tiny as tiny  # noqa: E402

from generative_sim2real.generation import gate, training  # noqa: E402
from generative_sim2real.generation import model as gm  # noqa: E402

CONFIG = Path(__file__).parents[1] / 'configs' / 'generator.yaml'


@pytest.fixture
def stage(monkeypatch):
    """Tiny real model in the LoRA stage with a deterministic per-sample loss."""
    model, controller, _ = tiny.tiny_model()
    named = gm.set_stage(model.transformer, controller, stage='lora', adapter='source')
    probe = next(p for n, p in named if 'lora_A' in n)

    def loss(model, scheduler, latents, embeds, mask, *, generator, control=None):
        return ((probe.sum() * latents.mean(dim=(1, 2, 3))) ** 2).mean()

    monkeypatch.setattr(training, 'diffusion_loss', loss)
    monkeypatch.setattr(training, 'encode_image', lambda vae, image, control, generator=None: image)
    return model, controller, named


def batches(n, size):
    g = torch.Generator().manual_seed(0)
    return [{'rgb': torch.randn(size, 4, 8, 8, generator=g)} for _ in range(n)]


def run(model, named, data, **kwargs):
    base = dict(updates=1, accumulation=1, lr=1e-2, weight_decay=0.0, max_grad_norm=1e9,
                use_control=False, seed=0, device='cpu', dtype=torch.float32)
    base.update(kwargs)
    return training.train_updates(model, None, None, tiny.tiny_prompt(), named, iter(data),
                                  **base)


def test_accumulation_equals_one_large_batch(stage):
    model, _, named = stage
    start = gm.snapshot(named)
    big = batches(1, 4)[0]
    run(model, named, [big])
    one = gm.snapshot(named)
    with torch.no_grad():
        for n, p in named:
            p.copy_(start[n])
    halves = [{'rgb': big['rgb'][:2]}, {'rgb': big['rgb'][2:]}]
    run(model, named, halves, accumulation=2)
    for n, p in named:
        assert torch.allclose(p.detach(), one[n], atol=1e-6), n
    assert any((one[n] - start[n]).abs().max() > 0 for n in one)


def test_linear_warmup_schedule(stage):
    model, _, named = stage
    records, _ = run(model, named, batches(6, 2), updates=6, warmup=4, lr=1e-3)
    assert [r['lr'] for r in records] == pytest.approx([2.5e-4, 5e-4, 7.5e-4, 1e-3, 1e-3, 1e-3])
    records, _ = run(model, named, batches(2, 2), updates=2, warmup=0, lr=1e-3)
    assert [r['lr'] for r in records] == [1e-3, 1e-3]


def test_trainable_parameter_guard(stage):
    model, controller, named = stage
    with pytest.raises(ValueError, match='exactly the trainable'):
        run(model, [], batches(1, 2))
    frozen = [(n, p) for n, p in model.transformer.named_parameters() if 'lora_B' not in n][:1]
    with pytest.raises(ValueError, match='exactly the trainable'):
        run(model, frozen, batches(1, 2))
    with pytest.raises(ValueError, match='differs from the parameters'):
        run(model, named[:-1], batches(1, 2))  # a trainable tensor would be silently skipped


def test_callback_must_restore_stage(stage):
    model, controller, named = stage

    def render_like(update, record):  # a careless eval that leaves the model in inference mode
        gm.set_stage(model.transformer, controller, stage='inference', adapter='source')

    with pytest.raises(RuntimeError, match='trainable set'):
        run(model, named, batches(2, 2), updates=2, callback=render_like)

    def switch_adapter(update, record):
        gm.set_stage(model.transformer, controller, stage='lora', adapter='target')

    gm.set_stage(model.transformer, controller, stage='lora', adapter='source')
    with pytest.raises(RuntimeError, match='trainable set'):
        run(model, named, batches(2, 2), updates=2, callback=switch_adapter)

    def harmless(update, record):
        record['seen'] = update

    gm.set_stage(model.transformer, controller, stage='lora', adapter='source')
    records, _ = run(model, named, batches(2, 2), updates=2, callback=harmless)
    assert [r['seen'] for r in records] == [1, 2]


def test_real_config_validates_and_bad_values_fail():
    config = yaml.safe_load(CONFIG.read_text())
    gate.validate_config(config)
    for path, value in [('backend', 'sd15'), ('lora.target_modules', ['to_q']),
                        ('controller.initialization', 'random'),
                        ('controller.latent_policy', 'posterior_sample'),
                        ('training.rgb_latent_policy', 'posterior_mode'),
                        ('inference.scheduler', 'DDIMScheduler'), ('resolution', 1024),
                        ('training.micro_batch', 3),
                        ('training.seeds.noise.controller', 101)]:
        bad = copy.deepcopy(config)
        node = bad
        *parents, leaf = path.split('.')
        for key in parents:
            node = node[key]
        node[leaf] = value
        with pytest.raises(ValueError, match='Unsupported|divisible'):
            gate.validate_config(bad)


def test_effective_batch_must_divide():
    assert gate.accumulation_for({'effective_batch': 8}, 4) == 2
    with pytest.raises(ValueError, match='divisible'):
        gate.accumulation_for({'effective_batch': 8}, 3)
