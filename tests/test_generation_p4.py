"""P4 fixtures: resumable training, checkpoints, data order, config rules, evaluation losses.

Tiny CPU models only (no downloads); the real-model smoke is scripts/train_p4.sh with
--smoke-steps."""
import copy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import yaml

pytest.importorskip('diffusers')
pytest.importorskip('peft')
import generation_tiny as tiny  # noqa: E402
from diffusers import DDPMScheduler  # noqa: E402

from generative_sim2real.generation import checkpoints as ck  # noqa: E402
from generative_sim2real.generation import gate, inputs, train, training  # noqa: E402
from generative_sim2real.generation import model as gm  # noqa: E402

ROOT = Path(__file__).parents[1]
CONFIG = ROOT / 'configs' / 'p4.yaml'
COMPAT = {'run': 'fixture', 'seed': 1}


class Frames(torch.utils.data.Dataset):
    def __init__(self, n=12):
        g = torch.Generator().manual_seed(11)
        self.rgb = torch.rand(n, 3, 64, 64, generator=g)
        self.control = torch.rand(n, 3, 64, 64, generator=g)

    def __len__(self):
        return len(self.rgb)

    def __getitem__(self, i):
        return {'rgb': self.rgb[i], 'control': self.control[i], 'sample_id': str(i)}


def run_updates(path, total, *, save_at=None, resume=False, cursor_override=None, stage='lora'):
    """Fresh tiny model/optimizer every call, optionally restoring `path`; returns params+records."""
    model, controller, _ = tiny.tiny_model()
    named = gm.set_stage(model.transformer, controller, stage=stage, adapter='source')
    vae, scheduler = tiny.tiny_vae(), DDPMScheduler.from_config(tiny.tiny_scheduler().config)
    optimizer, schedule = training.make_optimizer(named, lr=1e-2, weight_decay=0.01, warmup=3)
    noise = training.noise_generator('cpu', 5)
    state = {'update': 0, 'cursor': 0}
    if resume:
        state = ck.load_training_state(path, named, optimizer, schedule, noise, COMPAT,
                                       accumulation=2)
    start = state['cursor'] if cursor_override is None else cursor_override
    loader = inputs.cursor_loader(Frames(), batch_size=2, seed=9, workers=0, start=start)

    def callback(step, record):
        if step == save_at:
            ck.save_training_state(path, named, optimizer, schedule, noise,
                                   {'update': step, 'cursor': step * 2}, COMPAT)

    records, _ = training.train_updates(
        model, vae, scheduler, tiny.tiny_prompt(), named, iter(loader),
        updates=total - state['update'], accumulation=2, lr=1e-2, weight_decay=0.01,
        max_grad_norm=1.0, use_control=stage == 'controller', device='cpu', dtype=torch.float32,
        callback=callback, optimizer=optimizer, lr_schedule=schedule, generator=noise,
        first_update=state['update'])
    return {n: p.detach().clone() for n, p in named}, records, model, controller


def test_interrupted_resume_equals_uninterrupted_bitwise(tmp_path):
    full, full_records, _, _ = run_updates(tmp_path / 'x.pt', 4)
    run_updates(tmp_path / 'last.pt', 2, save_at=2)
    resumed, records, _, _ = run_updates(tmp_path / 'last.pt', 4, resume=True)
    assert all(torch.equal(full[n], resumed[n]) for n in full)
    assert [r['loss'] for r in records] == [r['loss'] for r in full_records[2:]]
    assert [r['lr'] for r in records] == [r['lr'] for r in full_records[2:]]  # warmup resumed


def test_resume_test_is_sensitive_to_cursor_optimizer_and_compat(tmp_path):
    full, _, _, _ = run_updates(tmp_path / 'x.pt', 4)
    run_updates(tmp_path / 'last.pt', 2, save_at=2)
    wrong_cursor, _, _, _ = run_updates(tmp_path / 'last.pt', 4, resume=True, cursor_override=0)
    assert any(not torch.equal(full[n], wrong_cursor[n]) for n in full)
    model, controller, _ = tiny.tiny_model()
    named = gm.set_stage(model.transformer, controller, stage='lora', adapter='source')
    optimizer, schedule = training.make_optimizer(named, lr=1e-2, weight_decay=0.01, warmup=3)
    noise = training.noise_generator('cpu', 5)
    with pytest.raises(ValueError, match='Resume mismatch'):
        ck.load_training_state(tmp_path / 'last.pt', named, optimizer, schedule, noise,
                               {**COMPAT, 'seed': 2}, accumulation=2)
    with pytest.raises(ValueError, match='cursor'):
        ck.load_training_state(tmp_path / 'last.pt', named, optimizer, schedule, noise, COMPAT,
                               accumulation=3)
    assert not list(tmp_path.glob('*.tmp'))  # atomic writes leave nothing behind


def test_controller_stage_resume_is_bitwise_and_trains_only_the_controller(tmp_path):
    full, _, model, controller = run_updates(tmp_path / 'x.pt', 4, stage='controller')
    assert all(n.startswith('controller.') for n in full)
    run_updates(tmp_path / 'last.pt', 2, save_at=2, stage='controller')
    resumed, _, _, _ = run_updates(tmp_path / 'last.pt', 4, resume=True, stage='controller')
    assert all(torch.equal(full[n], resumed[n]) for n in full)
    initial = tiny.tiny_model()[1].state_dict()
    assert any(not torch.equal(full[f'controller.{k}'], v) for k, v in initial.items()
               if f'controller.{k}' in full)  # the controller actually moved


def test_non_finite_loss_stops_immediately(monkeypatch):
    model, controller, _ = tiny.tiny_model()
    named = gm.set_stage(model.transformer, controller, stage='lora', adapter='source')
    monkeypatch.setattr(training, 'diffusion_loss',
                        lambda *a, **k: sum(p.sum() for _, p in named) * float('nan'))
    monkeypatch.setattr(training, 'encode_image', lambda vae, image, control, generator=None: image)
    with pytest.raises(FloatingPointError, match='update 1'):
        training.train_updates(
            model, None, None, tiny.tiny_prompt(), named, iter([{'rgb': torch.zeros(2, 4, 8, 8)}] * 3),
            updates=3, accumulation=1, lr=1e-3, weight_decay=0.0, max_grad_norm=1.0,
            use_control=False, seed=0, device='cpu', dtype=torch.float32)


def test_external_optimizer_must_cover_trainable_parameters():
    model, controller, _ = tiny.tiny_model()
    named = gm.set_stage(model.transformer, controller, stage='lora', adapter='source')
    optimizer, schedule = training.make_optimizer(named[:-1], lr=1e-3, weight_decay=0.0)
    with pytest.raises(ValueError, match='optimizer must cover'):
        training.train_updates(
            model, None, None, tiny.tiny_prompt(), named, iter([]), updates=1, accumulation=1,
            lr=1e-3, weight_decay=0.0, max_grad_norm=1.0, use_control=False, device='cpu',
            dtype=torch.float32, optimizer=optimizer, lr_schedule=schedule, seed=0)


def test_batch_order_is_a_pure_function_of_the_cursor():
    order = inputs.BatchOrder(10, 3, seed=4)
    first = [order.indices(k) for k in range(9)]
    again = inputs.BatchOrder(10, 3, seed=4)
    assert first == [again.indices(k) for k in range(9)]
    assert [again.indices(k) for k in (7, 3, 0)] == [first[7], first[3], first[0]]  # random access
    for epoch in range(3):  # each epoch is a permutation prefix without repeats
        flat = sum(first[epoch * 3:epoch * 3 + 3], [])
        assert len(set(flat)) == 9 and set(flat) <= set(range(10))
    assert first[:3] != first[3:6]  # reshuffled every epoch
    assert inputs.BatchOrder(10, 3, seed=5).indices(0) != first[0]
    with pytest.raises(ValueError):
        inputs.BatchOrder(2, 3, seed=0)


def test_cursor_loader_resumes_at_the_cursor():
    data = Frames(12)
    uninterrupted = iter(inputs.cursor_loader(data, batch_size=2, seed=3, workers=0))
    batches = [next(uninterrupted)['sample_id'] for _ in range(9)]
    resumed = iter(inputs.cursor_loader(data, batch_size=2, seed=3, workers=0, start=5))
    assert [next(resumed)['sample_id'] for _ in range(4)] == batches[5:9]


def test_frame_tensors_subset_and_label_free_rgb():
    class Manifest:
        split = 'target_appearance_train'
        rows = [{}] * 5
        sidecar = {}

        def __len__(self):
            return 5

        def __getitem__(self, i):
            return {'rgb': torch.full((3, 2, 2), i / 10), 'sample_id': f's{i}'}

    data = inputs.FrameTensors(Manifest(), with_control=False, indices=[4, 1])
    assert len(data) == 2 and data[0]['sample_id'] == 's4' and 'control' not in data[1]
    with pytest.raises(ValueError, match='nonempty subset'):
        inputs.FrameTensors(Manifest(), with_control=False, indices=[5])
    with pytest.raises(ValueError, match='labeled source roles'):
        inputs.FrameTensors(Manifest(), with_control=True)


def test_exports_are_separate_files_with_metadata(tmp_path):
    model, controller, _ = tiny.tiny_model()
    lora, meta = ck.export_checkpoint(tmp_path, 250, kind='lora', model=model, controller=controller,
                                      adapter='source', rank=2, alpha=2, revision='rev')
    ctrl, ctrl_meta = ck.export_checkpoint(tmp_path, 500, kind='controller', model=model,
                                           controller=controller, adapter='source', rank=2,
                                           alpha=2, revision='rev')
    from safetensors import safe_open

    with safe_open(str(lora / 'lora_source.safetensors'), framework='pt') as handle:
        assert all('lora_' in k for k in handle.keys()) and handle.metadata()['adapter'] == 'source'
    with safe_open(str(ctrl / 'controller.safetensors'), framework='pt') as handle:
        assert not any('lora_' in k for k in handle.keys())
        assert handle.metadata()['base_revision'] == 'rev'
    assert meta['sha256'] == ck.file_sha256(lora / 'lora_source.safetensors')
    assert json.loads((ctrl / 'meta.json').read_text())['step'] == 500
    assert lora.name == 'step_000250' and not list(tmp_path.glob('.step_*'))
    # Loading a controller file into the wrong kind is rejected (existing P3 guard).
    with pytest.raises(ValueError):
        gm.load_controller(controller, lora / 'lora_source.safetensors', revision='rev')
    # Re-exporting a step (resume after a crash between export and last.pt) replaces it cleanly.
    again, _ = ck.export_checkpoint(tmp_path, 250, kind='lora', model=model, controller=controller,
                                    adapter='source', rank=2, alpha=2, revision='rev')
    assert again == lora


def test_truncate_jsonl_and_eta(tmp_path):
    path = tmp_path / 'train.jsonl'
    path.write_text('\n'.join(json.dumps({'step': s}) for s in (1, 2, 3, 4)) + '\n{"step": 5')
    ck.truncate_jsonl(path, 3)
    assert [json.loads(line)['step'] for line in path.read_text().splitlines()] == [1, 2, 3]
    assert ck.eta_seconds([], 10) is None and ck.eta_seconds([2.0, 4.0], 10) == 30.0


def stub_context(model, controller):
    scheduler = DDPMScheduler.from_config(tiny.tiny_scheduler().config)
    return SimpleNamespace(model=model, train_scheduler=scheduler, prompt=tiny.tiny_prompt(),
                           device=torch.device('cpu'), dtype=torch.float32)


def test_eval_loss_is_fixed_noise_and_detects_controller_use():
    model, controller, _ = tiny.tiny_model()
    ctx = stub_context(model, controller)
    g = torch.Generator().manual_seed(1)
    evalset = {'ids': list('abcd'), 'rgb': torch.randn(4, 4, 8, 8, generator=g),
               'control': torch.randn(4, 4, 8, 8, generator=g), 'wrong': [2, 3, 0, 1]}
    first = {m: train.eval_loss(ctx, evalset, mode=m, seed=7, micro=2) for m in ('correct', 'off', 'wrong')}
    assert first == {m: train.eval_loss(ctx, evalset, mode=m, seed=7, micro=2) for m in first}
    assert first['correct'] == first['off'] == first['wrong']  # zero-initialised projections
    with torch.no_grad():
        for name, p in controller.named_parameters():
            if name.split('.')[-2] in ('before_proj', 'after_proj'):
                p.normal_(0, 0.5)
    moved = {m: train.eval_loss(ctx, evalset, mode=m, seed=7, micro=2) for m in first}
    assert len({moved['correct'], moved['off'], moved['wrong']}) == 3
    # Evaluation never consumes training state: no gradients, no stage change.
    assert all(p.grad is None for p in model.parameters())


def test_select_overfit_pairs_rule():
    medians = {'S2': [(m, f'b{m}') for m in (9, 3, 7, 1, 5, 8, 2)],
               'S1': [(m, f'a{m}') for m in (4, 6, 2, 8, 1, 3, 5)]}
    ids, partner = train.select_overfit_pairs(medians, 4)
    assert ids == ['a1', 'a3', 'a5', 'a8', 'b1', 'b3', 'b7', 'b9']  # ranks 0, 2, 4, 6 within scenes
    assert partner == list(range(7, -1, -1)) and all(p != i for i, p in enumerate(partner))
    assert {ids[i][0] != ids[p][0] for i, p in enumerate(partner)} == {True}  # always cross-scene
    with pytest.raises(ValueError, match='two scenes'):
        train.select_overfit_pairs({'only': medians['S1']}, 4)


def test_p4_config_validates_and_rejects_protocol_violations():
    config = yaml.safe_load(CONFIG.read_text())
    generator = gate.load_config(ROOT / 'configs' / 'generator.yaml')
    train.validate_p4(config, generator)
    assert config['gate']['arms'] == ['correct', 'off', 'wrong']  # bare YAML off parses as False
    runs = config['runs']
    assert runs['source_lora']['steps'] == 1000 and runs['controller_pilot']['steps'] == 2000
    assert runs['controller_pilot']['checkpoint_every'] == 500
    cases = [
        ('runs.source_lora.train', 'source_validation'),          # validation can never train
        ('runs.source_lora.train', 'target_appearance_train'),    # wrong domain for the role
        ('runs.target_lora.adapter', 'source'),
        ('runs.controller_pilot.source_lora_run', 'target_lora'),
        ('runs.controller_overfit.seeds.noise', 1303),            # seed shared with the pilot
        ('runs.source_lora.seeds.loader', 42),                    # equals an adapter init seed
        ('runs.source_lora.checkpoint_every', 300),
        ('runs.source_lora.micro_batch', 3),
        ('runs.source_lora.lr_from', 'controller_lr'),
    ]
    for path, value in cases:
        bad = copy.deepcopy(config)
        node = bad
        *parents, leaf = path.split('.')
        for key in parents:
            node = node[key]
        node[leaf] = value
        with pytest.raises(ValueError):
            train.validate_p4(bad, generator)


def test_compatibility_code_files_exist():
    code = ROOT / 'src' / 'generative_sim2real' / 'generation'
    assert all((code / name).exists() for name in train.FINGERPRINT_CODE)


def test_v2_retrain_run_is_declared_and_lr_override_is_validated():
    config = yaml.safe_load(CONFIG.read_text())
    generator = gate.load_config(ROOT / 'configs' / 'generator.yaml')
    run = config['runs']['target_lora_v2']
    assert (run['steps'], run['lr'], run['checkpoint_every'], run['seeds']) == (
        2000, 5e-5, 500, {'loader': 212, 'noise': 1212})
    assert run['train'] == 'target_appearance_train' and run['adapter'] == 'target'
    train.validate_p4(config, generator)
    for bad in (0.0, 5, 2.0):
        broken = copy.deepcopy(config)
        broken['runs']['target_lora_v2']['lr'] = bad
        with pytest.raises(ValueError):
            train.validate_p4(broken, generator)
