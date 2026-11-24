"""run_training end to end on CPU with tiny models: checkpoint/callback/resume interplay.

Real data and the real model are replaced by fixtures; everything between (loop, evaluation,
exports, last.pt, status.json, logs, signal handling, resume) is the production code."""
import json
import signal
from types import SimpleNamespace

import pytest
import torch
import yaml  # noqa: F401  (config dicts mirror configs/p4.yaml)

pytest.importorskip('diffusers')
pytest.importorskip('peft')
import generation_tiny as tiny  # noqa: E402
from diffusers import DDPMScheduler  # noqa: E402

from generative_sim2real.generation import gate, train  # noqa: E402


class FakeManifest:
    def __init__(self, split, n=12, seed=0):
        g = torch.Generator().manual_seed(seed)
        self.split, self.sidecar = split, {}
        self.rgb = torch.rand(n, 3, 64, 64, generator=g)
        self.rows = [{'sample_id': f'{split}/{i}', 'sequence_id': 's'} for i in range(n)]

    def __len__(self):
        return len(self.rgb)

    def __getitem__(self, i):
        return {'rgb': self.rgb[i], 'sample_id': self.rows[i]['sample_id']}


@pytest.fixture
def env(tmp_path, monkeypatch):
    handlers = {}
    monkeypatch.setattr(train.signal, 'signal', lambda number, handler: handlers.update({number: handler}))
    monkeypatch.setenv('DATA_ROOT', str(tmp_path))
    (tmp_path / 'train.jsonl').write_text('x\n')
    (tmp_path / 'train.meta.json').write_text('{}\n')
    (tmp_path / 'prompt.safetensors').write_text('cache\n')
    manifests = {'source_train': FakeManifest('source_train', seed=1),
                 'source_validation': FakeManifest('source_validation', 8, seed=2)}
    monkeypatch.setattr(train, 'generation_dataset',
                        lambda path, root, stage: manifests[
                            'source_validation' if 'validation' in str(path) else 'source_train'])
    monkeypatch.setattr(gate, 'environment_info', lambda g: {})
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: False)

    def start(generator, root, device='cpu'):
        model, controller, _ = tiny.tiny_model()
        return SimpleNamespace(
            model=model, controller=controller, transformer=model.transformer, vae=tiny.tiny_vae(),
            train_scheduler=DDPMScheduler.from_config(tiny.tiny_scheduler().config),
            prompt=tiny.tiny_prompt(), device=torch.device('cpu'), dtype=torch.float32,
            base_sha256='fixture-base')

    monkeypatch.setattr(gate, 'start', start)
    generator = {
        'training': {'lora_lr': 1e-2, 'controller_lr': 1e-3, 'weight_decay': 0.01,
                     'max_grad_norm': 1.0, 'effective_batch': 4,
                     'seeds': {'adapter_init': {'source': 42, 'target': 43}}},
        'lora': {'rank': 2, 'alpha': 2}, 'controller': {'num_layers': 2}, 'revision': 'rev',
        'precision': 'float32', 'smoke': {'prompt_cache': str(tmp_path / 'prompt.safetensors')}}
    config = {
        'output_root': str(tmp_path / 'out'), 'num_workers': 0, 'eval_seed': 5,
        'train_probe_frames': 4,
        'manifests': {'source_train': str(tmp_path / 'train.jsonl'),
                      'source_validation': str(tmp_path / 'validation.jsonl')},
        'runs': {'source_lora': {
            'stage': 'lora', 'adapter': 'source', 'role': 'source_lora', 'train': 'source_train',
            'heldout': 'source_validation', 'lr_from': 'lora_lr', 'steps': 6,
            'checkpoint_every': 3, 'warmup': 0, 'micro_batch': 2,
            'seeds': {'loader': 11, 'noise': 12}}}}
    yield SimpleNamespace(config=config, generator=generator, tmp=tmp_path, handlers=handlers,
                          monkeypatch=monkeypatch)
    torch.use_deterministic_algorithms(False)


def run(env, name, **kwargs):
    return train.run_training(env.config, env.generator, 'source_lora', output=env.tmp / name,
                              smoke_steps=6, **kwargs)


def state(env, name):
    return torch.load(env.tmp / name / 'last.pt', map_location='cpu', weights_only=False)


def lines(env, name, file):
    return [json.loads(line) for line in (env.tmp / name / file).read_text().splitlines()]


def same_params(a, b):
    return a['parameters'].keys() == b['parameters'].keys() and all(
        torch.equal(a['parameters'][k], b['parameters'][k]) for k in a['parameters'])


def test_uninterrupted_run_writes_status_logs_and_separate_exports(env):
    final = run(env, 'full')
    assert final['completed_all_steps'] and final['steps_completed'] == 6
    status = json.loads((env.tmp / 'full' / 'status.json').read_text())
    assert status['state'] == 'completed' and status['step'] == 6 and status['eta_seconds'] == 0
    assert status['last_checkpoint'].endswith('step_000006') and status['exit_reason'] == 'completed'
    assert [r['step'] for r in lines(env, 'full', 'train.jsonl')] == [1, 2, 3, 4, 5, 6]
    assert [r['step'] for r in lines(env, 'full', 'eval.jsonl')] == [0, 3, 6]
    for step in ('step_000003', 'step_000006'):
        files = sorted(p.name for p in (env.tmp / 'full' / 'checkpoints' / step).iterdir())
        assert files == ['lora_source.safetensors', 'meta.json']  # no controller tensors in a LoRA export
    assert all(final['hash_changed_as_expected'].values())
    assert final['peak_vram_gib'] is None


def test_stop_then_resume_equals_uninterrupted_bitwise(env):
    run(env, 'full')
    stopped = run(env, 'split', stop_after_step=3)
    assert stopped['completed_all_steps'] is False
    assert json.loads((env.tmp / 'split' / 'status.json').read_text())['state'] == 'stopped_early'
    final = run(env, 'split', resume=True)
    assert final['completed_all_steps']
    assert same_params(state(env, 'full'), state(env, 'split'))
    for log in ('train.jsonl', 'eval.jsonl'):
        assert [r['step'] for r in lines(env, 'split', log)] == [r['step'] for r in lines(env, 'full', log)]
    assert [r['loss'] for r in lines(env, 'split', 'train.jsonl')] == [
        r['loss'] for r in lines(env, 'full', 'train.jsonl')]


def test_signal_between_checkpoints_saves_state_and_resumes_exactly(env):
    run(env, 'full')
    real = train.train_updates

    def interrupting(*args, callback=None, **kwargs):
        def wrapped(step, record):
            if step == 4 and not env.handlers.get('fired'):
                env.handlers['fired'] = True
                env.handlers[signal.SIGTERM](signal.SIGTERM, None)  # as if the scheduler sent SIGTERM
            callback(step, record)
        return real(*args, callback=wrapped, **kwargs)

    env.monkeypatch.setattr(train, 'train_updates', interrupting)
    run(env, 'cut')
    status = json.loads((env.tmp / 'cut' / 'status.json').read_text())
    assert status['state'] == 'interrupted' and status['step'] == 4
    assert state(env, 'cut')['state'] == {'update': 4, 'cursor': 8}
    assert sorted(p.name for p in (env.tmp / 'cut' / 'checkpoints').iterdir()) == ['step_000003']
    env.monkeypatch.setattr(train, 'train_updates', real)
    run(env, 'cut', resume=True)
    assert same_params(state(env, 'full'), state(env, 'cut'))
    assert [r['step'] for r in lines(env, 'cut', 'train.jsonl')] == [1, 2, 3, 4, 5, 6]  # no duplicates


def test_torn_log_line_and_changed_config_on_resume(env):
    run(env, 'x', stop_after_step=3)
    with (env.tmp / 'x' / 'train.jsonl').open('a') as stream:
        stream.write('{"step": 4, "loss"')  # a line torn by a crash after the checkpoint
    run(env, 'x', resume=True)
    assert [r['step'] for r in lines(env, 'x', 'train.jsonl')] == [1, 2, 3, 4, 5, 6]
    run(env, 'y', stop_after_step=3)
    env.config['runs']['source_lora']['seeds'] = {'loader': 11, 'noise': 99}
    with pytest.raises(ValueError, match='Resume mismatch'):
        run(env, 'y', resume=True)
    assert json.loads((env.tmp / 'y' / 'status.json').read_text())['state'] == 'failed'
    with pytest.raises(ValueError, match='requires --resume'):
        run(env, 'y')  # an existing run directory is never silently overwritten


def test_explicit_lr_overrides_the_shared_value(env):
    env.config['runs']['source_lora']['lr'] = 3e-3
    run(env, 'lr')
    assert {r['lr'] for r in lines(env, 'lr', 'train.jsonl')} == {3e-3}
    resolved = json.loads((env.tmp / 'lr' / 'resolved.json').read_text())
    assert resolved['compatibility']['lr'] == 3e-3
