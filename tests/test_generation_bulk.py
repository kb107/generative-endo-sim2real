"""P5b bulk generation: shards, idempotence, kill/resume, corruption, retries, and the proof that
the DA3 G/M loader accepts the manifest. Generated fixtures only; CPU; no model downloads."""
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

import da3_fakes as fakes
from generative_sim2real.data import rgb_modes as rm
from generative_sim2real.generation import bulk

N = 7


@pytest.fixture(autouse=True)
def small_shards(monkeypatch):
    monkeypatch.setattr(bulk, 'SHARD_SIZE', 3)  # 7 IDs -> shards of 3, 3, 1


@pytest.fixture
def source(tmp_path):
    split = fakes.FakeSplit(tmp_path, 'source_train', 'source_generation512', N, 512)
    split.sidecar['depth_control_encoding'] = {'constants': {'near_m': .01, 'far_m': .1, 'invalid_fill': 0.}}
    return split


def write_recipe(tmp_path, *, seed=11, steps=2, scale=1.0, purpose='smoke', content=b'w', batch=2, guidance=4.5,
                 prompt='a prompt', control_scale=1.0, precision='float32', extra=None, record_sha=None):
    for run, name in (('src', 'lora_source'), ('tgt', 'lora_target'), ('ctl', 'controller')):
        directory = tmp_path / 'runs' / run / 'checkpoints' / 'step_000007'
        directory.mkdir(parents=True, exist_ok=True)
        (directory / f'{name}.safetensors').write_bytes(content + name.encode())
    source_file = 'runs/src/checkpoints/step_000007/lora_source.safetensors'
    (tmp_path / 'runs' / 'ctl' / 'resolved.json').write_text(json.dumps({'compatibility': {
        'files_sha256': {source_file: record_sha or bulk.file_sha256(tmp_path / source_file)},
        'revision': 'rev1', 'precision': precision}}))
    recipe = {'schema_version': 1, 'purpose': purpose, 'seed': seed, 'generator_config': 'configs/generator.yaml',
              'base': {'model_id': 'fake/model', 'revision': 'rev1'}, 'prompt': prompt, 'precision': precision,
              'size': 512, 'scheduler': 'DPMSolverMultistepScheduler', 'steps': steps, 'guidance': guidance,
              'batch': batch, 'source_lora': {'run_dir': 'runs/src', 'step': 7},
              'target_lora': {'run_dir': 'runs/tgt', 'step': 7, 'scale': scale},
              'controller': {'run_dir': 'runs/ctl', 'step': 7, 'scale': control_scale}, **(extra or {})}
    path = tmp_path / 'recipe.json'
    path.write_text(json.dumps(recipe))
    return bulk.load_recipe(path, root=tmp_path)


class FakeRender:
    """Deterministic image per seed; can inject crashes and technical failures."""
    def __init__(self, crash_after_calls=None, fail=None):
        self.calls, self.rendered_seeds = 0, []
        self.crash_after_calls, self.fail = crash_after_calls, fail or (lambda seeds, call: None)

    def __call__(self, samples, seeds):
        self.calls += 1
        if self.crash_after_calls is not None and self.calls > self.crash_after_calls:
            raise KeyboardInterrupt('simulated kill')
        self.fail(seeds, self.calls)
        self.rendered_seeds += list(seeds)
        return torch.stack([torch.rand(3, 512, 512, generator=torch.Generator().manual_seed(s % 2**31))
                            for s in seeds])


def ids_of(source):
    return [r['sample_id'] for r in source.rows]


def setup(tmp_path, source, **kw):
    resolved = write_recipe(tmp_path, **kw)
    layout = bulk.Layout(tmp_path / 'out', resolved['recipe_hash'])
    return resolved, layout, ids_of(source)


def quiet(*_):
    pass


def test_full_run_manifest_and_rgb_modes_accept_it(tmp_path, source):
    resolved, layout, ids = setup(tmp_path, source)
    render = FakeRender()
    summary = bulk.generate(resolved, source, ids, layout, render, log=quiet)
    assert (summary['generated'], summary['skipped'], summary['failed']) == (N, 0, 0)
    assert sorted(p.name for p in (layout.directory / 'shards').iterdir()) == ['shard_0000', 'shard_0001', 'shard_0002']
    assert all(json.loads(layout.status(s).read_text())['complete'] for s in range(3))
    info = bulk.build_manifest(resolved, source, ids, layout)
    assert info['records'] == N and info['rgb_modes_validation'] == {'G_len': N, 'M_len': N, 'G_all_items_loaded': N}
    records = [json.loads(line) for line in layout.manifest.read_text().splitlines()]
    assert [r['sample_id'] for r in records] == ids  # frozen order
    seeds = [r['seed'] for r in records]
    assert len(set(seeds)) == N and seeds == [bulk.per_id_seed(11, s) for s in ids]
    for key in rm.GENERATED_PROVENANCE:  # identical across records except the seed
        assert len({json.dumps(r[key]) for r in records}) == (N if key == 'seed' else 1)
    assert records[0]['source_transform_id'] == 'source_generation512' and records[0]['size_hw'] == [512, 512]
    # the consumer, independent of this module's own checks, also accepts it
    dataset = rm.RGBModeDataset(source, ids, mode='G', generated_manifest=layout.manifest, outputs_root=layout.out_root)
    assert dataset[0]['rgb'].shape == (3, 504, 504)


def test_rerun_is_idempotent_and_skips_valid_images(tmp_path, source):
    resolved, layout, ids = setup(tmp_path, source)
    bulk.generate(resolved, source, ids, layout, FakeRender(), log=quiet)
    before = {p: p.read_bytes() for p in layout.directory.rglob('*.png')}
    render = FakeRender()
    summary = bulk.generate(resolved, source, ids, layout, render, log=quiet)
    assert render.calls == 0 and summary['skipped'] == N and summary['generated'] == 0
    assert before == {p: p.read_bytes() for p in layout.directory.rglob('*.png')}
    assert bulk.verify(resolved, source, ids, layout)['invalid_count'] == 0


def test_kill_at_any_batch_boundary_resumes_to_the_same_result(tmp_path, source):
    resolved, layout, ids = setup(tmp_path, source)
    bulk.generate(resolved, source, ids, layout, FakeRender(), log=quiet)
    reference = {p.name: p.read_bytes() for p in layout.directory.rglob('*.png')}
    for crash in range(0, 4):
        sub = tmp_path / f'run{crash}'
        sub.mkdir()
        resolved2 = write_recipe(sub)
        layout2 = bulk.Layout(sub / 'out', resolved2['recipe_hash'])
        with pytest.raises(KeyboardInterrupt):
            bulk.generate(resolved2, source, ids, layout2, FakeRender(crash_after_calls=crash), log=quiet)
        (layout2.directory / 'shards' / 'shard_0000' / '.tmp-1-00001.png').parent.mkdir(parents=True, exist_ok=True)
        (layout2.directory / 'shards' / 'shard_0000' / '.tmp-1-00001.png').write_bytes(b'torn')  # leftover temp
        render = FakeRender()
        summary = bulk.generate(resolved2, source, ids, layout2, render, log=quiet)
        assert summary['skipped'] + summary['generated'] == N and summary['failed'] == 0
        assert len(render.rendered_seeds) == summary['generated'] <= N
        assert not list(layout2.directory.rglob('.tmp-*'))
        assert {p.name: p.read_bytes() for p in layout2.directory.rglob('*.png')} == reference
        assert bulk.build_manifest(resolved2, source, ids, layout2)['records'] == N


def test_corrupt_missing_and_mismatched_images_are_detected_and_regenerated(tmp_path, source):
    resolved, layout, ids = setup(tmp_path, source)
    bulk.generate(resolved, source, ids, layout, FakeRender(), log=quiet)
    reference = {p.name: p.read_bytes() for p in layout.directory.rglob('*.png')}
    corrupt, missing, swapped = layout.image(0), layout.image(1), layout.image(4)
    corrupt.write_bytes(corrupt.read_bytes()[:-9] + b'garbage!!')          # checksum mismatch
    missing.unlink()                                                       # file gone
    swapped.write_bytes(layout.image(5).read_bytes())                      # a valid but wrong image
    check = bulk.verify(resolved, source, ids, layout)
    assert check['invalid_count'] == 3
    with pytest.raises(bulk.BulkError, match='3 of 7 IDs'):                # manifest refuses, not silently drops
        bulk.build_manifest(resolved, source, ids, layout)
    status_path = layout.status(2)                                         # stale fingerprint in a status file
    status = json.loads(status_path.read_text())
    status['images'][ids[6]]['fingerprint'] = '0' * 64
    status_path.write_text(json.dumps(status))
    assert bulk.verify(resolved, source, ids, layout)['invalid_count'] == 4
    render = FakeRender()
    summary = bulk.generate(resolved, source, ids, layout, render, log=quiet)
    assert summary['regenerated'] == 4 and summary['generated'] == 4 and summary['skipped'] == 3
    assert {p.name: p.read_bytes() for p in layout.directory.rglob('*.png')} == reference
    assert bulk.build_manifest(resolved, source, ids, layout)['records'] == N


def test_unreadable_status_file_means_regeneration_not_acceptance(tmp_path, source):
    resolved, layout, ids = setup(tmp_path, source)
    bulk.generate(resolved, source, ids, layout, FakeRender(), log=quiet)
    layout.status(1).write_text('{"torn')
    render = FakeRender()
    summary = bulk.generate(resolved, source, ids, layout, render, log=quiet)
    assert summary['generated'] == 3 and summary['skipped'] == 4


def test_changed_source_geometry_invalidates_existing_images(tmp_path, source):
    resolved, layout, ids = setup(tmp_path, source)
    bulk.generate(resolved, source, ids, layout, FakeRender(), log=quiet)
    source.depth_bias = .001  # same IDs, different 512-stage depth bytes
    assert bulk.verify(resolved, source, ids, layout)['invalid_count'] == N
    with pytest.raises(bulk.BulkError):
        bulk.build_manifest(resolved, source, ids, layout)


def test_technical_failures_are_retried_then_recorded_never_dropped(tmp_path, source):
    resolved, layout, ids = setup(tmp_path, source)
    flaky = FakeRender(fail=lambda seeds, call: (_ for _ in ()).throw(RuntimeError('CUDA out of memory')) if call == 1 else None)
    summary = bulk.generate(resolved, source, ids, layout, flaky, log=quiet)
    assert summary['generated'] == N and summary['failed'] == 0
    entry = json.loads(layout.status(0).read_text())['images'][ids[0]]
    assert entry['attempts'] == 2

    sub = tmp_path / 'persistent'
    sub.mkdir()
    resolved2 = write_recipe(sub)
    layout2 = bulk.Layout(sub / 'out', resolved2['recipe_hash'])
    bad_seed = bulk.per_id_seed(11, ids[2])

    def fail(seeds, call):
        if bad_seed in seeds:
            raise RuntimeError('CUDA error: out of memory')
    summary = bulk.generate(resolved2, source, ids, layout2, FakeRender(fail=fail), log=quiet)
    assert summary['failed'] == 1 and summary['generated'] == N - 1
    status = json.loads(layout2.status(0).read_text())
    assert ids[2] in status['failures'] and not status['complete']
    with pytest.raises(bulk.BulkError, match='recorded technical failure'):
        bulk.build_manifest(resolved2, source, ids, layout2)
    summary = bulk.generate(resolved2, source, ids, layout2, FakeRender(), log=quiet)  # next run retries it
    assert summary['generated'] == 1 and summary['skipped'] == N - 1
    assert bulk.build_manifest(resolved2, source, ids, layout2)['records'] == N


def test_non_technical_errors_propagate_and_systematic_failure_aborts(tmp_path, source):
    resolved, layout, ids = setup(tmp_path, source)

    def bug(seeds, call):
        raise ValueError('a real bug')
    with pytest.raises(ValueError, match='real bug'):
        bulk.generate(resolved, source, ids, layout, FakeRender(fail=bug), log=quiet)

    def always(seeds, call):
        raise RuntimeError('CUDA out of memory')
    with pytest.raises(bulk.BulkError, match='Three consecutive'):
        bulk.generate(resolved, source, ids, layout, FakeRender(fail=always), max_attempts=1, log=quiet)


def test_nonfinite_output_is_a_technical_failure(tmp_path, source):
    resolved, layout, ids = setup(tmp_path, source)

    def nan_render(samples, seeds):
        return torch.full((len(seeds), 3, 512, 512), float('nan'))
    with pytest.raises(bulk.BulkError, match='Three consecutive'):
        bulk.generate(resolved, source, ids, layout, nan_render, max_attempts=1, log=quiet)
    assert not list(layout.directory.rglob('*.png'))


def test_recipe_hash_tracks_every_render_defining_value_and_rejects_unsupported_ones(tmp_path):
    a = write_recipe(tmp_path)
    assert write_recipe(tmp_path)['recipe_hash'] == a['recipe_hash']
    variants = [dict(seed=12), dict(steps=3), dict(content=b'other'), dict(guidance=5.0), dict(prompt='other'),
                dict(control_scale=0.5), dict(precision='bfloat16'), dict(batch=1), dict(batch=4)]
    hashes = [write_recipe(tmp_path, **v)['recipe_hash'] for v in variants]
    assert a['recipe_hash'] not in hashes and len(set(hashes)) == len(hashes)
    assert write_recipe(tmp_path, batch=1, purpose='production')['recipe_hash'] == write_recipe(
        tmp_path, batch=1, purpose='smoke')['recipe_hash']  # purpose is not render-defining
    with pytest.raises(bulk.BulkError, match='scale'):
        write_recipe(tmp_path, scale=0.5)
    with pytest.raises(bulk.BulkError, match='batch: 1'):
        write_recipe(tmp_path, purpose='production', batch=4)
    with pytest.raises(bulk.BulkError, match='Unknown recipe keys'):
        write_recipe(tmp_path, extra={'guidence': 1})
    with pytest.raises(bulk.BulkError, match='source LoRA recorded'):
        write_recipe(tmp_path, record_sha='0' * 64)
    recipe = json.loads((tmp_path / 'recipe.json').read_text())
    del recipe['batch']
    (tmp_path / 'recipe.json').write_text(json.dumps(recipe))
    with pytest.raises(bulk.BulkError, match="missing 'batch'"):
        bulk.load_recipe(tmp_path / 'recipe.json', root=tmp_path)
    recipe.update(batch=1, controller={'run_dir': 'runs/ctl', 'step': 7, 'scale': 1.0, 'extra': 1})
    (tmp_path / 'recipe.json').write_text(json.dumps(recipe))
    with pytest.raises(bulk.BulkError, match='exactly the keys'):
        bulk.load_recipe(tmp_path / 'recipe.json', root=tmp_path)
    bulk.check_destination('smoke', tmp_path / 'smoke' / 'x')
    with pytest.raises(bulk.BulkError, match='Smoke'):
        bulk.check_destination('smoke', tmp_path / 'outputs')
    with pytest.raises(bulk.BulkError, match='Production'):
        bulk.check_destination('production', tmp_path / 'smoke')
    assert set(a['provenance']) | {'seed'} == set(rm.GENERATED_PROVENANCE)


def test_batch_is_in_fingerprints_and_manifest_records(tmp_path, source):
    ids = ids_of(source)[:2]
    prints = []
    for batch in (1, 2):
        sub = tmp_path / f'b{batch}'
        sub.mkdir()
        resolved = write_recipe(sub, batch=batch)
        layout = bulk.Layout(sub / 'out', resolved['recipe_hash'])
        bulk.generate(resolved, source, ids, layout, FakeRender(), log=quiet)
        bulk.build_manifest(resolved, source, ids, layout)
        record = json.loads(layout.manifest.read_text().splitlines()[0])
        assert record['batch_size'] == batch
        prints.append(record['generation_fingerprint'])
    assert prints[0] != prints[1]


def test_select_ids_refuses_limit_for_production_recipes(tmp_path, source):
    path = fakes.write_ids(tmp_path, source, ids_of(source), smoke=False, frozen=True)
    assert bulk.select_ids(path, source, None, 'production') == ids_of(source)
    assert bulk.select_ids(path, source, 3, 'smoke') == ids_of(source)[:3]
    with pytest.raises(bulk.BulkError, match='smoke recipe'):
        bulk.select_ids(path, source, 3, 'production')


def test_resume_refuses_changed_environment_unless_overridden_and_history_is_append_only(tmp_path, source):
    resolved, layout, ids = setup(tmp_path, source)
    env = bulk.environment_fingerprint()
    assert set(env) == {'code_sha256', 'code_files', 'versions', 'gpu'} and set(env['versions']) == set(bulk.PACKAGES)
    with pytest.raises(KeyboardInterrupt):
        bulk.generate(resolved, source, ids, layout, FakeRender(crash_after_calls=2), log=quiet, environment=env)
    recipe_bytes = (layout.directory / 'recipe.json').read_bytes()
    bulk.generate(resolved, source, ids, layout, FakeRender(), log=quiet, environment=env)  # identical environment resumes; recipe.json untouched
    assert (layout.directory / 'recipe.json').read_bytes() == recipe_bytes
    changed = {**env, 'code_sha256': '1' * 64}
    sub = tmp_path / 'again'
    sub.mkdir()
    resolved2 = write_recipe(sub)
    layout2 = bulk.Layout(sub / 'out', resolved2['recipe_hash'])
    with pytest.raises(KeyboardInterrupt):
        bulk.generate(resolved2, source, ids, layout2, FakeRender(crash_after_calls=2), log=quiet, environment=env)
    with pytest.raises(bulk.BulkError, match='different code/library/GPU'):
        bulk.generate(resolved2, source, ids, layout2, FakeRender(), log=quiet, environment=changed)
    with pytest.raises(bulk.BulkError, match='different code/library/GPU'):
        bulk.generate(resolved2, source, ids, layout2, FakeRender(), log=quiet,
                      environment={**env, 'gpu': 'Other GPU'})
    bulk.generate(resolved2, source, ids, layout2, FakeRender(), log=quiet, environment=changed,
                  allow_environment_change=True)
    history = [json.loads(line) for line in (layout2.directory / 'run_history.jsonl').read_text().splitlines()]
    assert len(history) == 2  # the crash and the overridden completion (refusals happen before a run starts)
    assert history[0]['outcome'].startswith('KeyboardInterrupt') and history[-1]['outcome'] == 'completed'
    assert history[-1]['environment_override'] is True and history[0]['environment_override'] is False
    assert history[-1]['code_sha256'] == '1' * 64 and history[-1]['counts']['failed'] == 0
    assert {'host', 'start_utc', 'end_utc', 'batch', 'failures', 'versions', 'images_per_second_render'} <= set(history[-1])
    # an environment change before anything was generated is simply adopted
    sub = tmp_path / 'fresh'
    sub.mkdir()
    resolved3 = write_recipe(sub)
    layout3 = bulk.Layout(sub / 'out', resolved3['recipe_hash'])
    bulk.generate(resolved3, source, ids[:1], layout3, FakeRender(), log=quiet, environment=env)
    (layout3.directory / 'shards' / 'shard_0000' / 'status.json').unlink()
    bulk.generate(resolved3, source, ids[:1], layout3, FakeRender(), log=quiet, environment=changed)
    assert json.loads((layout3.directory / 'recipe.json').read_text())['environment']['code_sha256'] == '1' * 64


def test_existing_recipe_json_must_match_hashed_fields(tmp_path, source):
    resolved, layout, ids = setup(tmp_path, source)
    bulk.generate(resolved, source, ids[:1], layout, FakeRender(), log=quiet)
    path = layout.directory / 'recipe.json'
    record = json.loads(path.read_text())
    record['defining']['guidance'] = 9.0
    path.write_text(json.dumps(record))
    with pytest.raises(bulk.BulkError, match='differs from this recipe'):
        bulk.generate(resolved, source, ids[:1], layout, FakeRender(), log=quiet)


def test_concurrent_runs_are_locked_out(tmp_path, source):
    resolved, layout, ids = setup(tmp_path, source)
    with bulk.locked(layout):
        with pytest.raises(bulk.BulkError, match='lock'):
            bulk.generate(resolved, source, ids, layout, FakeRender(), log=quiet)
        with pytest.raises(bulk.BulkError, match='lock'):
            bulk.build_manifest(resolved, source, ids, layout)
    bulk.generate(resolved, source, ids, layout, FakeRender(), log=quiet)  # released afterwards


def test_only_oom_and_nonfinite_are_technical(tmp_path, source):
    assert bulk.is_technical(RuntimeError('CUDA out of memory. Tried to allocate'))
    assert bulk.is_technical(bulk.TechnicalFailure('nan'))
    for error in (RuntimeError('Expected all tensors to be on the same device, cuda:0 and cpu'),
                  RuntimeError('CUDA error: device-side assert triggered'), ValueError('out of memory')):
        assert not bulk.is_technical(error)
    resolved, layout, ids = setup(tmp_path, source)

    def mismatch(seeds, call):
        raise RuntimeError('Expected all tensors to be on the same device, cuda:0 and cpu')
    with pytest.raises(RuntimeError, match='same device'):
        bulk.generate(resolved, source, ids, layout, FakeRender(fail=mismatch), log=quiet)


def test_truncated_png_with_matching_checksum_and_wrong_status_path_are_invalid(tmp_path, source):
    resolved, layout, ids = setup(tmp_path, source)
    bulk.generate(resolved, source, ids, layout, FakeRender(), log=quiet)
    path = layout.image(0)
    path.write_bytes(path.read_bytes()[:-2000])                     # truncated data, valid header
    status = json.loads(layout.status(0).read_text())
    status['images'][ids[0]]['sha256'] = bulk.file_sha256(path)     # checksum agrees with the broken file
    other = str(layout.image(2).relative_to(layout.out_root))       # and another valid file for ID 1
    status['images'][ids[1]]['file'] = other
    status['images'][ids[1]]['sha256'] = bulk.file_sha256(layout.image(2))
    layout.status(0).write_text(json.dumps(status))
    assert bulk.verify(resolved, source, ids, layout)['invalid_count'] == 2
    summary = bulk.generate(resolved, source, ids, layout, FakeRender(), log=quiet)
    assert summary['regenerated'] == 2 and summary['generated'] == 2


def test_manifest_is_rejected_by_rgb_modes_when_provenance_drifts(tmp_path, source):
    """Guards stay in the consumer: a hand-edited manifest does not slip through build or load."""
    resolved, layout, ids = setup(tmp_path, source)
    bulk.generate(resolved, source, ids, layout, FakeRender(), log=quiet)
    bulk.build_manifest(resolved, source, ids, layout)
    records = [json.loads(line) for line in layout.manifest.read_text().splitlines()]
    records[3]['prompt'] = 'different'
    bad = tmp_path / 'bad.jsonl'
    bad.write_text(''.join(json.dumps(r) + '\n' for r in records))
    with pytest.raises(ValueError, match='differs across records'):
        bulk.validate_with_rgb_modes(source, ids, bad, layout.out_root)


def test_real_renderer_path_on_tiny_models(tmp_path, source):
    """make_renderer -> p5_matrix.render_images -> model.generate with tiny random weights."""
    pytest.importorskip('diffusers')
    pytest.importorskip('peft')
    import generation_tiny as tiny

    model, controller, _ = tiny.tiny_model()
    ctx = SimpleNamespace(transformer=model.transformer, controller=controller, model=model,
                          vae=tiny.tiny_vae(), scheduler=tiny.tiny_scheduler(), prompt=tiny.tiny_prompt(),
                          config={'inference': {'steps': 2, 'guidance_scale': 4.5}, 'resolution': 512},
                          dtype=torch.float32)
    resolved = write_recipe(tmp_path)
    render = bulk.make_renderer(ctx, source, resolved)
    samples = [source[0], source[1]]
    out = render(samples, [5, 6])
    assert out.shape == (2, 3, 512, 512) and torch.isfinite(out).all()
    # each image depends only on its own seed and depth, not on its batch-mates
    assert torch.allclose(out[1:], render(samples[1:], [6]), atol=1e-4)
    assert not torch.allclose(out[0], out[1])
    # full pipeline with the real renderer on the CPU fixture
    layout = bulk.Layout(tmp_path / 'out', resolved['recipe_hash'])
    ids = ids_of(source)[:4]
    summary = bulk.generate(resolved, source, ids, layout, render, log=quiet)
    assert summary['generated'] == 4
    assert bulk.build_manifest(resolved, source, ids, layout)['records'] == 4
