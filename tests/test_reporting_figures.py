"""Reporting figures from a generated one-condition-pair matrix directory (tiny PNGs, no models)."""
import json

import numpy as np
import pytest

pytest.importorskip('matplotlib')
torch = pytest.importorskip('torch')
from PIL import Image  # noqa: E402

from generative_sim2real.reporting import common, figures  # noqa: E402
from generative_sim2real.reporting.matrix import MatrixRun  # noqa: E402

ARMS = ['real', 'control'] + [f'{a}_{c}' for a in ('none', 'source', 'target') for c in ('off', 'correct', 'wrong')]


def make_matrix(root, n=3):
    rng = np.random.default_rng(0)
    for arm in ARMS:
        (root / 'images' / arm).mkdir(parents=True)
        for i in range(n):
            Image.fromarray(rng.integers(0, 255, (16, 16, 3), dtype=np.uint8)).save(root / 'images' / arm / f'c{i}.png')
    depth = torch.full((n, 1, 16, 16), 0.02)
    depth[1] = 0.04
    valid = torch.ones(n, 1, 16, 16, dtype=torch.bool)
    valid[2, :, :8] = False
    depth[2, :, :8] = 5.0  # invalid pixels must not influence the median
    torch.save({'depth_m': depth, 'valid': valid, 'K': torch.eye(3).repeat(n, 1, 1)}, root / 'refs.pt')
    ids = [f'simcol3d/SyntheticColon_I/Frames_S14/{i:04d}' for i in range(n)]
    partner = [n - 1 - i for i in range(n)]
    weights = {k: {'path': f'outputs/runs/{k}/checkpoints/step_001000/x.safetensors'}
               for k in ('controller', 'lora_source', 'lora_target')}
    (root / 'render.json').write_text(json.dumps({
        'ids': ids, 'partner': partner, 'seed': 7, 'control_scale': 1.0, 'weights': weights,
        'inference': {'guidance_scale': 4.5, 'scheduler': 'Sched', 'steps': 20}}))
    gains = {'M1_correct_beats_off': {'gains': [0.1, 0.2, 0.3]}}
    adapters = {a: {'checks': gains} for a in ('none', 'source', 'target')}
    (root / 'matrix.json').write_text(json.dumps({'primary_abs_rel': {'per_adapter': adapters}}))
    return MatrixRun(root)


def test_median_depth_ignores_invalid_pixels(tmp_path):
    run = make_matrix(tmp_path)
    assert run.median_depth_mm() == pytest.approx([20.0, 40.0, 20.0])
    assert run.m1_gains('target') == [0.1, 0.2, 0.3]


def test_generation_and_failure_figures(tmp_path):
    run = make_matrix(tmp_path / 'm')
    encoding = {'constants': {'near_m': 0.007, 'far_m': 0.085}}
    info = figures.generation_figure(run, encoding, tmp_path / 'out' / 'fig1.png')
    assert info['bytes'] <= common.MAX_PNG_BYTES
    assert Image.open(info['path']).size[1] > 0
    cells = [{'adapter': 'target', 'condition': 1, 'M1_gain': 0.2}, {'adapter': 'none', 'condition': 0, 'M1_gain': 0.1}]
    info = figures.failure_figure(run, cells, tmp_path / 'out' / 'fail.png', rule_text='lowest gains.',
                                  title='Fixture.', shown_gain=[0.2, 0.1])
    assert (tmp_path / 'out' / 'fail.png').exists() and info['bytes'] > 0


def test_pipeline_figure_marks_status(tmp_path):
    stages = {name: {'text': name, 'kind': 'trained', 'status': 'run', 'detail': ''} for name in (
        'data_source', 'data_c1', 'base', 'lora_source', 'lora_target', 'controller', 'swap', 'generated', 'da3', 'c2')}
    stages['c2']['status'] = 'not run'
    stages['da3']['status'] = 'partial'
    info = figures.pipeline_figure(stages, tmp_path / 'p.png')
    assert info['bytes'] > 0


def test_png_size_limit_is_enforced(tmp_path):
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(6, 6))
    fig.figimage(np.random.default_rng(0).random((600, 600, 3)))
    with pytest.raises(RuntimeError):
        common.save_figure_png(fig, tmp_path / 'big.png', max_bytes=1000, dpis=(50,))
    plt.close(fig)
