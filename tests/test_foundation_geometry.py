"""Registration metrics on synthetic images with known shifts/zooms; depth scores on toy tensors."""
import cv2
import numpy as np
import pytest

from generative_sim2real.foundation import geometry


def textured(size=256, seed=0):
    rng = np.random.default_rng(seed)
    noise = cv2.GaussianBlur(rng.random((size, size, 3)).astype(np.float32), (0, 0), 3)
    noise = (noise - noise.min()) / (noise.max() - noise.min())
    return (noise * 255).astype(np.uint8)


def shift(image, dx, dy):
    matrix = np.float32([[1, 0, dx], [0, 1, dy]])
    return cv2.warpAffine(image, matrix, image.shape[1::-1], flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)


def zoom(image, factor):
    h, w = image.shape[:2]
    matrix = cv2.getRotationMatrix2D(((w - 1) / 2, (h - 1) / 2), 0, factor)
    return cv2.warpAffine(image, matrix, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)


def test_identity_is_zero():
    image = textured()
    r = geometry.registration(image, image)
    assert r['flow_median_px'] < 0.05 and r['ecc_shift_px'] < 0.05 and abs(r['ecc_scale'] - 1) < 1e-3
    assert r['phase_shift_px'] < 0.05


def test_translation_recovered_with_sign():
    image = textured()
    r = geometry.registration(image, shift(image, 4, -3))  # content moves right 4, up 3
    assert r['phase_dx'] == pytest.approx(4, abs=0.5) and r['phase_dy'] == pytest.approx(-3, abs=0.5)
    assert r['ecc_dx'] == pytest.approx(4, abs=0.5) and r['ecc_dy'] == pytest.approx(-3, abs=0.5)
    assert r['flow_median_px'] == pytest.approx(5, abs=1.0)


def test_zoom_recovered():
    image = textured()
    r = geometry.registration(image, zoom(image, 1.06))
    assert r['ecc_converged'] and r['ecc_scale'] == pytest.approx(1.06, abs=0.01)


def test_appearance_change_is_not_misread_as_motion():
    image = textured()
    recoloured = np.clip(image[..., ::-1].astype(np.float32) * 0.7 + 30, 0, 255).astype(np.uint8)
    r = geometry.registration(image, recoloured)
    assert r['ecc_shift_px'] < 0.5 and abs(r['ecc_scale'] - 1) < 0.01


def test_size_mismatch_rejected():
    with pytest.raises(ValueError):
        geometry.registration(textured(256), textured(128))


def test_depth_scores_scale_invariant_and_rank():
    torch = pytest.importorskip('torch')
    ref = torch.linspace(0.02, 0.1, 64 * 64).reshape(1, 64, 64)
    valid = torch.ones_like(ref, dtype=torch.bool)
    exact = geometry.depth_scores(ref * 3.7, ref, valid)  # global scale must not matter
    assert exact['abs_rel'] < 1e-6 and exact['spearman'] == pytest.approx(1, abs=1e-6)
    flipped = geometry.depth_scores(ref.flatten().flip(0).reshape(ref.shape), ref, valid)
    assert flipped['spearman'] < 0.5 and flipped['abs_rel'] > 0.1


def test_summarize_ignores_nan():
    rows = [{'arm': 'a', 'x': 1.0}, {'arm': 'a', 'x': float('nan')}, {'arm': 'a', 'x': 3.0}]
    s = geometry.summarize(rows, ['x'])
    assert s['a']['x'] == {'mean': 2.0, 'median': 2.0, 'n': 2}
