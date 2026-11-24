import json

import numpy as np
import pytest
import torch

from generative_sim2real.data.preprocessing import (
    InverseDepthEncoding, depth_audit, fit_inverse_depth, uniform_indices,
)


def sample(values, valid=None):
    depth = torch.tensor(values, dtype=torch.float32).reshape(1, 1, -1)
    return {'sample_id': 'simcol3d/s/train/0000',
            'targets': {'depth_m': depth, 'depth_valid': (
                torch.ones_like(depth, dtype=torch.bool) if valid is None
                else torch.tensor(valid, dtype=torch.bool).reshape(depth.shape))}, 'meta': {}}


def test_polarity_invalid_clipping_and_labels_unchanged():
    encoding = InverseDepthEncoding(.01, .1)
    depth = torch.tensor([.001, .01, .1, 1, 0, float('nan'), float('inf'), -.1]).reshape(1, 1, -1)
    before = depth.clone()
    control = encoding.encode(depth, torch.ones_like(depth, dtype=torch.bool))
    assert control.dtype == torch.float32
    assert control.shape == (3, 1, 8)
    assert control[0, 0].tolist() == pytest.approx([1, 1, 0, 0, 0, 0, 0, 0])
    torch.testing.assert_close(before, depth, equal_nan=True)
    assert torch.equal(control[0], control[2])


def test_serialized_constants_roundtrip():
    encoder = InverseDepthEncoding(.005, .08)
    restored = InverseDepthEncoding.from_dict(json.loads(json.dumps(encoder.to_dict())))
    assert encoder == restored
    for near, far in ((0, 1), (.1, .1), (.2, .1), (np.nan, 1)):
        with pytest.raises(ValueError):
            InverseDepthEncoding(near, far)


def test_fit_train_only_deterministic_pixel_sampling_and_quantiles():
    item = sample([.01, .02, .03, .04, .05])
    row = {'sample_id': item['sample_id'], 'split': 'source_train', 'dataset': 'simcol3d'}
    fitted, metadata = fit_inverse_depth([(row, item)], pixels_per_frame=3,
                                         near_quantile=0, far_quantile=1)
    assert fitted.near_m == pytest.approx(.01)
    assert fitted.far_m == pytest.approx(.05)
    assert metadata['pixel_counts'] == [3]
    for split in ('source_validation', 'target_test', 'target_appearance_train'):
        with pytest.raises(ValueError, match='source_train'):
            fit_inverse_depth([({**row, 'split': split}, item)])
    with pytest.raises(ValueError, match='No valid'):
        fit_inverse_depth([(row, sample([0, 0]))])
    with pytest.raises(ValueError, match='finite'):
        fit_inverse_depth([(row, sample([.01, .01]))])


def test_uniform_cap_and_audit():
    assert uniform_indices(10, 4) == [0, 3, 6, 9]
    assert uniform_indices(3, 9) == [0, 1, 2]
    assert uniform_indices(0, 2) == []
    for n in range(1, 25):
        for cap in range(1, 26):
            chosen = uniform_indices(n, cap)
            assert chosen == sorted(set(chosen))
            assert len(chosen) == min(n, cap)
    result = depth_audit([sample([0, .001, .02, .2])], InverseDepthEncoding(.01, .1))
    assert result['invalid_fraction'] == .25
    assert result['below_near_fraction_of_valid'] == pytest.approx(1 / 3)
    assert result['above_far_fraction_of_valid'] == pytest.approx(1 / 3)
