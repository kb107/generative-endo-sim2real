"""Fixed-input selection rules (numpy only)."""
import numpy as np
import pytest

from generative_sim2real.foundation import selection


def test_pick_conditions_fixed_first_then_even_spread():
    picks = selection.pick_conditions(32)
    assert picks[:4] == [13, 3, 12, 7]
    assert len(set(picks)) == 8
    extra = picks[4:]
    assert extra == sorted(extra) and extra[0] == 0 and extra[-1] == 31
    assert not set(extra) & {13, 3, 12, 7}


def test_pick_conditions_rejects_bad_input():
    with pytest.raises(ValueError):
        selection.pick_conditions(10, fixed=(3, 3))
    with pytest.raises(ValueError):
        selection.pick_conditions(10, fixed=(12,))
    with pytest.raises(ValueError):
        selection.pick_conditions(6, fixed=(0, 1, 2, 3, 4), extra=4)


def test_evenly_spaced_endpoints_and_uniqueness():
    ids = [f'id{i}' for i in range(2000)]
    chosen = selection.evenly_spaced(ids, 300)
    assert len(chosen) == len(set(chosen)) == 300
    assert chosen[0] == 'id0' and chosen[-1] == 'id1999'
    with pytest.raises(ValueError):
        selection.evenly_spaced(ids[:5], 6)


def test_nearest_to_mean_prefers_distinct_sequences():
    means = np.array([[.5, .5, .5], [.501, .5, .5], [.2, .2, .2], [.8, .8, .8], [.52, .5, .5]])
    sequences = ['a', 'a', 'b', 'c', 'd']
    # global mean ~ [.504, ...]; frame 1 is nearest, frame 0 (same sequence 'a') is skipped
    chosen = selection.nearest_to_mean(means, sequences, 2)
    assert sequences[chosen[0]] == 'a' and sequences[chosen[1]] != 'a'
    with pytest.raises(ValueError):
        selection.nearest_to_mean(means[:2], ['a', 'a'], 2)
