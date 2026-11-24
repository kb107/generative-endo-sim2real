"""Deterministic selection rules for the fixed comparison inputs (numpy only; unit-tested)."""
import numpy as np

FIXED_FIRST = (13, 3, 12, 7)  # conditions fixed in advance (failure / informative cells)


def pick_conditions(n_conditions, fixed=FIXED_FIRST, extra=4):
    """`fixed` plus `extra` more: evenly spaced (np.linspace, rounded) positions over the sorted
    remaining condition indices, so the picks span the whole matrix (both sequences)."""
    if len(set(fixed)) != len(fixed) or any(not 0 <= c < n_conditions for c in fixed):
        raise ValueError('fixed conditions must be unique and in range')
    remaining = [c for c in range(n_conditions) if c not in fixed]
    if extra > len(remaining):
        raise ValueError('not enough remaining conditions')
    positions = np.round(np.linspace(0, len(remaining) - 1, extra)).astype(int)
    return list(fixed) + [remaining[p] for p in positions]


def evenly_spaced(items, count):
    """`count` items at np.linspace(0, len-1, count) rounded positions (unique, order kept)."""
    if not 0 < count <= len(items):
        raise ValueError('count must be in 1..len(items)')
    positions = np.round(np.linspace(0, len(items) - 1, count)).astype(int)
    if len(set(positions.tolist())) != count:
        raise ValueError('evenly spaced positions collide')
    return [items[p] for p in positions]


def nearest_to_mean(frame_means, sequence_ids, count=2):
    """Indices of the `count` frames whose mean RGB is nearest the mean of all frame means.

    The nearest frame is taken first; each further one is the nearest frame from a sequence not
    yet used (adjacent frames of one video are near-duplicates, which would waste a reference)."""
    means = np.asarray(frame_means, dtype=np.float64)
    distance = np.linalg.norm(means - means.mean(axis=0), axis=1)
    chosen, used = [], set()
    for index in np.argsort(distance, kind='stable'):
        if sequence_ids[index] in used:
            continue
        chosen.append(int(index))
        used.add(sequence_ids[index])
        if len(chosen) == count:
            return chosen
    raise ValueError('fewer distinct sequences than requested references')
