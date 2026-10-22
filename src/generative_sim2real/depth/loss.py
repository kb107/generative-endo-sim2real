"""Masked canonical L1 + adjacent-gradient L1, adapted from the reference loss.

Select valid values before arithmetic, so NaNs outside the mask never enter
forward reductions or backward derivatives. Marked-valid bad labels fail.
"""
import math
import torch


def validate_targets(target, valid):
    if target.ndim != 4 or target.shape[1] != 1 or target.shape != valid.shape or valid.dtype != torch.bool:
        raise ValueError('Expected depth [B,1,H,W] and matching boolean mask')
    values = target[valid]
    if values.numel() == 0:
        raise ValueError('Depth loss received no valid target pixels')
    if not torch.isfinite(values).all() or bool((values <= 0).any()):
        raise ValueError('Marked-valid target must be finite and positive')
    if not bool(valid.flatten(1).any(1).all()):
        raise ValueError('Depth loss received an all-invalid image')


def mask_counts(valid):
    return (int(valid.sum()), int((valid[..., 1:] & valid[..., :-1]).sum()),
            int((valid[..., 1:, :] & valid[..., :-1, :]).sum()))


def canonical_depth_loss(prediction, target, valid, *, depth_weight=1., gradient_weight=1., denominators=None):
    if prediction.shape != target.shape:
        raise ValueError('Prediction and target shapes differ')
    validate_targets(target, valid)
    if not all(math.isfinite(x) and x >= 0 for x in (depth_weight, gradient_weight)) or depth_weight + gradient_weight == 0:
        raise ValueError('Loss weights must be finite nonnegative with positive total')
    if not torch.isfinite(prediction[valid]).all():
        raise ValueError('Nonfinite valid prediction')
    counts = mask_counts(valid) if denominators is None else denominators
    if len(counts) != 3 or counts[0] < 1:
        raise ValueError('Invalid global loss denominators')
    # An empty selected sum is a finite differentiable zero.
    depth = (prediction[valid] - target[valid]).abs().sum() / counts[0]
    gradients = []
    for axis, denominator in ((-1, counts[1]), (-2, counts[2])):
        left, right = [slice(None)] * 4, [slice(None)] * 4
        left[axis], right[axis] = slice(1, None), slice(None, -1)
        left, right = tuple(left), tuple(right)
        mask = valid[left] & valid[right]
        difference = (prediction[left][mask] - prediction[right][mask]) - (target[left][mask] - target[right][mask])
        gradients.append(difference.abs().sum() / max(1, denominator))
    gradient = sum(gradients)
    total = depth_weight * depth + gradient_weight * gradient
    return total, {'total': total.detach(), 'depth': depth.detach(), 'gradient': gradient.detach()}
