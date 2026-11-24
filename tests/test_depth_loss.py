import pytest
import torch

from generative_sim2real.depth.loss import canonical_depth_loss, mask_counts


def tensors(pred, target, valid):
    shape = (1, 1, len(pred), len(pred[0]))
    return (torch.tensor(pred, requires_grad=True).reshape(shape), torch.tensor(target).reshape(shape),
            torch.tensor(valid, dtype=torch.bool).reshape(shape))


def test_hand_computed_loss_value():
    pred, target, valid = tensors([[2., 2., 5.], [1., 1., 1.]], [[1., 2., 3.], [1., 1., 1.]], [[1, 1, 1], [1, 1, 1]])
    loss, terms = canonical_depth_loss(pred, target, valid)
    # L1 = (1+0+2)/6; horizontal: row0 |0-1|+|3-1|=3, row1 0 -> 3/4; vertical: |(1-2)-(1-1)|... computed by hand below
    depth = 3 / 6
    horizontal = (abs((2 - 2) - (1 - 2)) + abs((5 - 2) - (3 - 2))) / 4          # row0: 1+2 = 3, /4
    vertical = (abs((1 - 2) - (1 - 1)) + abs((1 - 2) - (1 - 2)) + abs((1 - 5) - (1 - 3))) / 3
    assert terms['depth'].item() == pytest.approx(depth)
    assert terms['gradient'].item() == pytest.approx(horizontal + vertical)
    assert loss.item() == pytest.approx(depth + horizontal + vertical)
    loss2, _ = canonical_depth_loss(pred, target, valid, depth_weight=2., gradient_weight=0.)
    assert loss2.item() == pytest.approx(2 * depth)


def test_invalid_pixels_cannot_contaminate_value_or_gradient():
    pred, target, valid = tensors([[2., 2., 5.], [1., 1., 1.]], [[1., 2., 3.], [1., 1., 1.]], [[1, 1, 1], [1, 1, 1]])
    clean, _ = canonical_depth_loss(pred, target, valid)
    pred2 = torch.tensor([[[[2., 2., 5., 7.], [1., 1., 1., float('nan')]]]], requires_grad=True)
    target2 = torch.tensor([[1., 2., 3., float('nan')], [1., 1., 1., float('inf')]]).reshape(1, 1, 2, 4)
    valid2 = torch.tensor([[1, 1, 1, 0], [1, 1, 1, 0]], dtype=torch.bool).reshape(1, 1, 2, 4)
    dirty, _ = canonical_depth_loss(pred2, target2, valid2)
    # Different valid-pair sets than the 3-column case only through the removed column; compare by recomputation.
    ref, _ = canonical_depth_loss(pred2[..., :3].detach().requires_grad_(), target2[..., :3], valid2[..., :3])
    assert torch.isfinite(dirty) and dirty.item() == pytest.approx(ref.item()) == pytest.approx(clean.item())
    dirty.backward()
    assert torch.isfinite(pred2.grad).all() and (pred2.grad[..., 3] == 0).all()


def test_invalid_boundary_pairs_are_excluded():
    # valid, invalid, valid on one row: no neighbor pair is valid, so gradient term is exactly zero.
    pred, target, valid = tensors([[1., 100., 3.]], [[1., 2., 4.]], [[1, 0, 1]])
    _, terms = canonical_depth_loss(pred, target, valid)
    assert terms['gradient'].item() == 0.0
    assert terms['depth'].item() == pytest.approx(0.5)
    assert mask_counts(valid) == (2, 0, 0)


def test_failure_modes_are_explicit():
    pred, target, valid = tensors([[1., 2.]], [[1., 2.]], [[0, 0]])
    with pytest.raises(ValueError):
        canonical_depth_loss(pred, target, valid)                  # all invalid
    pred, target, valid = tensors([[1., 2.]], [[1., float('nan')]], [[1, 1]])
    with pytest.raises(ValueError):
        canonical_depth_loss(pred, target, valid)                  # marked-valid nan label
    pred, target, valid = tensors([[1., 2.]], [[1., -1.]], [[1, 1]])
    with pytest.raises(ValueError):
        canonical_depth_loss(pred, target, valid)                  # nonpositive label
    pred, target, valid = tensors([[1., float('inf')]], [[1., 2.]], [[1, 1]])
    with pytest.raises(ValueError):
        canonical_depth_loss(pred, target, valid)                  # nonfinite valid prediction
    pred, target, valid = tensors([[1., 2.]], [[1., 2.]], [[1, 0]])
    with pytest.raises(ValueError):
        canonical_depth_loss(pred, target, valid.float())          # mask must be boolean
    with pytest.raises(ValueError):
        canonical_depth_loss(pred, target, valid, depth_weight=0., gradient_weight=0.)


def test_global_denominators_make_microbatches_equal_full_batch():
    torch.manual_seed(0)
    target = torch.rand(4, 1, 6, 7) + .1
    valid = torch.rand(4, 1, 6, 7) > .3
    valid[:, :, 0, 0] = True
    base = torch.rand(4, 1, 6, 7) + .1
    full = base.clone().requires_grad_()
    loss, _ = canonical_depth_loss(full, target, valid)
    loss.backward()
    parts = base.clone().requires_grad_()
    counts, total = mask_counts(valid), 0.
    for s in (slice(0, 1), slice(1, 4)):
        part_loss, _ = canonical_depth_loss(parts[s], target[s], valid[s], denominators=counts)
        part_loss.backward()
        total += part_loss.item()
    assert total == pytest.approx(loss.item(), rel=1e-6)
    assert torch.allclose(parts.grad, full.grad, atol=1e-7)
