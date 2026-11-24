import json

import numpy as np
import pytest
import torch

from da3_fakes import FakeSplit, write_generated, write_ids
from generative_sim2real.data import rgb_modes as rm

N = 6


@pytest.fixture
def source(tmp_path):
    return FakeSplit(tmp_path, 'source_train', 'source_generation512', N, 512)


def ids_of(source):
    return [r['sample_id'] for r in source.rows]


def build(tmp_path, source, mode, mutate=None, ids=None, **kw):
    ids = ids or ids_of(source)
    manifest = write_generated(tmp_path, source, ids, tmp_path / 'out', mutate=mutate) if mode in {'G', 'M'} or kw.pop('force', False) else None
    return rm.RGBModeDataset(source, ids, mode=mode, seed=0, generated_manifest=manifest, outputs_root=tmp_path / 'out')


def test_matched_id_artifact_contract(tmp_path, source):
    ids = ids_of(source)
    assert rm.load_matched_ids(write_ids(tmp_path, source, ids), source, smoke=True) == ids
    with pytest.raises(ValueError):                 # smoke artifact cannot serve an experiment
        rm.load_matched_ids(write_ids(tmp_path, source, ids), source, smoke=False)
    with pytest.raises(ValueError):                 # experiment artifact must be frozen
        rm.load_matched_ids(write_ids(tmp_path, source, ids, smoke=False, frozen=False), source)
    assert rm.load_matched_ids(write_ids(tmp_path, source, ids, smoke=False, frozen=True), source) == ids
    with pytest.raises(ValueError):                 # smoke is rejected where experiment is required
        rm.load_matched_ids(write_ids(tmp_path, source, ids, smoke=True), source, smoke=False)
    for bad in (ids[:2] + ids[:1], ids + ['fake/unknown/9999'], []):
        with pytest.raises(ValueError):
            rm.load_matched_ids(write_ids(tmp_path, source, bad), source, smoke=True)


def test_matched_ids_bind_current_source_manifest(tmp_path, source):
    path = write_ids(tmp_path, source, ids_of(source))
    source.path.write_text(source.path.read_text() + '\n')
    with pytest.raises(ValueError):
        rm.load_matched_ids(path, source, smoke=True)


def test_write_smoke_ids_is_labeled_unfrozen_and_not_an_experiment_artifact(tmp_path, source):
    artifact = rm.write_smoke_ids(source.path, 3, tmp_path / 'smoke/ids.json')
    assert artifact['purpose'] == 'smoke' and artifact['frozen'] is False and 'SMOKE' in artifact['label']
    assert len(artifact['sample_ids']) == 3 == len(set(artifact['sample_ids']))
    with pytest.raises(ValueError):
        rm.load_matched_ids(tmp_path / 'smoke/ids.json', source, smoke=False)
    val = FakeSplit(tmp_path, 'source_validation', 'source_native', 3, 56)
    with pytest.raises(ValueError):
        rm.write_smoke_ids(val.path, 2, tmp_path / 'x.json')


def test_freeze_ids_writes_loadable_frozen_artifact_and_refuses_overwrite(tmp_path, source):
    out = tmp_path / 'frozen/ids.json'
    artifact = rm.freeze_ids(source.path, 3, out)
    assert artifact['purpose'] == 'experiment' and artifact['frozen'] is True
    assert rm.load_matched_ids(out, source) == artifact['sample_ids']
    assert rm.freeze_ids.__doc__ and out.read_text() == (tmp_path / 'frozen/ids.json').read_text()
    with pytest.raises(FileExistsError):
        rm.freeze_ids(source.path, 3, out)


def test_only_source_train_generation_geometry_is_accepted(tmp_path):
    for split, geometry in (('target_test', 'target_native'), ('source_validation', 'source_native'),
                            ('source_train', 'source_da3_504')):
        with pytest.raises(ValueError):
            rm.RGBModeDataset(FakeSplit(tmp_path, split, geometry, 2, 512), ['fake/seq0/0000'])


def test_labels_and_geometry_identical_across_all_modes(tmp_path, source):
    datasets = {m: build(tmp_path, source, m) for m in 'RAGM'}
    for i in range(N):
        ref = datasets['R'].get_occurrence(i, i)
        assert ref['rgb'].shape == (3, 504, 504) and ref['targets']['depth_m'].shape == (1, 504, 504)
        assert ref['rgb'].dtype == torch.float32 and 0 <= ref['rgb'].min() and ref['rgb'].max() <= 1
        assert ref['camera']['K'][0, 0] == pytest.approx(600 * 504 / 512)
        for mode in 'AGM':
            s = datasets[mode].get_occurrence(i, i)
            assert s['sample_id'] == ref['sample_id']
            assert torch.equal(s['targets']['depth_valid'], ref['targets']['depth_valid'])
            valid = ref['targets']['depth_valid']
            assert torch.equal(s['targets']['depth_m'][valid], ref['targets']['depth_m'][valid])
            assert torch.equal(s['camera']['K'], ref['camera']['K'])
            assert s['meta']['transform_history'] == ref['meta']['transform_history']
        assert not torch.equal(datasets['G'].get_occurrence(i, i)['rgb'], ref['rgb'])


def test_raw_uses_the_512_to_504_chain_and_a_changes_only_rgb_within_bounds(tmp_path, source):
    r, a = build(tmp_path, source, 'R'), build(tmp_path, source, 'A')
    raw = r.get_occurrence(0, 0)['rgb']
    direct = rm.ResizeSample((504, 504))(source[0])['rgb']
    assert torch.equal(raw, direct)
    aug = a.get_occurrence(0, 0)['rgb']
    assert not torch.equal(aug, raw) and 0 <= aug.min() and aug.max() <= 1
    # Modest photometric change only: bounded per-pixel deviation and no spatial reordering.
    assert (aug - raw).abs().max() < .35
    assert torch.corrcoef(torch.stack([aug.flatten(), raw.flatten()]))[0, 1] > .95


def test_appearance_randomness_is_occurrence_local_not_global_or_order_dependent(tmp_path, source):
    a = build(tmp_path, source, 'A')
    first = a.get_occurrence(2, 17)['rgb']
    torch.manual_seed(5); np.random.seed(5)
    a.get_occurrence(0, 3); a.get_occurrence(1, 4)          # consuming other draws must not matter
    assert torch.equal(a.get_occurrence(2, 17)['rgb'], first)
    assert torch.equal(build(tmp_path, source, 'A').get_occurrence(2, 17)['rgb'], first)
    assert not torch.equal(a.get_occurrence(2, 18)['rgb'], first)
    other_seed = rm.RGBModeDataset(source, ids_of(source), mode='A', seed=1)
    assert not torch.equal(other_seed.get_occurrence(2, 17)['rgb'], first)


def test_mixture_replaces_about_half_by_occurrence_and_matches_g_or_r(tmp_path, source):
    r, g, m = (build(tmp_path, source, x) for x in 'RGM')
    flags = []
    for occ in range(160):
        i = occ % N
        s = m.get_occurrence(i, occ)['rgb']
        is_g = torch.equal(s, g.get_occurrence(i, occ)['rgb'])
        assert is_g or torch.equal(s, r.get_occurrence(i, occ)['rgb'])
        flags.append(is_g)
    assert 0.35 < np.mean(flags) < 0.65
    assert [torch.equal(m.get_occurrence(i % N, i)['rgb'], g.get_occurrence(i % N, i)['rgb']) for i in range(40)] == flags[:40]


def test_occurrence_order_is_mode_independent_permutation_per_epoch():
    order = rm.OccurrenceOrder(5, 3)
    epoch0 = [order.at(c) for c in range(5)]
    epoch1 = [order.at(c) for c in range(5, 10)]
    assert sorted(epoch0) == sorted(epoch1) == list(range(5)) and epoch0 != epoch1
    assert [rm.OccurrenceOrder(5, 3).at(c) for c in range(10)] == epoch0 + epoch1
    assert [rm.OccurrenceOrder(5, 3).at(c) for c in range(9, -1, -1)] == (epoch0 + epoch1)[::-1]
    for bad in (-1, 1.5):
        with pytest.raises(ValueError):
            order.at(bad)


def drop_last(records): records.pop()
def duplicate(records): records[-1] = dict(records[0])
def rename(records): records[0]['sample_id'] = 'fake/other/0000'
def bad_checksum(records): records[0]['generated_rgb_sha256'] = '0' * 64
def other_depth(records): records[0]['source_depth_path'] = 'fake/depth_9999.png'
def bad_fingerprint(records): records[0]['preprocessing_sha256'] = '1' * 64
def no_geometry(records): records[0].pop('source_geometry_sha256')
def missing_provenance(records): records[0].pop('controller')
def wrong_size(records): records[0]['size_hw'] = [256, 256]
def escape(records): records[0]['generated_rgb_path'] = '../ids.json'
def absolute(records): records[0]['generated_rgb_path'] = '/etc/hostname'
def null_value(records): records[1]['prompt'] = None
def different_adapter(records): records[2]['target_adapter'] = 'another'
def different_weight(records): records[1]['adapter_weight'] = 0.5
def wrong_transform(records): records[0]['source_transform_id'] = 'source_da3_504'


@pytest.mark.parametrize('mutate', [drop_last, duplicate, rename, bad_checksum, other_depth, bad_fingerprint, no_geometry,
                                    missing_provenance, wrong_size, escape, absolute, wrong_transform,
                                    null_value, different_adapter, different_weight])
@pytest.mark.parametrize('mode', ['G', 'M'])
def test_incomplete_or_mismatched_generated_manifests_are_rejected(tmp_path, source, mode, mutate):
    with pytest.raises((ValueError, KeyError)):
        build(tmp_path, source, mode, mutate=mutate)


def test_generated_manifest_policy_by_mode(tmp_path, source):
    ids = ids_of(source)
    manifest = write_generated(tmp_path, source, ids, tmp_path / 'out')
    for mode in 'RA':
        with pytest.raises(ValueError):
            rm.RGBModeDataset(source, ids, mode=mode, generated_manifest=manifest, outputs_root=tmp_path / 'out')
    for mode in 'GM':
        with pytest.raises(ValueError):
            rm.RGBModeDataset(source, ids, mode=mode)
        with pytest.raises(ValueError):
            rm.RGBModeDataset(source, ids, mode=mode, generated_manifest=manifest)
    with pytest.raises(ValueError):
        rm.RGBModeDataset(source, ids, mode='Z')


def test_generated_label_identity_changes_are_detected_at_load(tmp_path, source):
    g = build(tmp_path, source, 'G')
    g.get_occurrence(0, 0)
    source.depth_bias = .001                     # the source labels changed after generation
    with pytest.raises(ValueError):
        g.get_occurrence(0, 0)


def test_generated_image_modified_after_validation_is_rejected(tmp_path, source):
    g = build(tmp_path, source, 'G')
    (tmp_path / 'out/gen/0000.png').write_bytes(b'tampered')
    with pytest.raises(ValueError):
        g.get_occurrence(0, 0)


def test_geometry_hash_binds_depth_mask_k_and_history(source):
    base = rm.geometry_sha256(source[0])
    changed = source[0]; changed['camera']['K'][0, 0] += 1
    assert rm.geometry_sha256(changed) != base
    changed = source[0]; changed['targets']['depth_valid'][0, 100, 100] ^= True
    assert rm.geometry_sha256(changed) != base
    changed = source[0]; changed['targets']['depth_m'][0, 100, 100] += .001
    assert rm.geometry_sha256(changed) != base
    changed = source[0]; changed['meta']['transform_history'] = [{'name': 'crop'}]
    assert rm.geometry_sha256(changed) != base
    unlabeled = source[0]; unlabeled['targets']['depth_m'] = None
    with pytest.raises(ValueError):
        rm.geometry_sha256(unlabeled)


def test_per_id_seed_may_differ_but_recipe_may_not(tmp_path, source):
    ds = build(tmp_path, source, 'G')            # write_generated gives every record a different seed
    assert len({r['seed'] for r, _ in ds.generated.values()}) == N


def test_order_seeds_do_not_share_shifted_epochs():
    a, b = rm.OccurrenceOrder(8, 0), rm.OccurrenceOrder(8, 1)
    assert [b.at(c) for c in range(8)] != [a.at(c) for c in range(8, 16)]
    assert [a.at(c) for c in range(8)] != [b.at(c) for c in range(8)]
