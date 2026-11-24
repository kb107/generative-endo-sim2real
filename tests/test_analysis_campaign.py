"""Fixture test for the pre-registered campaign analysis (no real campaign or model needed)."""
import csv
import json
import math

import pytest

from generative_sim2real.analysis import campaign

FAMILIES = [f'c2_fam{i:02d}' for i in range(18)]


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def fake_campaign(tmp_path, arms):
    """arms: name -> (headline abs_rel, per-family abs_rel function, scale s)."""
    headline, oracle, records = [], [], []
    for name, (abs_rel, fam_value, s) in arms.items():
        headline.append({'method': name, 'abs_rel': abs_rel, 'rmse_mm': 10.0, 'delta_1_25': 0.5})
        oracle.append({'method': name, 'sequence_macro_oracle_rmse_mm': 5.0})
        records.append({'name': name, 'selected_step': 3000})
        frames, seqs = [], []
        for i, fam in enumerate(FAMILIES):
            for v in ('v1', 'v2'):
                sid = f'{fam}_{v}'
                seqs.append({'sequence_id': sid, 'abs_rel': fam_value(i)})
                for k, gt in enumerate((0.02, 0.04)):
                    frames.append({'sample_id': f'{sid}/{k}', 'sequence_id': sid, 'status': 'ok',
                                   'oracle_median_scale': s, 'median_gt_m': gt, 'median_pred_m': gt / s})
        write_csv(tmp_path / 'methods' / name / 'frames.csv', frames)
        write_csv(tmp_path / 'methods' / name / 'sequences.csv', seqs)
    write_csv(tmp_path / 'comparison_headline.csv', headline)
    write_csv(tmp_path / 'comparison_oracle_diagnostic.csv', oracle)
    (tmp_path / 'registry.json').write_text(json.dumps({'methods': records}))
    (tmp_path / 'COMPLETE').write_text('x')
    return tmp_path


def test_family_strips_version_suffix():
    assert campaign.family('c2_cecum_t1_v2') == 'c2_cecum_t1'


def test_rules_and_scale(tmp_path):
    arms = {
        'R': (0.75, lambda i: 0.75, 0.7), 'R_s1': (0.73, lambda i: 0.73, 0.7),
        'G': (0.37, lambda i: 0.37, 1.0), 'G_s1': (0.38, lambda i: 0.38, 1.0), 'Gp': (0.36, lambda i: 0.36, 1.0),
        'S': (0.38, lambda i: 0.36 if i % 2 else 0.38, 1.0),   # close, wins 9/18 -> S ~ G
        'T': (0.60, lambda i: 0.60, 0.8),                      # far, loses all -> G >> T
        'A_mild': (0.42, lambda i: 0.42, 0.9),                 # 0.05 away, loses all, below 3 sigma=0.06 -> inconclusive
    }
    result = campaign.analyse(fake_campaign(tmp_path, arms))
    assert result['sigma'] == pytest.approx(0.02)
    verdicts = {c['x']: c['verdict'] for c in result['comparisons']}
    assert verdicts['S'] == 'S ~ G'
    assert verdicts['T'] == 'G >> T'
    assert verdicts['A_mild'] == 'inconclusive'
    assert verdicts['R'] == 'G >> R'
    r = next(a for a in result['arms'] if a['method'] == 'R')
    assert r['scale_factor_pred_over_gt'] == pytest.approx(1 / 0.7)
    assert r['sd_log_s_after_global'] == pytest.approx(0.0, abs=1e-12)
    assert r['slope_log_pred_on_log_gt'] == pytest.approx(1.0)
    assert r['intercept'] == pytest.approx(-math.log(0.7))
