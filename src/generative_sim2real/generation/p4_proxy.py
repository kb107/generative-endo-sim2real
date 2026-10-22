"""DA3 proxy scoring of P4 gate renders (DA3 venv; separate step; read-only model).

Loads the frozen official DA3Metric-Large checkpoint (no fine-tuned head), predicts depth for
every image listed in render.json (including the real source-validation RGB for the proxy audit
R0), and scores each against the condition's own reference depth and its wrong-depth partner's
(scale-free median-scaled AbsRel and stride-4 Spearman; see p4_report.py and configs/p4.yaml).
It reports what it measures; the pass rules are applied later by p4_report.py.
Run: PYTHONPATH=src external/da3-venv/bin/python -m generative_sim2real.generation.p4_proxy --render-dir D
"""
import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F

from generative_sim2real.depth.checkpoints import head_digest
from generative_sim2real.depth.model import (CHECKPOINT, CHECKPOINT_REVISION, DA3MetricDepth,
                                             verify_upstream_checkout)

from . import p4_report as report
from .provenance import provenance


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--render-dir', required=True)
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()
    directory = Path(args.render_dir)
    render = json.loads((directory / 'render.json').read_text())
    refs = torch.load(directory / 'refs.pt', weights_only=True)
    partner = render['partner']
    verify_upstream_checkout()
    model = DA3MetricDepth.from_pretrained().to(args.device).eval().requires_grad_(False)
    digest_before = head_digest(model.head_state_dict())
    rows = []
    with torch.no_grad():
        for item in render['files']:
            rgb = report.png_tensor(directory / item['file'])[None].to(args.device)
            i = item['cond']
            K = refs['K'][i][None].to(args.device)
            depth = model(rgb, K)['depth_m']
            depth = F.interpolate(depth, size=refs['depth_m'].shape[-2:], mode='bilinear',
                                  align_corners=False)[0].cpu()
            row = {k: item[k] for k in ('arm', 'cond', 'seed_index', 'seed', 'file')}
            for name, j in (('own', i), ('partner', partner[i])):
                reference, valid = refs['depth_m'][j], refs['valid'][j] & (refs['depth_m'][j] > 0)
                row[f'abs_rel_{name}'] = report.median_scaled_abs_rel(depth, reference, valid)
                row[f'spearman_{name}'] = report.spearman(depth, reference, valid)
            rows.append(row)
    digest_after = head_digest(model.head_state_dict())
    result = {'model': {'checkpoint': CHECKPOINT, 'revision': CHECKPOINT_REVISION,
                        'head_sha256_before': digest_before, 'head_sha256_after': digest_after,
                        'unchanged': digest_before == digest_after,
                        'fine_tuned': False, 'scoring': 'median-scaled, K-independent'},
              'rows': rows, 'code_provenance': provenance()}
    (directory / 'proxy.json').write_text(json.dumps(result, indent=1, allow_nan=False) + '\n')
    print(f'scored {len(rows)} images; head unchanged: {result["model"]["unchanged"]}')


if __name__ == '__main__':
    main()
