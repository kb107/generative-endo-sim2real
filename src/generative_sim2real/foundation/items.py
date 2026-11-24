"""Read the fixed-input spec (inputs.json) into editor work items (no torch)."""
import json
from pathlib import Path

SIZE = 512


def load_items(inputs_json, which):
    """-> (items, refs): items = [{'key', 'name', 'real'}]; refs = [ref1_path, ref2_path].

    Output file names: fixed8 -> c{cond}.png, subset300 -> {index:03d}.png."""
    spec = json.loads(Path(inputs_json).read_text())
    if which == 'fixed8':
        items = [{'key': r['cond'], 'name': f'c{r["cond"]}', 'real': r['real'], 'sample_id': r['sample_id']}
                 for r in spec['fixed8']]
    elif which == 'subset300':
        items = [{'key': r['index'], 'name': f'{r["index"]:03d}', 'real': r['real'],
                  'sample_id': r['sample_id']} for r in spec['subset300']]
    else:
        raise ValueError(f'unknown set {which!r}')
    return items, [r['file'] for r in spec['refs']]
