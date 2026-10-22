"""Shared helpers: JSON/markdown/CSV writers, image loading and size-limited PNG saving."""
import csv
import io
import json
import re
from pathlib import Path

import numpy as np

MAX_PNG_BYTES = 2_000_000


def read_json(path):
    return json.loads(Path(path).read_text())


def fmt(value, digits=3):
    """Compact cell text. None -> 'pending'."""
    if value is None:
        return 'pending'
    if isinstance(value, bool):
        return 'yes' if value else 'no'
    if isinstance(value, float):
        return f'{value:.{digits}f}'
    return str(value)


def _md_cell(value):
    return str(value).replace('|', '/').replace('\n', ' ')


def write_table(stem, title, header, rows, notes=()):
    """Write `<stem>.md` and `<stem>.csv` from the same rows (strings or numbers)."""
    stem = Path(stem)
    stem.parent.mkdir(parents=True, exist_ok=True)
    rows = [[fmt(c) if not isinstance(c, str) else c for c in row] for row in rows]
    for row in rows:
        if len(row) != len(header):
            raise ValueError(f'Row width {len(row)} != header width {len(header)}: {row}')
    lines = [f'# {title}', '']
    for note in notes:
        lines += [note, '']
    lines.append('| ' + ' | '.join(_md_cell(h) for h in header) + ' |')
    lines.append('| ' + ' | '.join('---' for _ in header) + ' |')
    lines += ['| ' + ' | '.join(_md_cell(c) for c in row) + ' |' for row in rows]
    stem.with_suffix('.md').write_text('\n'.join(lines) + '\n')
    buffer = io.StringIO()
    writer = csv.writer(buffer, lineterminator='\n')
    writer.writerow(header)
    writer.writerows(rows)
    stem.with_suffix('.csv').write_text(buffer.getvalue())


def load_rgb(path, size=None):
    from PIL import Image

    image = Image.open(path).convert('RGB')
    if size is not None and image.size != (size, size):
        image = image.resize((size, size), Image.LANCZOS)
    return np.asarray(image)


def scene_label(sample_id):
    """'simcol3d/SyntheticColon_I/Frames_S14/0600' -> 'Colon I S14 #0600'."""
    parts = sample_id.split('/')
    if len(parts) < 4:
        return sample_id
    return f"{parts[1].replace('SyntheticColon_', 'Colon ')} {parts[2].replace('Frames_', '')} #{parts[3]}"


def checkpoint_label(path):
    """'.../runs/target_lora_v2/checkpoints/step_002000/lora_target.safetensors' -> 'target_lora_v2 step 2000'."""
    match = re.search(r'runs/([^/]+)/checkpoints/step_0*(\d+)/', str(path))
    return f'{match.group(1)} step {match.group(2)}' if match else Path(path).name


def inverse_depth_to_mm(value, near_m, far_m):
    """Invert the controller encoding v = (1/d - 1/far) / (1/near - 1/far); returns millimetres."""
    inverse = 1.0 / far_m + np.asarray(value, dtype=np.float64) * (1.0 / near_m - 1.0 / far_m)
    return 1000.0 / inverse


def save_figure_png(fig, path, max_bytes=MAX_PNG_BYTES, dpis=(120, 100, 85, 70, 60, 50)):
    """Save at the largest dpi from `dpis` whose PNG fits `max_bytes`; return (dpi, bytes).

    The figure content is identical at every dpi (only resolution changes)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    size = 0
    for dpi in dpis:
        fig.savefig(path, dpi=dpi, facecolor='white')
        size = path.stat().st_size
        if size <= max_bytes:
            return dpi, size
    raise RuntimeError(f'{path} is {size} bytes at dpi {dpis[-1]}; limit {max_bytes}')
