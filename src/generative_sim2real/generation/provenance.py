"""Code/config provenance for P4/P5 outputs: sha256 and modification time of the files that
produced them, recorded inside render.json / proxy.json / appearance.json / matrix.json.

This records the file versions present when the output was written; it cannot prove which
version produced an older output."""
import argparse
import datetime
import hashlib
import json
from pathlib import Path

CONFIGS = ('configs/generator.yaml', 'configs/p4.yaml', 'configs/p5.yaml', 'configs/p5_v2.yaml',
           'configs/p5_sweep.yaml', 'configs/p5_review_v2.yaml')


def provenance(root='.'):
    root = Path(root)
    files = sorted((root / 'src/generative_sim2real/generation').glob('*.py'))
    files += [root / c for c in CONFIGS if (root / c).exists()]
    files += sorted((root / 'scripts').glob('*p4*.sh')) + sorted((root / 'scripts').glob('*p5*.sh'))
    return {str(p.relative_to(root)): {
        'sha256': hashlib.sha256(p.read_bytes()).hexdigest(),
        'mtime_utc': datetime.datetime.fromtimestamp(
            p.stat().st_mtime, datetime.timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')}
        for p in files}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', required=True)
    parser.add_argument('--note', default='')
    args = parser.parse_args()
    Path(args.out).write_text(json.dumps(
        {'recorded_utc': datetime.datetime.now(datetime.timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ'),
         'note': args.note, 'files': provenance()}, indent=1) + '\n')


if __name__ == '__main__':
    main()
