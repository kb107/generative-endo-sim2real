"""Contact sheet of the fixed frames: real SimCol | depth control | ours (target) | editor | C1 refs.

  external/foundation-venv/bin/python -m generative_sim2real.foundation.contact_sheet \
      --editor Qwen=outputs/foundation/qwen/fixed8/c{cond}.png --out outputs/foundation/contact_sheet.png
"""
import argparse
import json
from pathlib import Path

from PIL import Image, ImageDraw

CELL = 224


def build(spec, editors, out, cell=CELL):
    """editors: [(label, template with {cond})]; a missing file is drawn as a blank cell."""
    columns = ['real SimCol', 'depth (control)', 'ours: target_correct', *[e[0] for e in editors],
               'C1 ref 1', 'C1 ref 2']
    header = 22
    sheet = Image.new('RGB', (cell * len(columns), header + cell * len(spec['fixed8'])), 'white')
    draw = ImageDraw.Draw(sheet)
    for j, name in enumerate(columns):
        draw.text((j * cell + 4, 5), name, fill='black')

    def put(path, row, col):
        if Path(path).exists():
            sheet.paste(Image.open(path).convert('RGB').resize((cell, cell), Image.LANCZOS),
                        (col * cell, header + row * cell))
    for i, item in enumerate(spec['fixed8']):
        paths = [item['real'], item['control'], item['target_correct'],
                 *[t.format(cond=item['cond']) for _, t in editors], spec['refs'][0]['file'],
                 spec['refs'][1]['file']]
        for j, path in enumerate(paths):
            put(path, i, j)
        draw.text((6, header + i * cell + 4), f'c{item["cond"]}', fill='white')
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inputs', default='outputs/foundation/inputs/inputs.json')
    parser.add_argument('--editor', action='append', default=[], help='LABEL=TEMPLATE using {cond}')
    parser.add_argument('--out', default='outputs/foundation/contact_sheet.png')
    args = parser.parse_args()
    editors = [tuple(e.split('=', 1)) for e in args.editor]
    build(json.loads(Path(args.inputs).read_text()), editors, args.out)
    print('wrote', args.out)


if __name__ == '__main__':
    main()
