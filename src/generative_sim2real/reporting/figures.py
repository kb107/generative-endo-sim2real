"""Figure builders. Each takes explicit paths/dicts and writes one PNG (or returns the figure)."""
import textwrap
from pathlib import Path

import numpy as np

from .common import (checkpoint_label, fmt, inverse_depth_to_mm, load_rgb, read_json, save_figure_png,
                     scene_label)

TILE = 1.55            # inches per square tile; every image panel uses the same square geometry
LABEL_W = 2.55         # inches reserved for row labels
HEADER_H = 0.55
GAP = 0.06


def _mpl():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    return plt


def _grid_figure(n_rows, col_titles, *, label_w=LABEL_W, footer_h=1.0, extra_bottom=0.0):
    """Figure with a label column, `len(col_titles)` square tile columns and `n_rows` rows.

    Returns (fig, axes[row][col], label_axes[row]) in a fixed pixel-true layout."""
    plt = _mpl()
    n_cols = len(col_titles)
    width = label_w + n_cols * (TILE + GAP)
    height = HEADER_H + n_rows * (TILE + GAP) + footer_h + extra_bottom
    fig = plt.figure(figsize=(width, height), facecolor='white')
    axes, labels = [], []
    for r in range(n_rows):
        top = height - HEADER_H - r * (TILE + GAP)
        labels.append(fig.add_axes([0.02 / width, (top - TILE) / height, (label_w - 0.06) / width, TILE / height]))
        labels[-1].axis('off')
        row = []
        for c in range(n_cols):
            left = label_w + c * (TILE + GAP)
            ax = fig.add_axes([left / width, (top - TILE) / height, TILE / width, TILE / height])
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_linewidth(0.4)
            row.append(ax)
        axes.append(row)
    for c, title in enumerate(col_titles):
        left = label_w + c * (TILE + GAP)
        fig.text((left + TILE / 2) / width, (height - HEADER_H + 0.05) / height, title, ha='center',
                 va='bottom', fontsize=8.5, fontweight='bold', linespacing=1.15)
    fig._geom = (width, height)
    return fig, axes, labels


def _caption(fig, text, y_in=0.08, wrap=150, size=8):
    width, _ = fig._geom
    fig.text(0.02 / width, y_in / fig._geom[1], '\n'.join(textwrap.wrap(text, wrap)), ha='left', va='bottom',
             fontsize=size, color='#222222', linespacing=1.25)


def _control_colorbar(fig, ax_rect_in, near_m, far_m):
    """Horizontal gray bar labelled in millimetres (the controller's inverse-depth encoding)."""
    width, height = fig._geom
    left, bottom, w, h = ax_rect_in
    cax = fig.add_axes([left / width, bottom / height, w / width, h / height])
    cax.imshow(np.linspace(0, 1, 256)[None, :], cmap='gray', aspect='auto', vmin=0, vmax=1, extent=[0, 1, 0, 1])
    ticks = [0.0, 0.25, 0.5, 0.75, 1.0]
    cax.set_xticks(ticks)
    cax.set_xticklabels([f'{inverse_depth_to_mm(t, near_m, far_m):.0f}' for t in ticks], fontsize=6)
    cax.set_yticks([])
    cax.set_xlabel('control brightness -> depth (mm)', fontsize=6, labelpad=1)
    return cax


def generation_figure(run, encoding, path, *, columns=None):
    """Required eight-row figure (docs spec section 5, figure 1) from one saved reduced-matrix directory."""
    mm = run.median_depth_mm()
    near_m, far_m = encoding['constants']['near_m'], encoding['constants']['far_m']
    columns = columns or [
        ('Input metric depth\n(inverse-depth control)', 'control'),
        ('Source RGB\n(real, held-out)', 'real'),
        ('Source LoRA\ncorrect depth', 'source_correct'),
        ('Target LoRA v2\ncorrect depth', 'target_correct'),
        ('Target LoRA v2\ncontroller OFF', 'target_off'),
        ('Target LoRA v2\nWRONG depth', 'target_wrong')]
    n = len(run.ids)
    fig, axes, labels = _grid_figure(n, [c[0] for c in columns], footer_h=1.45)
    for r in range(n):
        for c, (_, arm) in enumerate(columns):
            ax = axes[r][c]
            image = run.image(arm, r)
            if arm == 'control':
                ax.imshow(image[..., 0], cmap='gray', vmin=0, vmax=255)
            else:
                ax.imshow(image)
        labels[r].text(0, 0.62, f'c{r}  {scene_label(run.ids[r])}', fontsize=8.5, fontweight='bold', va='center')
        labels[r].text(0, 0.40, f'median depth {mm[r]:.1f} mm', fontsize=8, va='center')
        labels[r].text(0, 0.22, f'wrong depth = c{run.partner[r]} ({mm[run.partner[r]]:.1f} mm)', fontsize=8,
                       va='center', color='#444444')
        labels[r].set_xlim(0, 1)
        labels[r].set_ylim(0, 1)
    width, height = fig._geom
    _control_colorbar(fig, (LABEL_W, 0.95 + 0.5, TILE, 0.12), near_m, far_m)
    weights = run.render['weights']
    inference = run.render['inference']
    caption = (
        f"Figure 1. Fixed-depth generation, SOURCE-validation conditions only (no phantom/C2 data). Single noise seed per "
        f"condition ({run.seed} + condition index, identical in every column), {n} conditions, recipe v2: "
        f"{checkpoint_label(weights['lora_target']['path'])}, {checkpoint_label(weights['lora_source']['path'])}, "
        f"controller {checkpoint_label(weights['controller']['path'])}, control scale {run.render['control_scale']}, "
        f"guidance {inference['guidance_scale']}, {inference['scheduler']} {inference['steps']} steps. Depth column: "
        f"the saved control image (gray = normalised inverse depth, white = near, black = far; linear in 1/depth between "
        f"{near_m * 1000:.1f} mm and {far_m * 1000:.1f} mm, outside values clipped; the bar converts brightness to mm). "
        f"Row labels give the condition id, scene and median valid depth. Wrong depth = another scene's depth (fixed "
        f"derangement, partner in the label). All panels are square 512 px frames shown without cropping. This is "
        f"qualitative; geometry is quantified only by the DA3 proxy scores in the tables.")
    _caption(fig, caption, y_in=0.08, wrap=int(width * 16), size=7.5)
    return _save(fig, path)


def _save(fig, path):
    plt = _mpl()
    dpi, size = save_figure_png(fig, path)
    plt.close(fig)
    return {'path': str(path), 'dpi': dpi, 'bytes': size}


def failure_figure(run, cells, path, *, rule_text, title, shown_gain=None, seed_label=None, extra_gain_label=None):
    """Failure panel: one row per declared (adapter, condition) cell.

    cells: list of {'adapter', 'condition', 'M1_gain'} in rank order (as stored by the declared rule)."""
    mm = run.median_depth_mm()
    col_titles = ['Source RGB\n(real)', 'Control\n(own depth)', 'Controller OFF', 'Correct depth', 'WRONG depth',
                  'Supplied (wrong)\ncontrol']
    fig, axes, labels = _grid_figure(len(cells), col_titles, footer_h=1.45)
    for r, cell in enumerate(cells):
        adapter, i = cell['adapter'], cell['condition']
        p = run.partner[i]
        tiles = [('real', i), ('control', i), (f'{adapter}_off', i), (f'{adapter}_correct', i),
                 (f'{adapter}_wrong', i), ('control', p)]
        for c, (arm, cond) in enumerate(tiles):
            image = run.image(arm, cond)
            if arm == 'control':
                axes[r][c].imshow(image[..., 0], cmap='gray', vmin=0, vmax=255)
            else:
                axes[r][c].imshow(image)
        labels[r].set_xlim(0, 1)
        labels[r].set_ylim(0, 1)
        labels[r].text(0, 0.80, f'rank {r + 1}: {adapter} adapter, c{i}', fontsize=8.5, fontweight='bold', va='center')
        labels[r].text(0, 0.62, scene_label(run.ids[i]), fontsize=8, va='center')
        labels[r].text(0, 0.46, f'median depth {mm[i]:.1f} mm; wrong = c{p}', fontsize=8, va='center')
        labels[r].text(0, 0.30, f"M1 gain {cell['M1_gain']:+.3f}"
                       + (f" ({extra_gain_label})" if extra_gain_label else ''), fontsize=8, va='center')
        if shown_gain is not None:
            labels[r].text(0, 0.14, f'this seed: {shown_gain[r]:+.3f}', fontsize=8, va='center', color='#444444')
    width, height = fig._geom
    caption = (f'{title} Selection rule (declared before the run, not by inspection): {rule_text} '
               f'M1 gain = proxy AbsRel(controller off) - AbsRel(correct depth); small or negative means the controller '
               f'helped least. The six lowest cells are shown in rank order, attractive or not; no cell was added or dropped after '
               f'viewing images, and a negative gain would mean the controller made the proxy score worse. "none" = both LoRAs off (base weights). Source-validation conditions, ' + (seed_label or f'noise seed {run.seed} + index') + '.')
    _caption(fig, caption, y_in=0.08, wrap=int(width * 16), size=7.5)
    return _save(fig, path)


def reload_figure(appearance_dir, control_dir, appearance, reload_smoke, path, *, n_show=4):
    """Source -> target -> source adapter-swap example with the measured recovery tolerance.

    The second source render equals the first bit for bit (measured max abs error), so it is reported as a
    number, not displayed twice."""
    appearance_dir, control_dir = Path(appearance_dir), Path(control_dir)
    swap = appearance['swap_recovery_trained_adapters']
    rec = swap['source_second_vs_first']
    col_titles = ['Control\n(supplied depth)', '1. Source LoRA\n(first render)', '2. Swapped to target\nLoRA (same noise)',
                  '3. Swapped back\nto source LoRA']
    fig, axes, labels = _grid_figure(n_show, col_titles, footer_h=1.45)
    for r in range(n_show):
        axes[r][0].imshow(load_rgb(control_dir / 'control' / f'c{r}.png')[..., 0], cmap='gray', vmin=0, vmax=255)
        axes[r][1].imshow(load_rgb(appearance_dir / 'images' / 'source_controller' / f'c{r}.png'))
        axes[r][2].imshow(load_rgb(appearance_dir / 'images' / 'target_controller' / f'c{r}.png'))
        axes[r][3].axis('off')
        axes[r][3].text(0.5, 0.5, f"max |render 3 - render 1|\n= {rec['max_abs_error']:g}\n(pixel scale 0-1)\nrender 3 identical\nto render 1",
                        ha='center', va='center', fontsize=9, transform=axes[r][3].transAxes)
        labels[r].set_xlim(0, 1)
        labels[r].set_ylim(0, 1)
        labels[r].text(0, 0.5, f'condition c{r}', fontsize=9, fontweight='bold', va='center')
    width, _ = fig._geom
    smoke_t = reload_smoke['target_reload_error']
    caption = (
        f"Figure 3. Source -> target -> source adapter swap on the TRAINED adapters (controller on, "
        f"{len(appearance['source_validation_condition_ids'])} source-validation conditions; first {n_show} shown). "
        f"Measured reproducibility: after swapping to the target LoRA and back, the source render differs from the first "
        f"by max abs error {rec['max_abs_error']:.1g} over {rec['elements']:,} values (tolerance atol {rec['atol']:g}, "
        f"rtol {rec['rtol']:g}; {rec['violations']} violations); controller / source LoRA / target LoRA hashes unchanged "
        f"by the swaps: {fmt(swap['hashes_unchanged_by_swaps'])}; mean abs source-vs-target adapter effect "
        f"{swap['mean_abs_source_vs_target']:.3f}. The earlier P3 reload smoke gives max abs error "
        f"{smoke_t['max_abs_error']:.1g} over {smoke_t['elements']:,} values for a fresh-process target reload. The "
        f"third render is not displayed because it is bit-identical to the first.")
    _caption(fig, caption, y_in=0.08, wrap=int(width * 16), size=7.5)
    return _save(fig, path)


def appearance_figure(real_c1_grid, v1_dir, v2_dir, rows_table, path, *, n_show=8, grid_size=256, header=14,
                      pitch=270):
    """Real C1 validation vs generated (controller off) source LoRA, target LoRA v1 and v2, plus metrics table.

    The real frames are cropped from the saved appearance grid (column 0, 8 frames, same 256 px tiles)."""
    from PIL import Image

    grid = Image.open(real_c1_grid).convert('RGB')
    real = [np.asarray(grid.crop((0, header + r * pitch, grid_size, header + r * pitch + grid_size)))
            for r in range(n_show)]
    plt = _mpl()
    tile, gap = 1.35, 0.05
    label_w = 1.8
    table_h = 2.0
    n_rows = 4
    width = label_w + n_show * (tile + gap)
    height = 0.45 + n_rows * (tile + gap) + table_h + 0.9
    fig = plt.figure(figsize=(width, height), facecolor='white')
    sources = [
        ('Real C1 validation\n(held out)', real),
        ('Source LoRA\n(unconditional)', [load_rgb(Path(v1_dir) / 'images' / 'source_off' / f's{i}.png') for i in range(n_show)]),
        ('Target LoRA v1\n(1000 steps)', [load_rgb(Path(v1_dir) / 'images' / 'target_off' / f's{i}.png') for i in range(n_show)]),
        ('Target LoRA v2\n(2000 steps, selected)', [load_rgb(Path(v2_dir) / 'images' / 'target_off' / f's{i}.png') for i in range(n_show)])]
    for r, (label, images) in enumerate(sources):
        top = height - 0.45 - r * (tile + gap)
        fig.text(0.05 / width, (top - tile / 2) / height, label, fontsize=8.5, fontweight='bold', va='center')
        for c, image in enumerate(images):
            ax = fig.add_axes([(label_w + c * (tile + gap)) / width, (top - tile) / height, tile / width, tile / height])
            ax.imshow(image)
            ax.set_xticks([])
            ax.set_yticks([])
    fig.text(label_w / width, (height - 0.1) / height,
             'Real column: 8 fixed C1 validation frames (not matched to generated seeds). Generated columns: seeds 77000..77007, '
             'controller off, same seeds for all three adapters.', fontsize=7.5, va='top')
    tab_ax = fig.add_axes([0.02, 0.75 / height, 0.96, (table_h - 0.1) / height])
    tab_ax.axis('off')
    header_row, body = rows_table
    table = tab_ax.table(cellText=body, colLabels=header_row, loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(7.5)
    table.scale(1, 1.35)
    for (row, col), cell in table.get_celld().items():
        cell.set_linewidth(0.3)
        if row == 0:
            cell.set_text_props(fontweight='bold')
            cell.set_facecolor('#eeeeee')
    fig._geom = (width, height)
    _caption(fig, 'Figure 4. Appearance check (RGB only; C1 depth is never used). Metrics come from the saved review json '
             'files (64 unconditional images per adapter, one seed set); W1 = mean RGB Wasserstein-1 over channels, lower is '
             'closer; diversity = mean pairwise L1 at 64x64; NN ratio = median generated-to-own-training-frame distance over '
             'real held-out distance (>= 1 argues against memorisation). Generic image statistics are an imperfect realism '
             'measure; v2 reduces but does not remove the collapse towards a centred dark lumen.',
             y_in=0.08, wrap=int(width * 16), size=7.5)
    return _save(fig, path)


def pipeline_figure(stages, path):
    """Implemented-pipeline diagram. stages: dict name -> {'text', 'kind': trained|frozen|data|artifact,
    'status': 'run'|'partial'|'not run', 'detail': str}; positions are fixed here."""
    plt = _mpl()
    from matplotlib.lines import Line2D
    from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Patch

    fig, ax = plt.subplots(figsize=(16, 9.2), facecolor='white')
    fig.subplots_adjust(0.01, 0.01, 0.99, 0.99)
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 9.2)
    ax.axis('off')
    colors = {'trained': '#cfe3f7', 'frozen': '#e6e6e6', 'data': '#fbe9c6', 'artifact': '#d9f0d3'}
    # name: (x, y, w, h)
    layout = {
        'data_source': (0.3, 4.8, 2.9, 1.9), 'data_c1': (0.3, 1.9, 2.9, 1.9),
        'base': (3.8, 7.55, 2.9, 0.8), 'lora_source': (3.8, 4.8, 2.9, 1.9), 'lora_target': (3.8, 1.9, 2.9, 1.9),
        'controller': (7.5, 4.8, 2.9, 1.9), 'swap': (7.5, 1.9, 2.9, 1.9),
        'generated': (11.0, 1.9, 2.5, 1.9), 'da3': (11.0, 4.8, 2.5, 2.3), 'c2': (13.9, 4.8, 1.9, 2.3)}
    status_text = {'run': 'status: run', 'partial': 'status: in progress / partly run', 'not run': 'status: not run'}
    boxes = {}
    for name, (x, y, w, h) in layout.items():
        stage = stages[name]
        run = stage['status']
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.03,rounding_size=0.12',
                                    fc=colors[stage['kind']], ec='black' if run != 'not run' else '#777777',
                                    lw=1.7 if run == 'run' else 1.2,
                                    ls={'run': 'solid', 'not run': 'dashed', 'partial': 'dashdot'}[run]))
        n_title = stage['text'].count('\n') + 1
        ax.text(x + w / 2, y + h - 0.12, stage['text'], ha='center', va='top', fontsize=9.5, fontweight='bold',
                linespacing=1.15)
        if stage['detail']:
            ax.text(x + w / 2, y + h - 0.2 - 0.26 * n_title - 0.08, stage['detail'], ha='center', va='top',
                    fontsize=7.6, color='#222222', linespacing=1.25)
        if name != 'base':
            ax.text(x + w / 2, y + 0.07, status_text[run], ha='center', va='bottom', fontsize=7, style='italic',
                    color='#444444')
        boxes[name] = (x, y, w, h)

    def pt(name, side, frac=0.5):
        x, y, w, h = boxes[name]
        return {'r': (x + w, y + h * frac), 'l': (x, y + h * frac), 't': (x + w * frac, y + h),
                'b': (x + w * frac, y)}[side]

    def arrow(a, b, a_side='r', b_side='l', a_frac=0.5, b_frac=0.5, ls='-'):
        ax.add_patch(FancyArrowPatch(pt(a, a_side, a_frac), pt(b, b_side, b_frac), arrowstyle='-|>',
                                     mutation_scale=14, lw=1.4, color='black', ls=ls, shrinkA=0, shrinkB=0))

    def polyline(points, ls='-', color='black'):
        xs, ys = zip(*points[:-1])
        ax.plot(xs + (points[-1][0],), ys + (points[-1][1],), color=color, lw=1.4, ls=ls)
        ax.add_patch(FancyArrowPatch(points[-2], points[-1], arrowstyle='-|>', mutation_scale=14, lw=1.4,
                                     color=color, shrinkA=0, shrinkB=0))

    arrow('data_source', 'lora_source')
    arrow('data_c1', 'lora_target', a_frac=0.3, b_frac=0.3)
    arrow('base', 'lora_source', 'b', 't')
    polyline([(3.8, 7.95), (3.5, 7.95), (3.5, 3.4), (3.8, 3.4)])
    # source RGB + depth also trains the controller (dashed route above the source LoRA)
    polyline([(1.75, 6.7), (1.75, 7.2), (8.95, 7.2), (8.95, 6.7)], ls='--', color='#555555')
    ax.text(5.9, 7.27, 'source RGB + depth pairs (controller training)', fontsize=7.2, style='italic', color='#555555',
            ha='center', va='bottom')
    arrow('lora_source', 'controller')
    arrow('controller', 'swap', 'b', 't')
    arrow('lora_target', 'swap')
    arrow('swap', 'generated')
    arrow('generated', 'da3', 't', 'b')
    arrow('da3', 'c2')
    ax.text(5.25, 8.55, 'both LoRAs start independently from this one identical base', ha='center', fontsize=7.8,
            style='italic')
    ax.text(8.95, 1.55, 'swap the appearance LoRA only; controller weights unchanged', ha='center', fontsize=7.6,
            style='italic')
    ax.text(0.3, 9.05, 'Implemented pipeline: what is trained, what is frozen, and what has actually been run',
            fontsize=12, fontweight='bold', va='top')
    ax.legend(handles=[Patch(fc=colors['trained'], ec='black', label='trained here'),
                       Patch(fc=colors['frozen'], ec='black', label='frozen / pretrained / evaluation'),
                       Patch(fc=colors['data'], ec='black', label='data (read-only)'),
                       Patch(fc=colors['artifact'], ec='black', label='generated artifact'),
                       Line2D([], [], color='black', lw=1.7, label='solid border: run'),
                       Line2D([], [], color='black', lw=1.2, ls='dashdot', label='dash-dot: in progress / partly run'),
                       Line2D([], [], color='#777777', lw=1.2, ls='dashed', label='dashed: not run')],
              loc='lower left', ncol=4, fontsize=8, frameon=False, bbox_to_anchor=(0.0, 0.0))
    return _save(fig, path)
