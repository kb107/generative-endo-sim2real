"""Read-only access to one saved swap-matrix render directory (P5)."""
from pathlib import Path

import numpy as np

from .common import load_rgb, read_json


class MatrixRun:
    """`directory` holds render.json, matrix.json, refs.pt and images/<arm>/c<i>.png."""

    def __init__(self, directory):
        self.directory = Path(directory)
        self.render = read_json(self.directory / 'render.json')
        self.matrix = read_json(self.directory / 'matrix.json')
        self.ids = self.render['ids']
        self.partner = self.render['partner']
        self.seed = self.render['seed']

    def image(self, arm, condition):
        return load_rgb(self.directory / 'images' / arm / f'c{condition}.png')

    def median_depth_mm(self):
        """Median valid depth per condition in millimetres, from the saved reference tensors."""
        import torch

        refs = torch.load(self.directory / 'refs.pt', map_location='cpu', weights_only=True)
        depth, valid = refs['depth_m'], refs['valid']
        out = []
        for i in range(depth.shape[0]):
            values = depth[i][valid[i] & (depth[i] > 0)]
            out.append(float(values.median().item()) * 1000.0)
        return np.asarray(out)

    def m1_gains(self, adapter):
        """Per-condition M1 gain (AbsRel off minus correct) for one adapter in this directory's seed."""
        return self.matrix['primary_abs_rel']['per_adapter'][adapter]['checks']['M1_correct_beats_off']['gains']
