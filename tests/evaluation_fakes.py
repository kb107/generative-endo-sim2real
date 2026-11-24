"""Generated fixtures for the registry and locked C2 campaign: a tiny repository, runs and fake C2 data."""
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from generative_sim2real.depth import model as depth_model
from generative_sim2real.data.rgb_modes import GENERATED_PROVENANCE
from generative_sim2real.depth.checkpoints import head_digest
from generative_sim2real.evaluation import registry as reg
from da3_fakes import fake_bb_model, fake_model, intrinsics

HW = (56, 70)
SEQUENCES = {'c2_a_v1': 3, 'c2_b_v1': 2, 'c2_c_v2': 4}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def put(path, text):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    return path


def write_manifest(root, name, split, rows, **extra):
    path = put(root / f'outputs/manifests/{name}.jsonl', ''.join(json.dumps(r, sort_keys=True) + '\n' for r in rows))
    sequences = {}
    for r in rows:
        sequences[r['sequence_id']] = sequences.get(r['sequence_id'], 0) + 1
    side = {'schema_version': 1, 'split': split, 'count': len(rows), 'manifest_sha256': sha(path),
            'per_sequence': dict(sorted(sequences.items())), 'config_sha256': 'c' * 64, 'source_code_sha256': {'x.py': 'd' * 64},
            **extra}
    put(path.with_suffix('.meta.json'), json.dumps(side))
    return path


def perturbed_head_state(seed):
    model = fake_model()
    with torch.no_grad():
        model.network.head.scratch.depth_conv.bias.fill_(-1. + .1 * seed)
    return {k: v.detach().clone() for k, v in model.head_state_dict().items()}


def perturbed_lora_state(seed):
    model = fake_bb_model()
    model.attach_backbone_lora(depth_model.LOCKED_BACKBONE_LORA, seed=seed)
    generator = torch.Generator().manual_seed(100 + seed)
    with torch.no_grad():
        for name, value in model.backbone_lora.named_parameters():
            if name.endswith('lora_B'):
                value.copy_(torch.randn(value.shape, generator=generator) * .05)
    return {k: v.detach().clone() for k, v in model.backbone_lora_state_dict().items()}


class FakeTarget:
    """Duck-typed ManifestDataset for fake C2 rows; counts decoded frames."""
    split, geometry = 'target_test', 'target_rectified_full'

    def __init__(self, rows, hw=HW):
        self.rows = rows
        self.hw = hw
        self.decoded = 0

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        self.decoded += 1
        h, w = self.hw
        rng = np.random.default_rng(77 + index)
        rgb = torch.from_numpy(rng.uniform(.05, .95, (3, h, w)).astype(np.float32))
        depth = (.02 + .03 * np.linspace(0, 1, h, dtype=np.float32)[:, None] * np.ones((1, w), np.float32))[None]
        valid = np.ones((1, h, w), bool)
        valid[:, :, : w // 5] = False
        return {'sample_id': self.rows[index]['sample_id'], 'rgb': rgb,
                'camera': {'K': intrinsics(h).clone(), 'T_world_from_camera': None, 'distortion': None, 'model': 'pinhole'},
                'targets': {'depth_m': torch.from_numpy(np.where(valid, depth, 0.).astype(np.float32)),
                            'depth_valid': torch.from_numpy(valid), 'sparse_depth': None, 'instances': None},
                'meta': {'sequence_id': self.rows[index]['sequence_id']}}


def make_run(root, name, seed, *, ids, source_val_count, mutate=None, pre=None, official_head=None, mode=None,
             run_name=None, lora=False, train_seed=0):
    """A finished, source-validation-selected run directory with real saved best.pt and last.pt files.

    ``name`` is the method name (and the default RGB mode); ``mode``, ``run_name``, ``lora`` (bbLoRA variant) and
    ``train_seed`` allow arms beyond the original R/A/G/M runs.
    """
    mode = mode or name
    run = root / 'outputs/runs' / (run_name or f'da3_{name}_seed0')
    config = json.loads(json.dumps({
        'schema_version': 1, 'purpose': 'experiment',
        'model': {'da3_revision': depth_model.DA3_REVISION, 'checkpoint': depth_model.CHECKPOINT,
                  'checkpoint_revision': depth_model.CHECKPOINT_REVISION, 'process_hw': [504, 504],
                  'trainable': 'head', 'canonical_focal_px': 300.0},
        'data': {'source_train': 'outputs/manifests/source_train.jsonl',
                 'source_validation': 'outputs/manifests/source_validation.jsonl',
                 'matched_ids': 'outputs/manifests/downstream_ids.json', 'mode': mode,
                 'generated_manifest': f'outputs/manifests/generated_{mode}.jsonl' if mode in ('G', 'M') else None,
                 'outputs_root': 'outputs', 'augmentation': {'brightness': [.9, 1.1], 'color': [.9, 1.1], 'gamma': [.9, 1.1]}},
        'training': {'seed': train_seed, 'optimizer_steps': 4, 'global_batch_size': 2, 'microbatch_size': 2, 'learning_rate': 5e-5,
                     'weight_decay': .01, 'warmup_steps': 1, 'gradient_clip_norm': 1., 'validation_every': 2,
                     'checkpoint_every': 2, 'depth_weight': 1., 'gradient_weight': 1., 'precision': 'float32',
                     'optimizer': 'AdamW', 'schedule': 'warmup_cosine', 'world_size': 1},
        'evaluation': {'selection': reg.SELECTION, 'prediction_resize': 'bilinear_align_corners_false',
                       'target_views': [], 'validation_frames': None}}))
    if lora:
        config['model'].update(trainable='head+backbone_lora', backbone_lora=json.loads(json.dumps(depth_model.LOCKED_BACKBONE_LORA)))
    files = {p: sha(root / p) for p in ('outputs/manifests/source_train.jsonl', 'outputs/manifests/source_train.meta.json',
                                        'outputs/manifests/source_validation.jsonl', 'outputs/manifests/source_validation.meta.json',
                                        'outputs/manifests/downstream_ids.json')}
    if config['data']['generated_manifest']:
        files[config['data']['generated_manifest']] = sha(root / config['data']['generated_manifest'])
    code = {k: sha(root / 'src/generative_sim2real' / k) for k in ('data/a.py', 'depth/b.py', 'evaluation/c.py')}
    resolved = {'config': config, 'files_sha256': files, 'code_sha256': code, 'environment': {'python': '3.11'},
                'smoke_only': False, 'sampler': 's', 'rgb_rng': 'r', 'loss_reduction': 'l'}
    if pre:
        pre(resolved)
    state = head_state = perturbed_head_state(seed)
    digest = head_digest(head_state)
    reports = {0: .5, 2: .2 - .01 * seed, 4: .3}            # best (strict first argmin) at step 2
    best_step = 2
    run.mkdir(parents=True, exist_ok=True)
    best = {'format_version': 1, 'kind': 'da3_head', 'head_state_dict': state, 'head_sha256': digest,
            'selected_step': best_step, 'source_validation_sequence_macro_abs_rel': reports[best_step],
            'compatibility': resolved}
    lora_state = perturbed_lora_state(seed) if lora else None
    if lora:
        best.update(kind='da3_head_bblora', backbone_lora_state_dict=lora_state, backbone_lora_sha256=head_digest(lora_state),
                    backbone_lora_config=json.loads(json.dumps(depth_model.LOCKED_BACKBONE_LORA)))
    torch.save(best, run / 'best.pt')
    for step, value in reports.items():
        put(run / f'validation/step_{step:06d}/summary.json', json.dumps({
            'metadata': {'smoke_only': False, 'split': 'source_validation', 'native_labels': True, 'selection': reg.SELECTION,
                         'validation_frames': None},
            'frame_count': source_val_count, 'sequence_macro': {'abs_rel': value}}))
    put(run / 'train.jsonl', ''.join(json.dumps({'step': s, 'sample_ids': ids[:2]}) + '\n' for s in range(1, 5)))
    status = {'state': {'optimizer_step': 4, 'cursor': 8, 'best_abs_rel': reports[best_step], 'best_step': best_step,
                        'best_head_sha256': digest,
                        'initial_head_sha256': official_head or reg.official_digests(fake_model())['official_head_sha256']}, 'target_steps': 4, 'completed_steps': 4,
              'selected_step': best_step, 'smoke_only': False, 'device': 'cpu'}
    if lora:
        status['state'].update(best_backbone_lora_sha256=head_digest(lora_state))
    if mutate:
        mutate(resolved, status)
    final_head = perturbed_head_state(seed + 10)          # last.pt: the head at the final step (differs from best)
    last = {'format_version': 1, 'kind': 'da3_training', 'head_state_dict': final_head, 'optimizer': {}, 'scheduler': {},
            'state': json.loads(json.dumps(status['state'])), 'rng': {}, 'compatibility': resolved}
    if lora:
        last['backbone_lora_state_dict'] = perturbed_lora_state(seed + 10)
    torch.save(last, run / 'last.pt')
    put(run / 'resolved.json', json.dumps(resolved))
    put(run / 'status.json', json.dumps(status))
    return run


class World:
    def __init__(self, root, methods=('R', 'A'), monkeypatch=None, seeds=None, model_fn=fake_model):
        self.model_fn = model_fn
        self.root = root
        self.data_root = root / 'data'
        monkeypatch.setenv('DATA_ROOT', str(self.data_root))
        monkeypatch.setattr(reg, 'TARGET_HW', list(HW))
        for rel, text in (('src/generative_sim2real/__init__.py', ''), ('src/generative_sim2real/data/a.py', 'A = 1\n'),
                          ('src/generative_sim2real/depth/b.py', 'B = 1\n'), ('src/generative_sim2real/evaluation/c.py', 'C = 1\n')):
            put(root / rel, text)
        train_rows = [{'sample_id': f'sim/t/{i:03d}', 'sequence_id': 'sim/t', 'split': 'source_train'} for i in range(6)]
        val_rows = [{'sample_id': f'sim/v/{i:03d}', 'sequence_id': 'sim/v', 'split': 'source_validation'} for i in range(2)]
        train = write_manifest(root, 'source_train', 'source_train', train_rows)
        write_manifest(root, 'source_validation', 'source_validation', val_rows)
        self.ids = [r['sample_id'] for r in train_rows[:4]]
        put(root / 'outputs/manifests/downstream_ids.json', json.dumps({
            'schema_version': 1, 'purpose': 'experiment', 'frozen': True, 'source_manifest_sha256': sha(train),
            'source_sidecar_sha256': sha(train.with_suffix('.meta.json')), 'sample_ids': self.ids}))
        for name in methods:
            if name in 'GM':
                put(root / f'outputs/manifests/generated_{name}.jsonl', ''.join(json.dumps({
                    'sample_id': i, **{k: (n if k == 'seed' else f'{k}-value') for k in GENERATED_PROVENANCE},
                    'size_hw': [512, 512]}) + '\n' for n, i in enumerate(self.ids)))
        self.rows = [{'sample_id': f'c3vdv2/c2/{seq}/{i:04d}', 'sequence_id': seq, 'split': 'target_test',
                      'transform_id': reg.TARGET_GEOMETRY} for seq, n in SEQUENCES.items() for i in range(n)]
        poses = {}
        for seq in SEQUENCES:
            key = f'C3VDv2/frames/C2/{seq}/pose.txt'
            poses[key] = sha(put(self.data_root / key, f'pose {seq}\n'))
        write_manifest(root, 'target_test', 'target_test', self.rows, governing_files_sha256=poses)
        for name in methods:
            make_run(root, name, (seeds or {}).get(name, 'RAGM'.index(name) + 1), ids=self.ids, source_val_count=len(val_rows))
        self.spec = {'schema_version': 1, 'name': 'fixture', 'output': 'outputs/c2_campaign/registry.json',
                     'methods': {'Z': None, **{m: f'outputs/runs/da3_{m}_seed0' for m in methods}},
                     'data': {'source_train': 'outputs/manifests/source_train.jsonl',
                              'source_validation': 'outputs/manifests/source_validation.jsonl',
                              'downstream_ids': 'outputs/manifests/downstream_ids.json',
                              'target_test': 'outputs/manifests/target_test.jsonl'}}
        self.registry_path = root / self.spec['output']
        self.campaign = root / 'outputs/c2_campaign'
        self.ledger = self.campaign / 'ledger.jsonl'
        self.digests = {**reg.official_digests(model_fn()), 'weights_file_sha256': 'w' * 64}
        monkeypatch.setattr(reg, 'official_digests_from_files', lambda: dict(self.digests))
        self.dataset = None

    def build(self):
        return reg.build_registry(self.spec, self.root, self.registry_path, digests=self.digests, code_root=self.root)

    def freeze(self):
        self.build()
        return reg.freeze_registry(self.registry_path, self.root, code_root=self.root)

    def dataset_factory(self, registry, root):
        self.dataset = FakeTarget(self.rows)
        return self.dataset

    def model_factory(self):
        return self.model_fn()
