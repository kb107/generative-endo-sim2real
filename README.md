# Generative sim-to-real for endoscopic depth

Depth-controlled appearance adaptation for colonoscopy: generate phantom-looking images from simulator depth, then test whether those pairs improve metric depth on a held-out phantom.

**Idea.** A frozen PixArt-α DiT carries two appearance LoRAs (source and target) and one depth controller (a low-rank "Control-LoRA") trained under the source LoRA. At generation time only the appearance LoRA is swapped, so the same depth map yields an image with the target (phantom) look. A DA3-Metric depth head is then fine-tuned on the generated pairs and evaluated on held-out C3VDv2 C2 phantom sequences. Controls isolate what carries the gain: resynthesis with the source look, Reinhard colour transfer, Fourier Domain Adaptation (FDA), photometric augmentation, a SimuScope-style baseline, and a closed image editor (Nano Banana 2.1).

| | Data | Use |
|---|---|---|
| Source | SimCol3D (RGB + metric depth) | controller training, source LoRA, depth labels |
| Target look | C3VDv2 C1 phantom sequences (RGB only) | target LoRA, colour-transfer references |
| Evaluation | C3VDv2 C2 phantom sequences (2,589 frames, 34 sequences, 18 families) | locked test set, never used for training, generation or selection |

## Results

C2 held-out phantom, raw unaligned metric depth, sequence-macro. Training seed 0 unless two values are shown (seed 0 / seed 1). The noise level on AbsRel (largest seed or generation-seed gap) is 0.027.

| depth head trained on | AbsRel | RMSE (mm) |
|---|---|---|
| pretrained DA3-Metric-Large (no fine-tuning) | 13.879 | 309.2 |
| raw SimCol | 0.750 / 0.758 | 14.9 |
| **generated phantom look (ours)** | **0.367** / 0.376 | **9.6** |
| SimCol-look resynthesis (same generator, source LoRA) | 0.542 | 11.2 |
| Reinhard colour transfer to C1 frames | 1.338 | 26.3 |
| FDA, beta 0.01 (canonical) / 0.05 | 1.217 / 0.530 | 23.7 / 11.3 |
| photometric augmentation, mild / strong | 0.962 / 0.816 | 18.4 / 15.7 |

The closed editor (Nano Banana 2.1, given the SimCol frame, two C1 frames and a prompt) is further from the C1 phantom look than ours (DINOv2 FD 1297 vs 1135, lower is closer) and on some frames invents folds or moves the lumen, so its images no longer match the simulator depth labels.

## Setup

Python 3.11, [uv](https://docs.astral.sh/uv/), a CUDA GPU for generation and training.

```bash
uv sync --locked
cp .env.example .env            # set DATA_ROOT to the folder holding SimCol3D/ and C3VDv2/
bash scripts/setup_generator.sh # generator environment under external/
bash scripts/setup_da3.sh       # depth environment under external/
```

Datasets are read-only inputs and are not part of this repository. The optional closed-editor comparison also needs `external/foundation-venv` with `google-genai` and `pillow`, and `GEMINI_API_KEY` in `.env`.

## Pipeline

Run from the repository root. Each script lists its settings at the top. Tests (CPU, generated fixtures only): `PYTHONPATH=src .venv/bin/python -m pytest -q`.

```bash
# 1. Data and generator
bash scripts/prepare_data.sh                      # splits, manifests, depth encoding
CONFIG=configs/p4_lowcap.yaml RUN=source_lora bash scripts/train_p4.sh
CONFIG=configs/p4_lowcap.yaml RUN=target_lora_v2 bash scripts/train_p4.sh
CONFIG=configs/p4_lowcap.yaml RUN=controller_lowrank_long bash scripts/train_p4.sh
CONFIG=configs/p4_lowcap.yaml RUN=controller_lowrank_long STEP=3000 SET=validation bash scripts/gate_p4.sh   # source-controller gate
TARGET_RUN=target_lora_v2 TARGET_STEP=2000 bash scripts/appearance_review_p4.sh    # target-look review against C1
CONFIG=configs/sweeps/p5_lc3000.yaml OUT=outputs/p5-diagnostics/matrix_lc3000_seed17 bash scripts/p5_matrix.sh   # LoRA-swap matrix
bash scripts/generate_downstream.sh               # one image per frozen ID; needs a recipe JSON, see script header

# 2. Depth heads (R raw, G generated, M 50:50 mix, A photometric; SEED and CONFIG select seed and variant)
MODE=R bash scripts/train_da3.sh
MODE=G GENERATED_MANIFEST=<generated_manifest.jsonl> bash scripts/train_da3.sh

# 3. Controls: one fixed image per ID without the generator
bash scripts/build_variant_set.sh                 # T_reinhard, A_strong, F_fda_b01, F_fda_b05
MODE=G GENERATED_MANIFEST=outputs/generated_variants/T_reinhard/generated_manifest.jsonl RUN_NAME=da3_T_seed0 bash scripts/train_da3.sh
bash scripts/simuscope.sh                         # SimuScope-style baseline

# 4. Locked C2 evaluation (final campaign; registries for 6A and 6B are configs/registry_c6a.yaml and registry_c6b.yaml)
SPEC=configs/registry_c5.yaml bash scripts/build_registry.sh
SPEC=configs/registry_c5.yaml ACTION=freeze bash scripts/build_registry.sh
REGISTRY=outputs/c2_campaign/registry_c5.json DRY_RUN=0 bash scripts/eval_c2.sh
PYTHONPATH=src python -m generative_sim2real.analysis.campaign --campaign outputs/c2_campaign/<registry hash, 12 chars> --output outputs/analysis/campaign5

# 5. Why the look moves metric scale (source labels only)
bash scripts/look_scale_probe.sh                  # look -> scale probe on the 32 source-validation conditions
bash scripts/scale_anatomy.sh                     # per-head scale anatomy and frozen predictions
PYTHONPATH=src python -m generative_sim2real.analysis.prereg_c6 --c5 <campaign 5 dir> [--c6a <dir>] [--c6b <dir>] --out outputs/analysis/prereg_c6

# 6. Closed-editor comparison (Nano Banana 2.1; billed calls need EXECUTE=1)
bash scripts/foundation_inputs.sh                 # fixed inputs
SET=subset300 MAX_REQUESTS=300 EXECUTE=1 bash scripts/foundation_gemini.sh
bash scripts/foundation_geometry.sh               # registration and depth-structure proxy
bash scripts/foundation_fd300.sh                  # FD / KID against C1 at matched n
```

`configs/p4.yaml` is the full-controller variant (the "G_full" arm); `configs/da3_bblora.yaml` adds LoRA on the last four backbone blocks. A frozen registry is never overwritten, and a second campaign needs `DISCLOSE_NEW_EXPERIMENT=<reason>`. Campaign 6A reads its output scales from `outputs/analysis/scale_anatomy/output_scales.json` (`scale = exp(-c_G)` per head, derived from `predictions.json`; the file records its own definition).

## Layout

- `src/generative_sim2real/`: data adapters, generation (LoRAs, controller), depth training, locked evaluation, baselines, variants (colour transfer, FDA), analysis, foundation-editor comparison
- `configs/`: experiment settings, registries and pinned requirements
- `scripts/`: launchers for the pipeline above
- `tests/`: CPU tests on generated fixtures

No license has been chosen yet. Code adapted from other projects keeps its own license (`src/generative_sim2real/generation/UPSTREAM_LICENSE.txt`).
