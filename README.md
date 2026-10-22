# Generative sim-to-real for endoscopic depth

Work in progress. Depth-controlled appearance adaptation for colonoscopy: generate target-looking images from simulator depth, then test whether those pairs improve metric depth on a held-out phantom.

**Idea.** A frozen PixArt-α DiT carries two appearance LoRAs (source and target) and one depth controller trained under the source LoRA. At generation time only the appearance LoRA is swapped, so the same depth map yields a target-style image. A DA3-Metric depth head is then fine-tuned on the generated pairs and evaluated on held-out C3VDv2 C2 phantom sequences.

| | Data | Use |
|---|---|---|
| Source | SimCol3D (RGB + metric depth) | controller training, source LoRA, depth labels |
| Target look | C3VDv2 C1 (RGB only) | target LoRA |
| Evaluation | C3VDv2 C2 | locked test set, never used for training, generation or selection |

Compared depth models: pretrained (Z), raw SimCol (R), generated pairs (G), 50:50 mix (M). A re-implementation of the SimuScope recipe is included as a baseline.

## Setup

Python 3.11, [uv](https://docs.astral.sh/uv/), a CUDA GPU for generation and training.

```bash
uv sync --locked
cp .env.example .env            # set DATA_ROOT to the folder holding SimCol3D/ and C3VDv2/
bash scripts/setup_generator.sh # generator environment under external/
bash scripts/setup_da3.sh       # depth environment under external/
```

Datasets are read-only inputs and are not part of this repository.

## Pipeline

Run from the repository root. Each script lists its settings at the top.

```bash
bash scripts/prepare_data.sh                  # splits, manifests, depth encoding
RUN=source_lora bash scripts/train_p4.sh      # source / target LoRA and controller (see script for RUN values)
bash scripts/appearance_p4.sh                 # appearance checks
bash scripts/p5_matrix.sh                     # LoRA-swap diagnostics
bash scripts/generate_downstream.sh           # one image per frozen ID; needs a recipe, see script header
MODE=R bash scripts/train_da3.sh              # depth head: R raw, G generated, M mix
bash scripts/build_registry.sh                # freeze the C2 evaluation registry
DRY_RUN=0 bash scripts/eval_c2.sh             # locked C2 evaluation
bash scripts/simuscope.sh                     # SimuScope-style baseline
```

Training runs read `configs/p4.yaml` (full controller) by default; `CONFIG=configs/p4_lowcap.yaml` selects the low-rank "Control-LoRA" controller runs.

## Layout

- `src/generative_sim2real/`: data adapters, generation (LoRAs, controller), depth training, locked evaluation, baselines, reporting
- `configs/`: experiment settings and pinned requirements
- `scripts/`: launchers for the pipeline above

Results are preliminary (single seed). No license has been chosen yet.
