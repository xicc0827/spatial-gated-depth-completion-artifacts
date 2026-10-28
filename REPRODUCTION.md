# Reproduction guide

Run the commands below from the repository root. Public statistical reaggregation and model-level replay have different input requirements: the former uses released records; the latter additionally needs local files that are not distributed here.

## Level 1: integrity and statistical reaggregation

Install the packages in `requirements.txt`, then run:

```bash
python -m pip install -r requirements.txt
python scripts/verify_artifacts.py
python scripts/reaggregate.py
```

The verifier checks every file listed in `manifests/repository_sha256.txt`. Reaggregation writes `derived/principal_rmse_reaggregated.csv`, recomputing the principal scene-balanced DIODE and site-balanced ARKitScenes RMSE values from the public per-image records. Neither command requires raw datasets, model weights, or a GPU.

The per-resource integrity records are `manifests/or1_sha256.txt`, `manifests/or2_sha256.txt`, `manifests/or3_sha256.txt`, and `manifests/or4_sha256.txt`; the manuscript supplement reports the SHA-256 of each record file. OR1--OR4 are repository directory identifiers, distinct from the journal's Online Resource numbering.

For all six fixed-checkpoint-intervention endpoints, paired confidence intervals, physical-unit coverage, and seed effects, run:

```bash
python resources/or4/analysis/aggregate_validate_publish.py aggregate
```

This command uses the published OR4 records. It overwrites the generated summaries and OR4 README; use a disposable clone when checking an immutable snapshot, and verify file integrity before reaggregation.

## Level 2: regional summaries and figures

Run `python figures/build_core_figures.py` to regenerate the framework diagram, two quantitative core figures, and their bootstrap interval table under `derived/figures`. The frozen regional tables are stored under `resources/or2/summary` and `resources/or3/summary`. The qualitative case-selection record is public, but the pixel-containing NPZ files needed to render the success/failure panels are not distributed.

## Level 3: single-condition CompletionFormer inference and replay

### Required local inputs and access status

The portable entry point is `scripts/replay_completionformer.py`. It requires:

1. The [CompletionFormer upstream source](https://github.com/youmi-zym/CompletionFormer) at commit `2744eddee9b57595dc3064f7d342569736a6803b`, with the original `src/model` files. The port validates five source-file hashes before loading them.
2. The official CompletionFormer NYUv2 Parent checkpoint, whose identity is recorded under `parent` in `manifests/checkpoint_hashes.csv`.
3. One source-trained Spatial checkpoint for seed 42, 43, or 44. Supplying the separately trained Image Mean checkpoint for the same seed adds that comparison.
4. One frozen converted DIODE or ARKitScenes NPZ matching its published manifest entry.

The six Spatial/Image Mean adapter checkpoint files are **not included and are not currently downloadable from this repository**. Converted pixel-containing inputs and upstream weights are also absent. Model-level replay requires obtaining these files separately; hashes identify the files but do not supply them. Acquiring upstream weights alone is insufficient to run the adapters. Dataset and upstream-project acquisition references are listed in `THIRD_PARTY_NOTICES.md`; they do not provide the unpublished adapter files.

The CLI reads the converted NPZ, not raw dataset directories. Users generating their own converted inputs must first acquire and preprocess the source data under the applicable terms and frozen protocol. The CLI is an executable inference path for users who possess the required inputs; it is not a raw-data converter, dataset downloader, training launcher, or OMNI-DC runner.

### Environment

The locally checked environment uses Python 3.12, PyTorch `2.7.0+cu128`, torchvision `0.22.0+cu128`, and timm `0.6.13`. `requirements-inference.txt` pins the PyTorch/torchvision/timm release versions and includes the statistical requirements.

For the tested CUDA 12.8 wheel family on Linux or Windows, install the matched PyTorch pair first, then the remaining requirements:

```bash
python -m pip install torch==2.7.0 torchvision==0.22.0 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r requirements-inference.txt
```

Choose another compatible wheel family, if needed, using the [official PyTorch 2.7.0 installation instructions](https://pytorch.org/get-started/previous-versions/#v270). The CLI accepts `--device cpu`, but the reported local replay check used CUDA; cross-device numerical agreement has not been established. The CompletionFormer port uses torchvision's deformable convolution backend rather than requiring the original custom CUDA extension.

### Frozen input schema

The NPZ must contain the following arrays:

| Key | Shape and type | Meaning |
|---|---|---|
| `rgb` | `(228, 304, 3)`, `uint8` | Preprocessed RGB input |
| `depth` | `(228, 304)`, `float32` | Reference depth in metres |
| `valid` | `(228, 304)`, Boolean | Exactly `isfinite(depth) & (depth > 0.1) & (depth <= 10)` |
| `support_indices` | `(5, 11, 500)`, integer | Flattened pixel indices for each support draw and sampling pattern |

The pattern order is `uniform`, `left`, `right`, `top`, `bottom`, `center_hole`, `diag_main_half`, `diag_anti_half`, `center_vertical_band`, `center_disk`, `two_corners`. Each chosen support set must contain 500 distinct, in-bounds valid indices. The script constructs sparse depth from those coordinates and the reference values.

`--index` selects a retained `primary` image in `training_snapshot/11_diode_converted_manifest.json` for DIODE or `resources/or3/protocol/evaluation_manifest.json` for ARKitScenes. The input file's SHA-256 must match that entry. Matching array dimensions alone does not satisfy this check; this schema documents the frozen inputs and is not a substitute for the preprocessing procedure.

### Commands and outputs

The `local_inputs/...` paths below are examples of files supplied by the user, not paths downloaded or created by the repository. Commands are shown on one line for shell portability. A DIODE condition with all four inference modes is:

```bash
python scripts/replay_completionformer.py --dataset diode --index 0 --input local_inputs/diode/0000.npz --upstream-root local_inputs/CompletionFormer --parent-checkpoint local_inputs/NYUv2.pt --spatial-checkpoint local_inputs/spatial_seed42.pt --image-mean-checkpoint local_inputs/image_mean_seed42.pt --seed 42 --law top --repetition 0 --device cuda --output derived/replay_diode_0000_seed42.json
```

For ARKitScenes, the public original per-image score file can additionally check Parent and the independently trained branches:

```bash
python scripts/replay_completionformer.py --dataset arkitscenes --index 0 --input local_inputs/arkitscenes/0000.npz --upstream-root local_inputs/CompletionFormer --parent-checkpoint local_inputs/NYUv2.pt --spatial-checkpoint local_inputs/spatial_seed42.pt --image-mean-checkpoint local_inputs/image_mean_seed42.pt --seed 42 --law top --repetition 0 --device cuda --original-records resources/or3/per_image/0000.csv --output derived/replay_arkitscenes_0000_seed42.json
```

Omit `--image-mean-checkpoint` to run Parent, Spatial, and the fixed-checkpoint intervention only. Seeds 43 and 44 require their corresponding checkpoint files. `--law` accepts any of the 11 patterns and defaults to `top`; `--repetition` accepts 0--4 and defaults to 0. `--device` defaults to `cuda`. Use a separate CLI invocation for each condition, as the upstream port permits one model initialization per process.

The report contains RMSE, MAE, REL, and their unobserved-pixel counterparts for each requested mode, along with coordinate/checkpoint identities, environment versions, and replay comparisons. It exports no image pixels. The inference modes are:

| Output name | Corrector and gate |
|---|---|
| `parent` | Frozen Parent with no learned correction |
| `spatial` | Spatial checkpoint and spatial gate |
| `image_mean_intervention` | The same Spatial residual with its gate replaced by the image mean |
| `image_mean` | Optional, separately trained Image Mean checkpoint with an image-constant gate |

For unseen8 (all patterns except `uniform`, `left`, and `right`), the CLI automatically checks the Spatial and intervention metrics against `resources/or4/per_image/<dataset>/<index>.csv`. Optional `--original-records` adds checks against an original per-image score CSV with matching index, pattern, repetition, model names, and coordinate identity. The public ARKitScenes files support this option; a DIODE original-score file must be supplied separately if that additional comparison is wanted.

A metric discrepancy must be strictly below `1e-5`. The JSON status is `PASS` when all requested reference comparisons pass, `REPLAY_MISMATCH` when a comparison fails, or `INFERRED_NO_REFERENCE_CHECK` when no reference comparison applies. Identity, schema, or checkpoint mismatches stop execution before a successful report. An output file or successful inference alone does not establish agreement with published scores.

Local smoke checks reproduced all six metrics exactly for the first retained image of each cohort, `top`, repetition 0, seed 42, across Parent, Spatial, the fixed-checkpoint intervention, and the independently trained Image Mean branch. The reports are [DIODE replay](checks/replay_diode_seed42.json) and [ARKitScenes replay](checks/replay_arkitscenes_seed42.json). These checks cover two retained conditions at seed 42, not a fresh full-cohort inference run or the other two seeds, and do not provide a public download path for the required inputs.

## Training provenance and legacy execution snapshots

The source-domain exposure manifests are `training_snapshot/14_source_seed42.csv`, `16_source_seed43.csv`, and `18_source_seed44.csv`; loss calibration files are adjacent. Checkpoint identities are in `manifests/checkpoint_hashes.csv`.

The compact corrector/core files, training snapshots, manifests, and frozen protocols support inspection of the implementation and training exposure. They do not constitute a complete, tested retraining workflow from downloaded raw datasets. Legacy scripts under `training_snapshot` and `resources/*/analysis` may retain dependencies on the original execution stack. Their sanitized values such as `${MODEL_ROOT}`, `${TRAINING_ROOT}`, or `${DIODE_CONVERTED_ROOT}` are provenance placeholders, not a configuration interface that those scripts automatically expand. Setting similarly named environment variables alone does not make those scripts runnable.

The portable commands above are the documented public entry points. Original dataset licenses, upstream licenses, and the scope of the root MIT License are described in `THIRD_PARTY_NOTICES.md` and `LICENSE`.
