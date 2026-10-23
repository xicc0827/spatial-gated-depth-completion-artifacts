# Reproduction guide

## Level 1: integrity and statistical reaggregation

Install the packages in `requirements.txt`, then run:

```bash
python scripts/verify_artifacts.py
python scripts/reaggregate.py
```

This level verifies every released file and recomputes the principal scene/venue-equal RMSE values from the per-image records.

## Level 2: regional summaries and figures

Run `python figures/build_core_figures.py` to regenerate the three core figures and their bootstrap interval table under `derived/figures`. The frozen regional tables are stored under `resources/or2/summary` and `resources/or3/summary`. Scripts retained inside `resources/*/analysis` are provenance snapshots of the original inference and audit pipeline; model-level reruns require the Level 3 local inputs and environment placeholders.

## Level 3: training and inference

Acquire NYUv2, DIODE, and ARKitScenes from their official sources and accept their licenses. Acquire the official CompletionFormer NYUv2 checkpoint and the OMNI-DC repository/weights. Configure local paths through the placeholders used by the sanitized snapshot:

- `NYUV2_ROOT`
- `DIODE_FREEZE_ROOT`
- `DIODE_RAW_ROOT`
- `DIODE_CONVERTED_ROOT`
- `MODEL_ROOT`
- `TRAINING_ROOT`
- `OMNI_RESULTS_ROOT`

The source-domain exposure manifests are `training_snapshot/14_source_seed42.csv`, `16_source_seed43.csv`, and `18_source_seed44.csv`; loss calibration files are adjacent. Checkpoint identities are in `manifests/checkpoint_hashes.csv`.

The public release is optimized for independent verification of reported results. It deliberately omits raw data, third-party weights, and local execution logs.
