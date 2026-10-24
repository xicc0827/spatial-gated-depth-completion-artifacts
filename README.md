# Spatial-gated depth-completion adaptation: reproducibility artifacts

This repository contains the frozen evaluation records, protocol files, reaggregation code, sampling manifests, checkpoint identities, and figure statistics used in the manuscript **Fixed-observation-budget spatial-gated depth-completion adaptation: cross-scene evaluation and regional error decomposition**.

## What is included

- `resources/or1`: DIODE and OMNI-DC frozen-result summaries and audit records.
- `resources/or2`: DIODE per-image regional diagnostics and aggregate tables.
- `resources/or3`: ARKitScenes frozen protocol, per-image scores, regional records, and venue summaries.
- `training_snapshot`: source-domain sampling manifests, loss calibration records, the compact adapter/core snapshot, and a machine-readable parameter/update/latency accounting record.
- `scripts`: portable gate, metric, reaggregation, integrity-check, and repository-manifest scripts.
- `manifests/checkpoint_hashes.csv`: model identity and SHA-256 values. Model weights are not redistributed.
- `manifests/or1_sha256.txt`, `or2_sha256.txt`, and `or3_sha256.txt`: file-level integrity records for the three online resources.
- `figures`: statistics and source code for the three quantitative core figures, plus the machine-readable selection record for the qualitative success/failure figure. Third-party RGB/depth pixels remain excluded.

## Quick verification

```bash
python -m pip install -r requirements.txt
python scripts/verify_artifacts.py
python scripts/reaggregate.py
```

The reaggregation command regenerates `derived/principal_rmse_reaggregated.csv` from the published per-image records. It does not need the raw datasets or model weights.

## Scope and exclusions

Raw NYUv2, DIODE, and ARKitScenes data are not redistributed. Ten diagnostic NPZ files containing DIODE RGB/depth pixels were removed from the public package; their original filenames and SHA-256 values remain in the corresponding OR2 marker records. Upstream CompletionFormer and OMNI-DC weights are also excluded. The repository records their identities and gives acquisition references instead.

The frozen protocol JSON files preserve the original scientific decisions. Machine-specific paths were replaced with environment placeholders for public release, so their original sealed-file hashes are provenance values rather than hashes of the sanitized copies. `manifests/repository_sha256.txt` is the integrity record for this public tree.

## Re-running model inference

Full inference requires locally acquired datasets and upstream model repositories/weights. Set the local roots described in `REPRODUCTION.md`, then use the scripts in `training_snapshot` and the protocol files under each online resource. The portable reaggregation path above reproduces the reported aggregate statistics directly from the released records.

## License and third-party material

Original code, documentation, and derived research artifacts in this repository are released under the MIT License in `LICENSE`. This root license does not relicense third-party datasets, upstream projects, model weights, or the ARKitScenes source snapshot under `resources/or3/source`; those materials remain subject to their original terms. See `THIRD_PARTY_NOTICES.md`, `resources/or3/source/LICENSE`, `training_snapshot/CompletionFormer_LICENSE`, and `training_snapshot/OMNI-DC_LICENSE`.

## Versioned preservation

The existing `v1.0-submission` tag is retained as an immutable record of the first public artifact snapshot. Licensing and post-tag documentation updates will be preserved in a new `v1.0.1-submission` release rather than by moving the existing tag. `ZENODO_RELEASE_CHECKLIST.md` records the remaining metadata and release steps. The repository will display the Zenodo concept DOI and version DOI after the release has been archived; no DOI has been minted yet.
