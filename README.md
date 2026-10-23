# Spatial-gated depth-completion adaptation: reproducibility artifacts

This repository contains the frozen evaluation records, protocol files, reaggregation code, sampling manifests, checkpoint identities, and figure statistics used in the manuscript **Fixed-observation-budget spatial-gated depth-completion adaptation: cross-scene evaluation and regional error decomposition**.

## What is included

- `resources/or1`: DIODE and OMNI-DC frozen-result summaries and audit records.
- `resources/or2`: DIODE per-image regional diagnostics and aggregate tables.
- `resources/or3`: ARKitScenes frozen protocol, per-image scores, regional records, and venue summaries.
- `training_snapshot`: source-domain sampling manifests, loss calibration records, and the compact adapter/core snapshot.
- `scripts`: portable gate, metric, reaggregation, and integrity-check scripts.
- `manifests/checkpoint_hashes.csv`: model identity and SHA-256 values. Model weights are not redistributed.
- `figures`: statistics and source code for the three core figures.

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

No blanket license is asserted for third-party datasets, upstream models, or their weights. See `THIRD_PARTY_NOTICES.md` and the license files retained with the ARKitScenes and CompletionFormer snapshots. Contact the repository owner before reusing original project code outside scholarly verification.
