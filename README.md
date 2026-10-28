# Spatial-gated depth-completion adaptation: reproducibility artifacts

This repository contains the frozen evaluation records, protocol files, reaggregation code, sampling manifests, checkpoint identities, and figure statistics used in the manuscript **An Attribution-Aware Empirical Study of Spatially Gated Depth-Completion Adaptation under a Fixed Observation Budget**.

## What is included

- `resources/or1`: DIODE and OMNI-DC frozen-result summaries and audit records.
- `resources/or2`: DIODE per-image regional diagnostics and aggregate tables.
- `resources/or3`: ARKitScenes frozen protocol, per-image scores, regional records, and venue summaries.
- `resources/or4`: fixed-checkpoint gate intervention on all 304 DIODE and 247 ARKitScenes images, with paired records, physical-unit confidence intervals, coverage, seed effects, and integrity checks.
- `training_snapshot`: source-domain sampling manifests, loss calibration records, the compact adapter/core snapshot, and a machine-readable parameter/update/latency accounting record.
- `scripts`: portable gate, metric, reaggregation, integrity-check, and repository-manifest scripts, plus a single-condition CompletionFormer inference/replay entry point.
- `checks`: portable-entry-point replay reports for one retained condition from each CompletionFormer target cohort.
- `manifests/checkpoint_hashes.csv`: model identity and SHA-256 values. Model weights are not redistributed.
- `manifests/or1_sha256.txt`, `or2_sha256.txt`, `or3_sha256.txt`, and `or4_sha256.txt`: file-level integrity records for all four artifact directories. These directory names are distinct from the journal's Online Resource numbering.
- `figures`: statistics and source code for the framework diagram and two quantitative core figures, plus the machine-readable selection record for the qualitative success/failure figure. Third-party RGB/depth pixels remain excluded.

## Quick verification

```bash
python -m pip install -r requirements.txt
python scripts/verify_artifacts.py
python scripts/reaggregate.py
```

The reaggregation command regenerates `derived/principal_rmse_reaggregated.csv` from the published per-image records. It does not need the raw datasets or model weights.

For the fixed-checkpoint intervention, run `python resources/or4/analysis/aggregate_validate_publish.py aggregate`. This reproduces all six endpoints, paired confidence intervals, physical-unit coverage, and seed effects from the released records. Reaggregation overwrites derived summaries and the generated OR4 README; use a disposable clone when checking an immutable snapshot.

## Scope and exclusions

Raw NYUv2, DIODE, and ARKitScenes data, converted RGB/depth NPZ inputs, and model weights are not redistributed. Ten diagnostic NPZ files containing DIODE RGB/depth pixels were removed from the public package; their original filenames and SHA-256 values remain in the corresponding OR2 marker records. This exclusion covers both upstream CompletionFormer/OMNI-DC weights and this study's trained adapters, including the six CompletionFormer Spatial and Image Mean checkpoint files for seeds 42, 43, and 44. The public repository currently provides no download for those adapter files. Model-level replay requires obtaining the required files separately; the published hashes identify them but do not supply them. Official acquisition references cover the upstream projects and datasets, not the unpublished adapter files.

The frozen protocol JSON files preserve the original scientific decisions. Machine-specific paths were replaced with environment placeholders for public release, so their original sealed-file hashes are provenance values rather than hashes of the sanitized copies. `manifests/repository_sha256.txt` is the integrity record for this public tree.

## Re-running model inference

`scripts/replay_completionformer.py` runs one frozen condition using explicitly supplied local inputs. It evaluates Parent, Spatial, and the fixed-checkpoint mean-gate intervention, and optionally the separately trained Image Mean branch. It checks input, upstream-source, and checkpoint identities and compares unseen8 Spatial/intervention scores with the published OR4 records. The command, input schema, dependencies, and reference-check options are documented in [REPRODUCTION.md](REPRODUCTION.md).

This entry point consumes a converted NPZ, an upstream source checkout, and the required checkpoint files; it does not read raw dataset directories. It does not download data or weights, convert raw images, retrain adapters, or run OMNI-DC. The training snapshots and source-domain manifests preserve implementation and training provenance; they are not a turnkey retraining workflow. Replacing machine-specific paths with literal placeholders did not make the legacy execution-stack scripts portable. Statistical reaggregation remains directly available from the public files without any model inputs.

## License and third-party material

Original code, documentation, and derived research artifacts in this repository are released under the MIT License in `LICENSE`. This root license does not relicense third-party datasets, upstream projects, model weights, or the ARKitScenes source snapshot under `resources/or3/source`; those materials remain subject to their original terms. See `THIRD_PARTY_NOTICES.md`, `resources/or3/source/LICENSE`, `training_snapshot/CompletionFormer_LICENSE`, and `training_snapshot/OMNI-DC_LICENSE`.

## Versioned preservation

The existing `v1.0-submission` tag is retained as an immutable record of the first public artifact snapshot. Licensing and post-tag documentation updates will be preserved in a new `v1.0.1-submission` release rather than by moving the existing tag. `ZENODO_RELEASE_CHECKLIST.md` records the remaining metadata and release steps. The repository will display the Zenodo concept DOI and version DOI after the release has been archived; no DOI has been minted yet.

The OR4 submission snapshot is identified by its full Git commit in the manuscript. It includes the MIT license and validated intervention records. The author-metadata-dependent named release and Zenodo deposit remain separate from this publicly accessible, commit-pinned snapshot.
