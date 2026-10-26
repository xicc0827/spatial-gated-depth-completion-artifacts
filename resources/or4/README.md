# OR4: fixed-checkpoint gate intervention

This resource reports the complete inference-only intervention in which
each trained Spatial checkpoint is held fixed and its position-dependent
gate is replaced by the within-image mean of that same gate. The separately
trained Image Mean branch is not used in this comparison.

## Scope and pairing

- DIODE: 304 images, all eight unseen layouts, five paired support draws,
  three Spatial checkpoints, and three equally weighted physical scenes.
- ARKitScenes: 247 images from 20 held-out sites, with the same eight
  unseen layouts, five paired draws, and three checkpoints.
- Pairing keys are dataset, image index, layout, repetition, training seed,
  support-coordinate hash, and scoring mask/valid-pixel count.
- Negative `delta_spatial_minus_intervention` means that retaining the
  spatial variation of the gate produced lower error.

## Primary RMSE result

| Dataset | Spatial | Fixed-checkpoint mean-gate | Delta | Change | 95% CI of delta |
|---|---:|---:|---:|---:|---:|
| DIODE | 0.991934 | 1.114202 | -0.122268 | -10.97% | [-0.238971, -0.048840] |
| ARKitScenes | 0.434090 | 0.464554 | -0.030464 | -6.56% | [-0.033161, -0.027531] |

The table above is descriptive of the frozen cohorts. DIODE intervals
resample three scenes and therefore mainly expose composition sensitivity.
All endpoints, seeds, individual physical units, and coverage counts are
retained under `summary/`.

## Files

- `protocol/frozen_protocol.json`: intervention, identities, grids, and
  aggregation rules fixed before the new counterfactual inference.
- `per_image/{diode,arkitscenes}/`: complete paired condition records and
  per-file receipts.
- `summary/paired_effects.csv`: physical-unit-balanced effects and CIs.
- `summary/coverage.csv`: image/scan/scene/site directional coverage.
- `summary/seed_effects.csv`: all three checkpoint-specific effects.
- `checks/validation.json`: identity, grid, pairing, replay, finite-value,
  and aggregation checks.

## Reaggregation

Run `python analysis/aggregate_validate_publish.py aggregate` from the OR4 directory. The script resolves its inputs relative to its own
location and verifies the frozen aggregation-code hash. Model-level reruns additionally require the private
Level 3 checkpoints, frozen converted inputs, and original source stack.

Validation status: `PASS`; total paired rows: 132240; protocol SHA-256: `044a1555c515ac570d59358cbbe12a580fe05fb48a628f519f49a7b1adf6db6f`.

## Execution notes

Run model inference once per dataset in separate processes using `--dataset diode`
and `--dataset arkitscenes`; the upstream loader permits one initialization per
process. The executed evaluator includes an elementwise broadcast check in place
of a redundant floating-point reduction. `checks/implementation_amendment.json`
records the original and executed code hashes; prediction computation and the
frozen scientific protocol are unchanged.
