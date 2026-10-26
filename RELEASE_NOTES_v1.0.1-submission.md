# v1.0.1-submission release candidate

This release candidate preserves `v1.0-submission` as the immutable first
public snapshot and prepares a new citable submission artifact.

Associated manuscript: **An Attribution-Aware Empirical Study of Spatially
Gated Depth-Completion Adaptation under a Fixed Observation Budget**.

## Changes since v1.0-submission

- Added a root MIT `LICENSE` for original project code, documentation, and
  derived research artifacts.
- Clarified third-party boundaries for NYUv2, DIODE, ARKitScenes,
  CompletionFormer, and OMNI-DC in `THIRD_PARTY_NOTICES.md`.
- Retained the upstream CompletionFormer, OMNI-DC, and ARKitScenes license
  records with their corresponding snapshots.
- Added release and Zenodo preservation checks without moving the original
  `v1.0-submission` tag.
- Added `resources/or4` for the fixed-checkpoint inference intervention in
  which one trained Spatial checkpoint is evaluated with its original spatial
  gate and with the same gate replaced by its within-image mean.

The validated OR4 files can be cited by the immutable Git commit before the
named preservation release. Final creator metadata remains necessary for
the named release and Zenodo deposit; it does not prevent access to the
commit-pinned research records.

## Release gate

The `v1.0.1-submission` tag and GitHub Release must not be created until all of
the following are true:

1. The final creator list, affiliations, and available ORCID identifiers have
   been entered into a validated root `CITATION.cff`.
2. The fixed-checkpoint intervention records in `resources/or4` are complete,
   internally validated, and cited consistently by the manuscript and online
   supplement.
3. `python scripts/reaggregate.py` reproduces the principal reported metrics.
4. `python scripts/build_repository_manifest.py` has refreshed the repository
   hash manifest after all final files are present.
5. `python scripts/verify_artifacts.py` and
   `python scripts/release_preflight.py --strict` both pass from a clean clone.
6. The public tag is created at the verified release commit and the existing
   `v1.0-submission` tag still resolves to
   `aa31cd4d8a8c424b9ceeffa274d8815a0d986494`.

No DOI has been minted for this release candidate. After Zenodo archives the
final GitHub Release, record the exact version DOI and concept DOI in the
README and the manuscript data/code availability statements.
