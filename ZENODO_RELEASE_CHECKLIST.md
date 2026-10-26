# Zenodo release checklist

This checklist prepares the repository for a citable preservation release without changing the existing `v1.0-submission` tag.

## Verified release state (2026-10-26)

- The public repository, `main` branch, root `LICENSE`, and existing
  `v1.0-submission` Release are accessible without authentication.
- The pre-OR4 public base commit is `fce4285d156e8de6fd2d1eea18cf8af794be8622`.
- The immutable `v1.0-submission` tag resolves to
  `aa31cd4d8a8c424b9ceeffa274d8815a0d986494`.
- No `v1.0.1-submission` tag or Release exists yet.
- The local environment does not provide GitHub CLI. The final annotated tag
  can be pushed with Git, but the GitHub Release must be created through the
  GitHub web interface or an authenticated API client.
- OR4 now contains all 132,240 records for 551 images, with complete-grid,
  hash, pairing, replay, and independent aggregation checks passed. A copy
  of the public OR4 tree reproduces the paired summary byte for byte.
- The manuscript cites the OR4 snapshot by its full immutable Git commit.
  Final creator metadata remains the gate for the named preservation
  release and Zenodo deposit; no new named tag or DOI has been created.

## Metadata required from the authors

- Final artifact title.
- Complete creator list in publication order, with given names and family names.
- ORCID identifiers where available.
- Final institutional affiliations.
- Release date and version (`v1.0.1-submission` is planned for the post-license snapshot).
- Funding metadata and grant identifiers, if applicable.
- Related manuscript identifier or DOI when available.

Do not publish a Zenodo record with placeholder author information. Add and validate a root `CITATION.cff` only after the creator list is final. Do not add both `CITATION.cff` and `.zenodo.json` unless there is a specific metadata need, because Zenodo gives `.zenodo.json` precedence.

## Repository checks before release

1. Run `python scripts/reaggregate.py` and inspect the regenerated primary metrics.
2. Add and validate the fixed-checkpoint intervention under `resources/or4`
   according to its `README.md`, including `manifests/or4_sha256.txt`.
3. After the final metadata files, including `CITATION.cff`, are present, run `python scripts/build_repository_manifest.py`.
4. Run `python scripts/verify_artifacts.py` and confirm that every file in `manifests/repository_sha256.txt` passes.
5. Run `python scripts/release_preflight.py --strict`; it must pass before the
   release commit is tagged.
6. Confirm that raw NYUv2, DIODE, and ARKitScenes pixels and all model weights remain excluded.
7. Confirm that the root MIT `LICENSE`, `THIRD_PARTY_NOTICES.md`, and retained upstream licenses are present.
8. Validate `CITATION.cff` with a Citation File Format validator.
9. Create the new annotated tag `v1.0.1-submission`; do not move or overwrite `v1.0-submission`.

## Zenodo/GitHub release sequence

1. Link the repository owner's GitHub account to Zenodo.
2. Enable `xicc0827/spatial-gated-depth-completion-artifacts` in Zenodo's GitHub integration.
3. Create the GitHub release from `v1.0.1-submission` only after all metadata checks pass.
4. Wait for Zenodo to ingest the release and inspect the resulting metadata.
5. Record both identifiers: the version DOI for the exact archived release and the concept DOI for all versions.
6. Add the DOI badge and citation text to the README, and replace the manuscript's tag-only artifact citation with the version DOI plus repository URL.

Zenodo's GitHub integration does not support pre-reserving a DOI before the GitHub release. If the DOI must be inserted into files before publication, use a manual Zenodo deposit, reserve its DOI in the draft, upload the exact tagged archive, verify metadata, and then publish the record.
