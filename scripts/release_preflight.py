"""Validate immutable-tag and release-readiness conditions.

The default check validates the public base artifact.  ``--strict`` additionally
requires final citation metadata and the fixed-checkpoint intervention files
needed for the v1.0.1-submission release.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import subprocess


ROOT = Path(__file__).resolve().parents[1]
V1_TAG = "v1.0-submission"
V1_COMMIT = "aa31cd4d8a8c424b9ceeffa274d8815a0d986494"

BASE_REQUIRED = (
    "LICENSE",
    "README.md",
    "REPRODUCTION.md",
    "THIRD_PARTY_NOTICES.md",
    "ZENODO_RELEASE_CHECKLIST.md",
    "training_snapshot/CompletionFormer_LICENSE",
    "training_snapshot/OMNI-DC_LICENSE",
    "resources/or3/source/LICENSE",
    "manifests/repository_sha256.txt",
)

STRICT_REQUIRED = (
    "CITATION.cff",
    "resources/or4/protocol/frozen_protocol.json",
    "resources/or4/summary/paired_effects.csv",
    "resources/or4/summary/coverage.csv",
    "resources/or4/summary/seed_effects.csv",
    "resources/or4/checks/validation.json",
    "manifests/or4_sha256.txt",
)


def git(*args: str) -> str:
    return subprocess.check_output(
        ["git", *args], cwd=ROOT, text=True, encoding="utf-8"
    ).strip()


def require_files(paths: tuple[str, ...], failures: list[str]) -> None:
    for relative in paths:
        path = ROOT / relative
        if not path.is_file() or path.stat().st_size == 0:
            failures.append(f"missing or empty required file: {relative}")


def validate_citation(failures: list[str]) -> None:
    path = ROOT / "CITATION.cff"
    if not path.is_file():
        return
    text = path.read_text(encoding="utf-8")
    required_patterns = {
        "cff-version": r"(?m)^cff-version:\s*['\"]?1\.2\.0['\"]?\s*$",
        "message": r"(?m)^message:\s*\S",
        "title": r"(?m)^title:\s*\S",
        "authors": r"(?m)^authors:\s*$",
        "version": r"(?m)^version:\s*['\"]?v?1\.0\.1-submission['\"]?\s*$",
        "license": r"(?m)^license:\s*MIT\s*$",
    }
    for field, pattern in required_patterns.items():
        if not re.search(pattern, text):
            failures.append(f"CITATION.cff missing or invalid field: {field}")
    forbidden = ("REQUIRED", "PLACEHOLDER", "TBD", "TODO", "YOUR NAME")
    upper = text.upper()
    for marker in forbidden:
        if marker in upper:
            failures.append(f"CITATION.cff contains placeholder marker: {marker}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--strict",
        action="store_true",
        help="require final citation metadata and OR4 intervention records",
    )
    args = parser.parse_args()

    failures: list[str] = []
    try:
        tag_commit = git("rev-parse", f"{V1_TAG}^{{}}")
    except subprocess.CalledProcessError:
        failures.append(f"missing immutable tag: {V1_TAG}")
    else:
        if tag_commit != V1_COMMIT:
            failures.append(
                f"immutable tag moved: {V1_TAG}={tag_commit}, expected {V1_COMMIT}"
            )

    require_files(BASE_REQUIRED, failures)
    validate_citation(failures)
    intervention = ROOT / "resources/or4/checks/validation.json"
    if intervention.exists():
        require_files(tuple(p for p in STRICT_REQUIRED if p != "CITATION.cff"), failures)
        result = json.loads(intervention.read_text(encoding="utf-8"))
        if result.get("status") != "PASS" or result.get("total_rows") != 132240:
            failures.append("OR4 validation is incomplete")
    if args.strict:
        require_files(STRICT_REQUIRED, failures)

    if failures:
        for failure in failures:
            print(f"FAIL: {failure}")
        raise SystemExit(1)

    mode = "strict release" if args.strict else "base artifact"
    print(f"PASS: {mode} preflight")


if __name__ == "__main__":
    main()
