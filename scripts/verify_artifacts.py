"""Verify the public artifact tree against manifests/repository_sha256.txt."""
from __future__ import annotations

import hashlib
from pathlib import Path


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


root = Path(__file__).resolve().parents[1]
manifest = root / "manifests/repository_sha256.txt"
checked = 0
for line in manifest.read_text(encoding="utf-8").splitlines():
    digest, rel = line.split("  ", 1)
    actual = sha256(root / rel)
    if actual != digest:
        raise SystemExit(f"hash mismatch: {rel}")
    checked += 1
print(f"PASS: {checked} files verified")
