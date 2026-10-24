"""Build the repository-wide SHA-256 manifest.

The manifest covers every Git-tracked file and every untracked, non-ignored
file.  The manifest itself is excluded because it cannot contain its own
stable digest.  Paths come directly from Git and therefore use POSIX
separators on every platform.
"""
from __future__ import annotations

import hashlib
import os
from pathlib import Path
import subprocess


CHUNK_SIZE = 4 * 1024 * 1024


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(CHUNK_SIZE), b""):
            digest.update(chunk)
    return digest.hexdigest()


def repository_paths(root: Path, excluded: set[str]) -> list[str]:
    raw = subprocess.check_output(
        ["git", "ls-files", "--cached", "--others", "--exclude-standard", "-z"],
        cwd=root,
    )
    paths = []
    for encoded_path in raw.split(b"\0"):
        if not encoded_path:
            continue
        relative_path = encoded_path.decode("utf-8")
        if relative_path in excluded:
            continue
        if "\n" in relative_path or "\r" in relative_path:
            raise ValueError(f"manifest cannot represent a path with a newline: {relative_path!r}")
        paths.append(relative_path)
    return sorted(set(paths))


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    manifest = root / "manifests/repository_sha256.txt"
    temporary = manifest.with_name(f".{manifest.name}.tmp")
    temporary.unlink(missing_ok=True)

    manifest_rel = manifest.relative_to(root).as_posix()
    temporary_rel = temporary.relative_to(root).as_posix()
    relative_paths = repository_paths(root, {manifest_rel, temporary_rel})

    lines = []
    for relative_path in relative_paths:
        source = root / Path(relative_path)
        if not source.is_file():
            raise FileNotFoundError(f"cannot hash missing tracked file: {relative_path}")
        lines.append(f"{sha256(source)}  {relative_path}")

    payload = ("\n".join(lines) + "\n").encode("utf-8")
    try:
        with temporary.open("wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(manifest)
    finally:
        temporary.unlink(missing_ok=True)

    print(f"WROTE: {len(relative_paths)} files to {manifest_rel}")


if __name__ == "__main__":
    main()
