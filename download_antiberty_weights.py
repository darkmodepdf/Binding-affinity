#!/usr/bin/env python
"""Download/install AntiBERTy weights into the expected project location.

Supports:
1) --url <http(s)://.../weights.zip|tar.gz>
2) --archive <local zip/tar.gz>

Expected extracted structure (or equivalent files):
- trained_models/AntiBERTy_md_smooth/
- trained_models/vocab.txt
"""

from __future__ import annotations

import argparse
import shutil
import tarfile
import tempfile
import urllib.request
import zipfile
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Install AntiBERTy weights")
    p.add_argument("--url", help="Direct URL to zip/tar(.gz) archive containing AntiBERTy trained_models")
    p.add_argument("--archive", type=Path, help="Local archive path (zip/tar/tar.gz/tgz)")
    p.add_argument(
        "--target-dir",
        type=Path,
        default=Path("models/AntiBERTy/antiberty/trained_models"),
        help="Target trained_models directory",
    )
    p.add_argument("--force", action="store_true", help="Overwrite existing files if present")
    return p.parse_args()


def extract_archive(archive_path: Path, dest_dir: Path) -> None:
    suffix = archive_path.name.lower()
    if suffix.endswith(".zip"):
        with zipfile.ZipFile(archive_path, "r") as zf:
            zf.extractall(dest_dir)
        return
    if suffix.endswith(".tar") or suffix.endswith(".tar.gz") or suffix.endswith(".tgz"):
        with tarfile.open(archive_path, "r:*") as tf:
            tf.extractall(dest_dir)
        return
    raise ValueError(f"Unsupported archive type: {archive_path}")


def find_required(root: Path) -> tuple[Path | None, Path | None]:
    ckpt = None
    vocab = None
    for p in root.rglob("*"):
        if p.is_dir() and p.name == "AntiBERTy_md_smooth":
            ckpt = p
        elif p.is_file() and p.name == "vocab.txt":
            # Prefer vocab under a trained_models path if present.
            if "trained_models" in p.parts:
                vocab = p
            elif vocab is None:
                vocab = p
    return ckpt, vocab


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def copy_tree(src: Path, dst: Path, force: bool) -> None:
    if dst.exists() and force:
        shutil.rmtree(dst)
    if dst.exists() and not force:
        raise FileExistsError(f"Target exists: {dst}. Use --force to overwrite.")
    shutil.copytree(src, dst)


def copy_file(src: Path, dst: Path, force: bool) -> None:
    ensure_parent(dst)
    if dst.exists() and not force:
        raise FileExistsError(f"Target exists: {dst}. Use --force to overwrite.")
    shutil.copy2(src, dst)


def main() -> None:
    args = parse_args()

    if bool(args.url) == bool(args.archive):
        raise SystemExit("Provide exactly one of --url or --archive.")

    target_dir = args.target_dir.resolve()
    target_dir.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="antiberty_weights_") as tmp:
        tmp_dir = Path(tmp)

        if args.url:
            archive_path = tmp_dir / "weights_download"
            print(f"Downloading: {args.url}")
            urllib.request.urlretrieve(args.url, archive_path)
            # Guess extension from URL for extractor.
            url_lower = args.url.lower()
            if url_lower.endswith(".zip"):
                archive_path = archive_path.with_suffix(".zip")
            elif url_lower.endswith(".tar.gz"):
                archive_path = archive_path.with_suffix(".tar.gz")
            elif url_lower.endswith(".tgz"):
                archive_path = archive_path.with_suffix(".tgz")
            elif url_lower.endswith(".tar"):
                archive_path = archive_path.with_suffix(".tar")
            # Re-download to final named file if suffix changed.
            urllib.request.urlretrieve(args.url, archive_path)
        else:
            archive_path = args.archive.resolve()
            if not archive_path.exists():
                raise FileNotFoundError(f"Archive not found: {archive_path}")

        extract_dir = tmp_dir / "extracted"
        extract_dir.mkdir(parents=True, exist_ok=True)
        print(f"Extracting: {archive_path}")
        extract_archive(archive_path, extract_dir)

        ckpt_src, vocab_src = find_required(extract_dir)
        if ckpt_src is None or vocab_src is None:
            raise FileNotFoundError(
                "Could not find required files in extracted archive: "
                "AntiBERTy_md_smooth directory and vocab.txt"
            )

        ckpt_dst = target_dir / "AntiBERTy_md_smooth"
        vocab_dst = target_dir / "vocab.txt"

        copy_tree(ckpt_src, ckpt_dst, force=args.force)
        copy_file(vocab_src, vocab_dst, force=args.force)

    print("AntiBERTy weights installed successfully.")
    print(f"Checkpoint dir: {target_dir / 'AntiBERTy_md_smooth'}")
    print(f"Vocab file: {target_dir / 'vocab.txt'}")


if __name__ == "__main__":
    main()
