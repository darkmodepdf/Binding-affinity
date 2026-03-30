#!/usr/bin/env python
"""Rebuild split dataset archives, extract them, and optionally run training.

Usage examples:
  python setup_h100_run.py --extract-only
  python setup_h100_run.py --run-train --backbone-mode mixed_antiberty_esm --fusion-mode attention
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Prepare split archives and run model training")
    p.add_argument("--archives-dir", type=Path, default=Path("compressed_for_github"))
    p.add_argument("--workdir", type=Path, default=Path("."))
    p.add_argument("--extract-only", action="store_true")
    p.add_argument("--run-train", action="store_true")

    p.add_argument("--backbone-mode", choices=["mixed_antiberty_esm", "shared_plm", "kmer"], default="mixed_antiberty_esm")
    p.add_argument("--fusion-mode", choices=["attention", "concat"], default="attention")
    p.add_argument("--plm-model", default="facebook/esm2_t33_650M_UR50D")
    p.add_argument("--antigen-model", default="facebook/esm2_t33_650M_UR50D")

    p.add_argument("--output-dir", type=Path, default=Path("csv/model_artifacts_h100"))
    p.add_argument("--max-rows", type=int, default=0)
    p.add_argument("--extra-args", nargs=argparse.REMAINDER, default=[])
    return p.parse_args()


def concat_parts(base_name: str, archives_dir: Path, out_zip: Path) -> None:
    parts = sorted(archives_dir.glob(base_name + ".part*"))
    if not parts:
        raise FileNotFoundError(f"No parts found for {base_name} in {archives_dir}")
    with out_zip.open("wb") as out:
        for part in parts:
            out.write(part.read_bytes())


def extract_zip(zip_path: Path, dest: Path) -> None:
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(dest)


def main() -> None:
    args = parse_args()
    workdir = args.workdir.resolve()
    archives_dir = (workdir / args.archives_dir).resolve()

    if not archives_dir.exists():
        raise FileNotFoundError(f"Archives dir not found: {archives_dir}")

    tmp_dir = workdir / "_tmp_rebuild_zips"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    asd_zip = tmp_dir / "asd_dataset.zip"
    csv_zip = tmp_dir / "csv_datasets.zip"

    print("Rebuilding zip files from parts...")
    concat_parts("asd_dataset.zip", archives_dir, asd_zip)
    concat_parts("csv_datasets.zip", archives_dir, csv_zip)

    print("Extracting archives...")
    extract_zip(asd_zip, workdir)
    extract_zip(csv_zip, workdir)

    shutil.rmtree(tmp_dir, ignore_errors=True)

    print("Extraction complete.")
    print(f"  asd dir: {(workdir / 'asd').resolve()}")
    print(f"  csv/asd_full.csv: {(workdir / 'csv' / 'asd_full.csv').resolve()}")
    print(f"  csv/asd_regression_ready_hla.csv: {(workdir / 'csv' / 'asd_regression_ready_hla.csv').resolve()}")

    if args.extract_only and not args.run_train:
        return

    if args.run_train:
        trainer = workdir / "csv" / "train_family_aware_regressor.py"
        if not trainer.exists():
            raise FileNotFoundError(f"Training script not found: {trainer}")

        cmd = [
            sys.executable,
            str(trainer),
            "--input-csv",
            str(workdir / "csv" / "asd_regression_ready_hla.csv"),
            "--output-dir",
            str(args.output_dir),
            "--backbone-mode",
            args.backbone_mode,
            "--fusion-mode",
            args.fusion_mode,
            "--plm-model",
            args.plm_model,
            "--antigen-model",
            args.antigen_model,
            "--regressor-backend",
            "torch",
        ]

        if args.max_rows > 0:
            cmd += ["--max-rows", str(args.max_rows)]
        if args.extra_args:
            cmd += args.extra_args

        print("Running training command:")
        print(" ".join(cmd))
        subprocess.run(cmd, check=True, cwd=workdir)


if __name__ == "__main__":
    main()
