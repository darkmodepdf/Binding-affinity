#!/usr/bin/env python
"""Family-aware representation regressor for antibody-antigen affinity.

Pipeline:
1) Load filtered H+L+antigen CSV with numeric affinity.
2) Build antigen-family groups from k-mer signatures to reduce leakage.
3) Group-aware split (train/val/test).
4) Extract embeddings for heavy/light/antigen (PLM or hashed k-mer).
5) Train shallow MLP regressor on fused embeddings.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Iterable, List, Sequence

import joblib
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

NUM_RE = re.compile(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train family-aware representation regressor")
    parser.add_argument("--input-csv", type=Path, default=Path("csv/asd_regression_ready_hla.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("csv/model_artifacts"))
    parser.add_argument("--embedding-mode", choices=["plm", "kmer"], default="kmer")
    parser.add_argument("--plm-model", default="facebook/esm2_t6_8M_UR50D")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--kmer-size", type=int, default=3)
    parser.add_argument("--kmer-dim", type=int, default=512)
    parser.add_argument("--family-k", type=int, default=4)
    parser.add_argument("--family-topn", type=int, default=8)
    parser.add_argument("--max-per-family", type=int, default=2000)
    parser.add_argument("--test-size", type=float, default=0.15)
    parser.add_argument("--val-size", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-rows", type=int, default=0, help="0 means use all rows")
    return parser.parse_args()


def extract_numeric_affinity(value: str) -> float:
    if value is None:
        return np.nan
    text = str(value).strip()
    match = NUM_RE.search(text)
    return float(match.group(0)) if match else np.nan


def clean_seq(seq: str) -> str:
    if not isinstance(seq, str):
        return ""
    seq = seq.strip().upper()
    return "".join(ch for ch in seq if "A" <= ch <= "Z")


def antigen_family_id(seq: str, k: int = 4, topn: int = 8) -> str:
    seq = clean_seq(seq)
    if len(seq) < k:
        return f"short::{seq}"
    kmers = [seq[i : i + k] for i in range(len(seq) - k + 1)]
    counts = Counter(kmers)
    signature = tuple(sorted(counts.items(), key=lambda x: (-x[1], x[0]))[:topn])
    return "|".join([f"{k}:{v}" for k, v in signature])


def load_data(path: Path, max_rows: int, family_k: int, family_topn: int) -> pd.DataFrame:
    usecols = [
        "dataset",
        "heavy_sequence",
        "light_sequence",
        "antigen_sequence",
        "affinity_type",
        "affinity",
    ]
    df = pd.read_csv(path, usecols=usecols)
    df["heavy_sequence"] = df["heavy_sequence"].map(clean_seq)
    df["light_sequence"] = df["light_sequence"].map(clean_seq)
    df["antigen_sequence"] = df["antigen_sequence"].map(clean_seq)
    df["affinity_numeric"] = df["affinity"].map(extract_numeric_affinity)

    df = df[
        (df["heavy_sequence"].str.len() > 0)
        & (df["light_sequence"].str.len() > 0)
        & (df["antigen_sequence"].str.len() > 0)
        & (df["affinity_type"].astype(str).str.lower() != "bool")
        & (~df["affinity_numeric"].isna())
    ].copy()

    # Different datasets use different affinity units; normalize within affinity_type.
    stats = df.groupby("affinity_type")["affinity_numeric"].agg(["mean", "std"]).reset_index()
    stats = stats.rename(columns={"mean": "at_mean", "std": "at_std"})
    df = df.merge(stats, on="affinity_type", how="left")
    df["at_std"] = df["at_std"].replace(0, 1.0).fillna(1.0)
    df["target"] = (df["affinity_numeric"] - df["at_mean"]) / df["at_std"]

    df["antigen_family"] = df["antigen_sequence"].map(lambda s: antigen_family_id(s, family_k, family_topn))

    if max_rows and max_rows > 0 and len(df) > max_rows:
        df = df.sample(n=max_rows, random_state=42).reset_index(drop=True)

    return df


def cap_per_family(df: pd.DataFrame, family_col: str, max_per_family: int, seed: int) -> pd.DataFrame:
    if max_per_family <= 0:
        return df
    pieces = []
    for _, group in df.groupby(family_col):
        if len(group) > max_per_family:
            pieces.append(group.sample(n=max_per_family, random_state=seed))
        else:
            pieces.append(group)
    return pd.concat(pieces, axis=0).sample(frac=1.0, random_state=seed).reset_index(drop=True)


class KmerEmbedder:
    def __init__(self, k: int = 3, dim: int = 512):
        self.k = k
        self.dim = dim

    def _embed_one(self, seq: str) -> np.ndarray:
        seq = clean_seq(seq)
        v = np.zeros(self.dim, dtype=np.float32)
        if len(seq) < self.k:
            return v
        total = 0
        for i in range(len(seq) - self.k + 1):
            kmer = seq[i : i + self.k]
            idx = hash(kmer) % self.dim
            sign = 1.0 if (hash("s_" + kmer) & 1) else -1.0
            v[idx] += sign
            total += 1
        if total > 0:
            v /= float(total)
        return v

    def embed(self, seqs: Sequence[str]) -> np.ndarray:
        return np.vstack([self._embed_one(s) for s in seqs])


class PLMEmbedder:
    def __init__(self, model_name: str, max_length: int = 512, batch_size: int = 32):
        import torch
        from transformers import AutoModel, AutoTokenizer

        self.torch = torch
        self.max_length = max_length
        self.batch_size = batch_size
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        self.model.eval()

    def embed(self, seqs: Sequence[str]) -> np.ndarray:
        torch = self.torch
        out_batches: List[np.ndarray] = []
        with torch.no_grad():
            for i in range(0, len(seqs), self.batch_size):
                batch = [clean_seq(s) for s in seqs[i : i + self.batch_size]]
                toks = self.tokenizer(
                    batch,
                    return_tensors="pt",
                    truncation=True,
                    padding=True,
                    max_length=self.max_length,
                )
                toks = {k: v.to(self.device) for k, v in toks.items()}
                out = self.model(**toks)
                hidden = out.last_hidden_state
                mask = toks["attention_mask"].unsqueeze(-1)
                pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
                out_batches.append(pooled.cpu().numpy())
        return np.vstack(out_batches)


def build_fused_features(df: pd.DataFrame, embedder) -> np.ndarray:
    eh = embedder.embed(df["heavy_sequence"].tolist())
    el = embedder.embed(df["light_sequence"].tolist())
    ea = embedder.embed(df["antigen_sequence"].tolist())
    return np.concatenate([eh, el, ea], axis=1)


def evaluate(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    rho = spearmanr(y_true, y_pred).correlation
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "r2": float(r2_score(y_true, y_pred)),
        "spearman": float(0.0 if np.isnan(rho) else rho),
    }


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    df = load_data(args.input_csv, args.max_rows, args.family_k, args.family_topn)

    # Family-aware split to reduce same-antigen-family leakage.
    gss1 = GroupShuffleSplit(n_splits=1, test_size=args.test_size, random_state=args.seed)
    train_val_idx, test_idx = next(gss1.split(df, groups=df["antigen_family"]))
    train_val_df = df.iloc[train_val_idx].reset_index(drop=True)
    test_df = df.iloc[test_idx].reset_index(drop=True)

    gss2 = GroupShuffleSplit(n_splits=1, test_size=args.val_size, random_state=args.seed + 1)
    tr_idx, va_idx = next(gss2.split(train_val_df, groups=train_val_df["antigen_family"]))
    train_df = train_val_df.iloc[tr_idx].reset_index(drop=True)
    val_df = train_val_df.iloc[va_idx].reset_index(drop=True)

    train_df = cap_per_family(train_df, "antigen_family", args.max_per_family, args.seed)

    if args.embedding_mode == "plm":
        embedder = PLMEmbedder(args.plm_model, max_length=args.max_length, batch_size=args.batch_size)
    else:
        embedder = KmerEmbedder(k=args.kmer_size, dim=args.kmer_dim)

    x_train = build_fused_features(train_df, embedder)
    x_val = build_fused_features(val_df, embedder)
    x_test = build_fused_features(test_df, embedder)

    y_train = train_df["target"].to_numpy(dtype=np.float32)
    y_val = val_df["target"].to_numpy(dtype=np.float32)
    y_test = test_df["target"].to_numpy(dtype=np.float32)

    scaler = StandardScaler()
    x_train_sc = scaler.fit_transform(x_train)
    x_val_sc = scaler.transform(x_val)
    x_test_sc = scaler.transform(x_test)

    model = MLPRegressor(
        hidden_layer_sizes=(512, 256),
        activation="relu",
        solver="adam",
        alpha=1e-4,
        batch_size=256,
        learning_rate_init=1e-3,
        max_iter=80,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=8,
        random_state=args.seed,
    )
    model.fit(x_train_sc, y_train)

    val_pred = model.predict(x_val_sc)
    test_pred = model.predict(x_test_sc)
    val_metrics = evaluate(y_val, val_pred)
    test_metrics = evaluate(y_test, test_pred)

    metrics = {
        "n_train": int(len(train_df)),
        "n_val": int(len(val_df)),
        "n_test": int(len(test_df)),
        "unique_families_train": int(train_df["antigen_family"].nunique()),
        "unique_families_val": int(val_df["antigen_family"].nunique()),
        "unique_families_test": int(test_df["antigen_family"].nunique()),
        "embedding_mode": args.embedding_mode,
        "plm_model": args.plm_model if args.embedding_mode == "plm" else None,
        "val": val_metrics,
        "test": test_metrics,
    }

    with (args.output_dir / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    train_df[["dataset", "affinity_type", "target", "antigen_family"]].to_csv(
        args.output_dir / "train_split.csv", index=False
    )
    val_df[["dataset", "affinity_type", "target", "antigen_family"]].to_csv(
        args.output_dir / "val_split.csv", index=False
    )
    test_df[["dataset", "affinity_type", "target", "antigen_family"]].to_csv(
        args.output_dir / "test_split.csv", index=False
    )

    joblib.dump(model, args.output_dir / "mlp_model.joblib")
    joblib.dump(scaler, args.output_dir / "feature_scaler.joblib")

    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
