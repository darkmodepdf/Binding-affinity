#!/usr/bin/env python
"""Family-aware representation regressor for antibody-antigen affinity.

Goal-aligned design:
- Frozen PLM embeddings for heavy/light/antigen
- Lightweight fusion regressor (concat or attention pooling)
- Antigen-family-aware split to reduce leakage
- GPU-first training/inference with optional embedding cache for speed
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import joblib
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    explained_variance_score,
    f1_score,
    matthews_corrcoef,
    mean_absolute_error,
    mean_squared_error,
    median_absolute_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GroupShuffleSplit
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from tqdm.auto import tqdm

NUM_RE = re.compile(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?")
PROGRESS_ENABLED = True


def set_progress_enabled(enabled: bool) -> None:
    global PROGRESS_ENABLED
    PROGRESS_ENABLED = enabled


def pbar(iterable, **kwargs):
    return tqdm(iterable, disable=not PROGRESS_ENABLED, dynamic_ncols=True, **kwargs)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train family-aware representation regressor")
    parser.add_argument("--input-csv", type=Path, default=Path("csv/asd_regression_ready_hla.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("csv/model_artifacts"))

    # Backbone config
    parser.add_argument("--backbone-mode", choices=["shared_plm", "mixed_antiberty_esm", "kmer"], default="mixed_antiberty_esm")
    parser.add_argument("--plm-model", default="facebook/esm2_t33_650M_UR50D")
    parser.add_argument("--antigen-model", default="facebook/esm2_t33_650M_UR50D")

    parser.add_argument("--antiberty-dir", type=Path, default=Path("models/AntiBERTy"))
    parser.add_argument(
        "--antiberty-checkpoint",
        type=Path,
        default=Path("models/AntiBERTy/antiberty/trained_models/AntiBERTy_md_smooth"),
    )
    parser.add_argument(
        "--antiberty-vocab",
        type=Path,
        default=Path("models/AntiBERTy/antiberty/trained_models/vocab.txt"),
    )

    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--kmer-size", type=int, default=3)
    parser.add_argument("--kmer-dim", type=int, default=512)

    # Fusion + head
    parser.add_argument("--fusion-mode", choices=["concat", "attention"], default="attention")
    parser.add_argument("--proj-dim", type=int, default=512)
    parser.add_argument("--regressor-backend", choices=["torch", "sklearn"], default="torch")
    parser.add_argument("--device", default="auto", help="auto|cuda|cpu")
    parser.add_argument("--hidden-dims", default="1024,512,256")
    parser.add_argument("--dropout", type=float, default=0.15)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--num-workers", type=int, default=0)

    # Leakage + balancing
    parser.add_argument("--family-k", type=int, default=4)
    parser.add_argument("--family-topn", type=int, default=8)
    parser.add_argument("--max-per-family", type=int, default=2000)
    parser.add_argument("--test-size", type=float, default=0.15)
    parser.add_argument("--val-size", type=float, default=0.15)

    # Speed controls
    parser.add_argument("--cache-embeddings-dir", type=Path, default=Path("csv/embedding_cache"))
    parser.add_argument("--disable-embedding-cache", action="store_true")

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-rows", type=int, default=0, help="0 means use all rows")
    parser.add_argument("--show-progress", action="store_true", help="Force-enable tqdm progress bars")
    parser.add_argument("--hide-progress", action="store_true", help="Disable tqdm progress bars")
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
    return "|".join([f"{kk}:{vv}" for kk, vv in signature])


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

    # Different datasets use different affinity scales; normalize per affinity_type.
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


@dataclass
class EmbeddingBundle:
    heavy: np.ndarray
    light: np.ndarray
    antigen: np.ndarray


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
        return np.vstack([self._embed_one(s) for s in seqs]).astype(np.float32)


class HFPLMEmbedder:
    def __init__(self, model_name: str, device: str, max_length: int = 512, batch_size: int = 64):
        import torch
        from transformers import AutoModel, AutoTokenizer

        self.torch = torch
        self.device = device
        self.max_length = max_length
        self.batch_size = batch_size
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(device)
        self.model.eval()

    def embed(self, seqs: Sequence[str]) -> np.ndarray:
        torch = self.torch
        out_batches: List[np.ndarray] = []
        with torch.no_grad():
            steps = range(0, len(seqs), self.batch_size)
            for i in pbar(steps, desc="Embedding (HF PLM)", leave=False):
                batch = [clean_seq(s) for s in seqs[i : i + self.batch_size]]
                toks = self.tokenizer(
                    batch,
                    return_tensors="pt",
                    truncation=True,
                    padding=True,
                    max_length=self.max_length,
                )
                toks = {k: v.to(self.device) for k, v in toks.items()}
                with torch.amp.autocast(device_type="cuda", enabled=(self.device == "cuda")):
                    out = self.model(**toks)
                hidden = out.last_hidden_state
                mask = toks["attention_mask"].unsqueeze(-1)
                pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
                out_batches.append(pooled.float().cpu().numpy())
        return np.vstack(out_batches).astype(np.float32)


class AntiBERTySeqEmbedder:
    def __init__(
        self,
        antiberty_dir: Path,
        checkpoint_path: Path,
        vocab_file: Path,
        device: str,
        max_length: int = 512,
        batch_size: int = 64,
    ):
        import torch
        import transformers

        self.torch = torch
        self.device = device
        self.max_length = max_length
        self.batch_size = batch_size

        sys.path.insert(0, str(antiberty_dir.resolve()))
        from antiberty.model.AntiBERTy import AntiBERTy  # noqa: WPS433

        self.model = AntiBERTy.from_pretrained(str(checkpoint_path)).to(device)
        self.model.eval()
        self.tokenizer = transformers.BertTokenizer(vocab_file=str(vocab_file), do_lower_case=False)

    def _to_token_text(self, s: str) -> str:
        s = clean_seq(s)
        return " ".join(list(s))

    def embed(self, seqs: Sequence[str]) -> np.ndarray:
        torch = self.torch
        out_batches: List[np.ndarray] = []
        with torch.no_grad():
            steps = range(0, len(seqs), self.batch_size)
            for i in pbar(steps, desc="Embedding (AntiBERTy)", leave=False):
                batch = [self._to_token_text(s) for s in seqs[i : i + self.batch_size]]
                toks = self.tokenizer(
                    batch,
                    return_tensors="pt",
                    truncation=True,
                    padding=True,
                    max_length=self.max_length,
                )
                toks = {k: v.to(self.device) for k, v in toks.items()}
                with torch.amp.autocast(device_type="cuda", enabled=(self.device == "cuda")):
                    out = self.model(
                        input_ids=toks["input_ids"],
                        attention_mask=toks["attention_mask"],
                        output_hidden_states=True,
                        return_dict=True,
                    )
                hidden = out.hidden_states[-1]
                mask = toks["attention_mask"].unsqueeze(-1)
                pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
                out_batches.append(pooled.float().cpu().numpy())
        return np.vstack(out_batches).astype(np.float32)


class SharedBackboneExtractor:
    def __init__(self, plm_model: str, device: str, max_length: int, batch_size: int):
        self.embedder = HFPLMEmbedder(plm_model, device=device, max_length=max_length, batch_size=batch_size)

    def extract(self, df: pd.DataFrame) -> EmbeddingBundle:
        return EmbeddingBundle(
            heavy=self.embedder.embed(df["heavy_sequence"].tolist()),
            light=self.embedder.embed(df["light_sequence"].tolist()),
            antigen=self.embedder.embed(df["antigen_sequence"].tolist()),
        )


class MixedBackboneExtractor:
    def __init__(
        self,
        device: str,
        antigen_model: str,
        antiberty_dir: Path,
        antiberty_checkpoint: Path,
        antiberty_vocab: Path,
        max_length: int,
        batch_size: int,
    ):
        self.antiberty = AntiBERTySeqEmbedder(
            antiberty_dir=antiberty_dir,
            checkpoint_path=antiberty_checkpoint,
            vocab_file=antiberty_vocab,
            device=device,
            max_length=max_length,
            batch_size=batch_size,
        )
        self.antigen = HFPLMEmbedder(antigen_model, device=device, max_length=max_length, batch_size=batch_size)

    def extract(self, df: pd.DataFrame) -> EmbeddingBundle:
        return EmbeddingBundle(
            heavy=self.antiberty.embed(df["heavy_sequence"].tolist()),
            light=self.antiberty.embed(df["light_sequence"].tolist()),
            antigen=self.antigen.embed(df["antigen_sequence"].tolist()),
        )


class KmerExtractor:
    def __init__(self, k: int, dim: int):
        self.embedder = KmerEmbedder(k=k, dim=dim)

    def extract(self, df: pd.DataFrame) -> EmbeddingBundle:
        return EmbeddingBundle(
            heavy=self.embedder.embed(df["heavy_sequence"].tolist()),
            light=self.embedder.embed(df["light_sequence"].tolist()),
            antigen=self.embedder.embed(df["antigen_sequence"].tolist()),
        )


def save_bundle(path: Path, bundle: EmbeddingBundle, y: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, heavy=bundle.heavy, light=bundle.light, antigen=bundle.antigen, y=y)


def load_bundle(path: Path) -> Tuple[EmbeddingBundle, np.ndarray]:
    d = np.load(path)
    b = EmbeddingBundle(heavy=d["heavy"], light=d["light"], antigen=d["antigen"])
    y = d["y"].astype(np.float32)
    return b, y


def _safe_metric(fn, *args, default=np.nan, **kwargs):
    try:
        return float(fn(*args, **kwargs))
    except Exception:
        return float(default)


def evaluate(y_true: np.ndarray, y_pred: np.ndarray, cls_threshold: float) -> dict:
    rho = spearmanr(y_true, y_pred).correlation
    pearson = np.corrcoef(y_true, y_pred)[0, 1] if len(y_true) > 1 else np.nan

    # Classification-style view over continuous regression outputs.
    y_true_bin = (y_true >= cls_threshold).astype(np.int32)
    y_pred_bin = (y_pred >= cls_threshold).astype(np.int32)

    tn = fp = fn = tp = 0
    cm = confusion_matrix(y_true_bin, y_pred_bin, labels=[0, 1])
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()

    specificity = _safe_metric(lambda tnn, fpp: tnn / (tnn + fpp), tn, fp)
    sensitivity = _safe_metric(lambda tpp, fnn: tpp / (tpp + fnn), tp, fn)

    return {
        "regression": {
            "mae": float(mean_absolute_error(y_true, y_pred)),
            "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
            "r2": float(r2_score(y_true, y_pred)),
            "spearman": float(0.0 if np.isnan(rho) else rho),
            "pearson": float(0.0 if np.isnan(pearson) else pearson),
            "explained_variance": float(explained_variance_score(y_true, y_pred)),
            "median_absolute_error": float(median_absolute_error(y_true, y_pred)),
        },
        "classification_like": {
            "threshold": float(cls_threshold),
            "roc_auc": _safe_metric(roc_auc_score, y_true_bin, y_pred),
            "aoc": _safe_metric(roc_auc_score, y_true_bin, y_pred),
            "pr_auc": _safe_metric(average_precision_score, y_true_bin, y_pred),
            "accuracy": _safe_metric(accuracy_score, y_true_bin, y_pred_bin),
            "precision": _safe_metric(precision_score, y_true_bin, y_pred_bin, zero_division=0),
            "recall": _safe_metric(recall_score, y_true_bin, y_pred_bin, zero_division=0),
            "f1": _safe_metric(f1_score, y_true_bin, y_pred_bin, zero_division=0),
            "balanced_accuracy": _safe_metric(balanced_accuracy_score, y_true_bin, y_pred_bin),
            "mcc": _safe_metric(matthews_corrcoef, y_true_bin, y_pred_bin),
            "specificity": specificity,
            "sensitivity": sensitivity,
            "confusion_matrix": {
                "tn": int(tn),
                "fp": int(fp),
                "fn": int(fn),
                "tp": int(tp),
            },
            "support": {
                "n_neg": int((y_true_bin == 0).sum()),
                "n_pos": int((y_true_bin == 1).sum()),
            },
        },
    }


def fit_scalers(train: EmbeddingBundle) -> Dict[str, StandardScaler]:
    sh = StandardScaler().fit(train.heavy)
    sl = StandardScaler().fit(train.light)
    sa = StandardScaler().fit(train.antigen)
    return {"heavy": sh, "light": sl, "antigen": sa}


def apply_scalers(bundle: EmbeddingBundle, scalers: Dict[str, StandardScaler]) -> EmbeddingBundle:
    return EmbeddingBundle(
        heavy=scalers["heavy"].transform(bundle.heavy).astype(np.float32),
        light=scalers["light"].transform(bundle.light).astype(np.float32),
        antigen=scalers["antigen"].transform(bundle.antigen).astype(np.float32),
    )


class LightweightFusionRegressorTorch:
    def __init__(self, dims: Tuple[int, int, int], args: argparse.Namespace):
        import torch
        from torch import nn

        self.torch = torch
        self.nn = nn
        self.args = args

        if args.device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = args.device

        if self.device == "cuda":
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

        self.dh, self.dl, self.da = dims
        hidden_dims = [int(x.strip()) for x in args.hidden_dims.split(",") if x.strip()]

        layers: List[nn.Module] = []
        if args.fusion_mode == "concat":
            fused_dim = self.dh + self.dl + self.da
            self.fuser = None
        else:
            proj = args.proj_dim
            self.proj_h = nn.Linear(self.dh, proj)
            self.proj_l = nn.Linear(self.dl, proj)
            self.proj_a = nn.Linear(self.da, proj)
            self.gate_h = nn.Linear(proj, 1)
            self.gate_l = nn.Linear(proj, 1)
            self.gate_a = nn.Linear(proj, 1)
            self.fuser = True
            fused_dim = proj

        prev = fused_dim
        for h in hidden_dims:
            layers.extend([nn.Linear(prev, h), nn.ReLU(), nn.Dropout(args.dropout)])
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.head = nn.Sequential(*layers)

        modules = [self.head]
        if self.fuser:
            modules.extend([self.proj_h, self.proj_l, self.proj_a, self.gate_h, self.gate_l, self.gate_a])
        self.model = nn.ModuleList(modules).to(self.device)

    def parameters(self):
        params = []
        for m in self.model:
            params.extend(list(m.parameters()))
        return params

    def forward(self, h, l, a):
        torch = self.torch
        if not self.fuser:
            x = torch.cat([h, l, a], dim=1)
            pred = self.head(x).squeeze(1)
            return pred, None

        ph = self.proj_h(h)
        pl = self.proj_l(l)
        pa = self.proj_a(a)
        scores = torch.cat([self.gate_h(ph), self.gate_l(pl), self.gate_a(pa)], dim=1)
        w = torch.softmax(scores, dim=1)
        stacked = torch.stack([ph, pl, pa], dim=1)
        fused = (w.unsqueeze(-1) * stacked).sum(dim=1)
        pred = self.head(fused).squeeze(1)
        return pred, w

    def state_dict(self):
        out = {"head": self.head.state_dict()}
        if self.fuser:
            out.update(
                {
                    "proj_h": self.proj_h.state_dict(),
                    "proj_l": self.proj_l.state_dict(),
                    "proj_a": self.proj_a.state_dict(),
                    "gate_h": self.gate_h.state_dict(),
                    "gate_l": self.gate_l.state_dict(),
                    "gate_a": self.gate_a.state_dict(),
                }
            )
        return out


def train_torch_fusion(
    train_b: EmbeddingBundle,
    y_train: np.ndarray,
    val_b: EmbeddingBundle,
    y_val: np.ndarray,
    test_b: EmbeddingBundle,
    args: argparse.Namespace,
):
    import torch
    from torch.utils.data import DataLoader, TensorDataset

    dims = (train_b.heavy.shape[1], train_b.light.shape[1], train_b.antigen.shape[1])
    model = LightweightFusionRegressorTorch(dims=dims, args=args)

    htr = torch.from_numpy(train_b.heavy)
    ltr = torch.from_numpy(train_b.light)
    atr = torch.from_numpy(train_b.antigen)
    ytr = torch.from_numpy(y_train.astype(np.float32))

    hva = torch.from_numpy(val_b.heavy)
    lva = torch.from_numpy(val_b.light)
    ava = torch.from_numpy(val_b.antigen)

    hte = torch.from_numpy(test_b.heavy)
    lte = torch.from_numpy(test_b.light)
    ate = torch.from_numpy(test_b.antigen)

    num_workers = args.num_workers
    if os.name == "nt" and num_workers > 0:
        num_workers = 0

    train_loader = DataLoader(
        TensorDataset(htr, ltr, atr, ytr),
        batch_size=max(128, args.batch_size * 8),
        shuffle=True,
        num_workers=num_workers,
        pin_memory=(model.device == "cuda"),
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    loss_fn = torch.nn.SmoothL1Loss()
    scaler = torch.amp.GradScaler("cuda", enabled=(model.device == "cuda"))

    best_state = None
    best_val = float("inf")
    bad_epochs = 0

    epoch_iter = pbar(range(args.epochs), desc="Training epochs")
    for epoch_idx in epoch_iter:
        for m in model.model:
            m.train()

        batch_iter = pbar(train_loader, desc=f"Epoch {epoch_idx + 1}/{args.epochs}", leave=False)
        for hb, lb, ab, yb in batch_iter:
            hb = hb.to(model.device, non_blocking=True)
            lb = lb.to(model.device, non_blocking=True)
            ab = ab.to(model.device, non_blocking=True)
            yb = yb.to(model.device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type="cuda", enabled=(model.device == "cuda")):
                pred, _ = model.forward(hb, lb, ab)
                loss = loss_fn(pred, yb)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            batch_iter.set_postfix({"loss": float(loss.detach().cpu().item())})

        for m in model.model:
            m.eval()
        with torch.no_grad():
            pv, _ = model.forward(
                hva.to(model.device),
                lva.to(model.device),
                ava.to(model.device),
            )
            val_pred = pv.cpu().numpy()
            val_loss = float(np.mean(np.abs(val_pred - y_val)))

        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v for k, v in model.state_dict().items()}
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= args.patience:
                break

    ckpt = {
        "dims": dims,
        "fusion_mode": args.fusion_mode,
        "proj_dim": args.proj_dim,
        "hidden_dims": [int(x.strip()) for x in args.hidden_dims.split(",") if x.strip()],
        "dropout": args.dropout,
        "state_dict": best_state if best_state is not None else model.state_dict(),
    }

    with torch.no_grad():
        pv, wv = model.forward(hva.to(model.device), lva.to(model.device), ava.to(model.device))
        pt, wt = model.forward(hte.to(model.device), lte.to(model.device), ate.to(model.device))

    attention_summary = None
    if wv is not None:
        wv_mean = wv.mean(dim=0).detach().cpu().numpy().tolist()
        wt_mean = wt.mean(dim=0).detach().cpu().numpy().tolist()
        attention_summary = {
            "val_mean_weights_h_l_a": [float(x) for x in wv_mean],
            "test_mean_weights_h_l_a": [float(x) for x in wt_mean],
        }

    return pv.cpu().numpy(), pt.cpu().numpy(), model.device, ckpt, attention_summary


def train_sklearn_mlp(
    train_b: EmbeddingBundle,
    y_train: np.ndarray,
    val_b: EmbeddingBundle,
    test_b: EmbeddingBundle,
):
    x_train = np.concatenate([train_b.heavy, train_b.light, train_b.antigen], axis=1)
    x_val = np.concatenate([val_b.heavy, val_b.light, val_b.antigen], axis=1)
    x_test = np.concatenate([test_b.heavy, test_b.light, test_b.antigen], axis=1)

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
        random_state=42,
    )
    model.fit(x_train, y_train)
    return model.predict(x_val), model.predict(x_test), model


def print_cli_metrics(metrics: dict) -> None:
    print("\n" + "=" * 72)
    print("Binding Affinity Training Summary")
    print("=" * 72)
    print(f"Backbone: {metrics.get('backbone_mode')} | Fusion: {metrics.get('fusion_mode')} | Device: {metrics.get('device_used')}")
    print(f"Splits -> train: {metrics.get('n_train')} | val: {metrics.get('n_val')} | test: {metrics.get('n_test')}")
    print(f"Families -> train: {metrics.get('unique_families_train')} | val: {metrics.get('unique_families_val')} | test: {metrics.get('unique_families_test')}")

    for split in ["val", "test"]:
        block = metrics.get(split, {})
        reg = block.get("regression", {})
        cls = block.get("classification_like", {})
        print(f"\n[{split.upper()}] Regression")
        print(
            f"  MAE={reg.get('mae', float('nan')):.4f}  RMSE={reg.get('rmse', float('nan')):.4f}  R2={reg.get('r2', float('nan')):.4f}  "
            f"Spearman={reg.get('spearman', float('nan')):.4f}  Pearson={reg.get('pearson', float('nan')):.4f}"
        )
        print(f"[{split.upper()}] Classification-like")
        print(
            f"  ROC-AUC={cls.get('roc_auc', float('nan')):.4f}  PR-AUC={cls.get('pr_auc', float('nan')):.4f}  "
            f"Accuracy={cls.get('accuracy', float('nan')):.4f}  Precision={cls.get('precision', float('nan')):.4f}  "
            f"Recall={cls.get('recall', float('nan')):.4f}  F1={cls.get('f1', float('nan')):.4f}"
        )

    print("=" * 72)


def choose_device(device_arg: str) -> str:
    import torch

    if device_arg == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device_arg


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.hide_progress:
        set_progress_enabled(False)
    elif args.show_progress:
        set_progress_enabled(True)
    else:
        set_progress_enabled(sys.stderr.isatty() or sys.stdout.isatty())

    device = choose_device(args.device)

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

    if args.backbone_mode == "kmer":
        extractor = KmerExtractor(k=args.kmer_size, dim=args.kmer_dim)
    elif args.backbone_mode == "shared_plm":
        extractor = SharedBackboneExtractor(
            plm_model=args.plm_model,
            device=device,
            max_length=args.max_length,
            batch_size=args.batch_size,
        )
    else:
        if not args.antiberty_checkpoint.exists() or not args.antiberty_vocab.exists():
            raise FileNotFoundError(
                f"AntiBERTy files not found. checkpoint={args.antiberty_checkpoint} vocab={args.antiberty_vocab}"
            )
        extractor = MixedBackboneExtractor(
            device=device,
            antigen_model=args.antigen_model,
            antiberty_dir=args.antiberty_dir,
            antiberty_checkpoint=args.antiberty_checkpoint,
            antiberty_vocab=args.antiberty_vocab,
            max_length=args.max_length,
            batch_size=args.batch_size,
        )

    cache_ok = not args.disable_embedding_cache
    cache_dir = args.cache_embeddings_dir
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_tag = f"{args.backbone_mode}_seed{args.seed}_mr{args.max_rows}_f{args.family_k}_{args.family_topn}"

    def get_or_make(split_name: str, split_df: pd.DataFrame, y_vals: np.ndarray):
        cache_file = cache_dir / f"{cache_tag}_{split_name}.npz"
        if cache_ok and cache_file.exists():
            return load_bundle(cache_file)
        b = extractor.extract(split_df)
        if cache_ok:
            save_bundle(cache_file, b, y_vals)
        return b, y_vals

    y_train = train_df["target"].to_numpy(dtype=np.float32)
    y_val = val_df["target"].to_numpy(dtype=np.float32)
    y_test = test_df["target"].to_numpy(dtype=np.float32)

    train_b, y_train = get_or_make("train", train_df, y_train)
    val_b, y_val = get_or_make("val", val_df, y_val)
    test_b, y_test = get_or_make("test", test_df, y_test)

    scalers = fit_scalers(train_b)
    train_b_sc = apply_scalers(train_b, scalers)
    val_b_sc = apply_scalers(val_b, scalers)
    test_b_sc = apply_scalers(test_b, scalers)

    attention_summary = None
    if args.regressor_backend == "torch":
        val_pred, test_pred, used_device, ckpt, attention_summary = train_torch_fusion(
            train_b_sc,
            y_train,
            val_b_sc,
            y_val,
            test_b_sc,
            args,
        )
        import torch

        torch.save(ckpt, args.output_dir / "torch_regressor.pt")
    else:
        val_pred, test_pred, model = train_sklearn_mlp(train_b_sc, y_train, val_b_sc, test_b_sc)
        joblib.dump(model, args.output_dir / "mlp_model.joblib")
        used_device = "cpu"

    cls_threshold = float(np.median(y_train))
    val_metrics = evaluate(y_val, val_pred, cls_threshold)
    test_metrics = evaluate(y_test, test_pred, cls_threshold)

    metrics = {
        "n_train": int(len(train_df)),
        "n_val": int(len(val_df)),
        "n_test": int(len(test_df)),
        "unique_families_train": int(train_df["antigen_family"].nunique()),
        "unique_families_val": int(val_df["antigen_family"].nunique()),
        "unique_families_test": int(test_df["antigen_family"].nunique()),
        "backbone_mode": args.backbone_mode,
        "plm_model": args.plm_model,
        "antigen_model": args.antigen_model,
        "fusion_mode": args.fusion_mode,
        "regressor_backend": args.regressor_backend,
        "device_used": used_device,
        "embedding_dims": {
            "heavy": int(train_b.heavy.shape[1]),
            "light": int(train_b.light.shape[1]),
            "antigen": int(train_b.antigen.shape[1]),
        },
        "val": val_metrics,
        "test": test_metrics,
        "attention_summary": attention_summary,
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

    joblib.dump(scalers, args.output_dir / "feature_scalers.joblib")

    with (args.output_dir / "run_config.json").open("w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2, default=str)

    print_cli_metrics(metrics)


if __name__ == "__main__":
    main()
