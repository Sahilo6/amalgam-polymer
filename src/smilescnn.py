"""Character-level CNN over SMILES strings.

The third representation. Our GBDTs read hashed fingerprints and the GNN reads
graph topology; every submission so far correlates at 0.999 because those two
views are already exhausted. A sequence model reads the molecule as text -
neither a fingerprint nor a graph - which is the same kind of move that made the
GNN worth +0.0053.

Trained from scratch: pretrained weights are banned (rule 6.2.1). Single
threaded and per-fold seeded for exact reproducibility (rule 7.2).
"""
import argparse
import re

import numpy as np
import torch
import torch.nn as nn

from . import config as C
from . import cv
from .data import load_raw, subset

torch.set_num_threads(1)

# Multi-character tokens must not be split: Cl is one atom, not C followed by l.
TOKEN_RE = re.compile(r"(\[[^\]]+\]|Br|Cl|@@|%\d{2}|.)")
MAX_LEN = 160


def build_vocab(smiles):
    toks = set()
    for s in smiles:
        toks.update(TOKEN_RE.findall(s))
    return {t: i + 1 for i, t in enumerate(sorted(toks))}      # 0 = padding


def encode(smi, vocab):
    ids = [vocab.get(t, 0) for t in TOKEN_RE.findall(smi)][:MAX_LEN]
    return ids + [0] * (MAX_LEN - len(ids))


class SmilesCNN(nn.Module):
    """Dilated 1-D convolutions: wide receptive field over the string without
    the depth that made the big GNN unusably slow."""

    def __init__(self, vocab_size, emb=64, ch=128, dropout=0.15):
        super().__init__()
        self.emb = nn.Embedding(vocab_size + 1, emb, padding_idx=0)
        self.convs = nn.ModuleList([
            nn.Conv1d(emb if i == 0 else ch, ch, k, padding=d * (k - 1) // 2, dilation=d)
            for i, (k, d) in enumerate([(5, 1), (5, 2), (3, 4), (3, 8)])
        ])
        self.norms = nn.ModuleList(nn.BatchNorm1d(ch) for _ in range(4))
        self.head = nn.Sequential(nn.Linear(ch * 2, ch), nn.ReLU(),
                                  nn.Dropout(dropout), nn.Linear(ch, 1))

    def forward(self, x):
        mask = (x != 0).float().unsqueeze(1)
        h = self.emb(x).transpose(1, 2)
        for conv, norm in zip(self.convs, self.norms):
            h = torch.relu(norm(conv(h)))
            h = h * mask
        avg = h.sum(2) / mask.sum(2).clamp(min=1)
        mx = h.masked_fill(mask == 0, -1e9).max(2).values
        return self.head(torch.cat([avg, mx], 1)).squeeze(-1)


def run_target(target, epochs=60, bs=64, lr=2e-3, n_folds=5, seeds=(C.SEED,)):
    tr, te = load_raw()
    tr_t, te_t = subset(tr, target), subset(te, target)
    vocab = build_vocab(list(tr.smiles) + list(te.smiles))
    X = torch.tensor([encode(s, vocab) for s in tr_t.smiles], dtype=torch.long)
    y_raw = tr_t.target.values
    mu, sd = y_raw.mean(), y_raw.std()
    y = (y_raw - mu) / sd
    yt = torch.tensor(y, dtype=torch.float32)

    oof_seeds = []
    for si, base in enumerate(seeds):
        oof = np.zeros(len(y))
        for fold, (tr_i, va_i) in enumerate(cv.make_folds(len(y), n_splits=n_folds)):
            torch.manual_seed(base + fold)
            rng = np.random.default_rng(base + fold)
            m = SmilesCNN(len(vocab))
            opt = torch.optim.AdamW(m.parameters(), lr=lr, weight_decay=1e-5)
            sched = torch.optim.lr_scheduler.OneCycleLR(
                opt, max_lr=lr, total_steps=epochs * max(1, len(tr_i) // bs))
            best, best_state, patience = 1e9, None, 0
            for ep in range(epochs):
                m.train()
                perm = rng.permutation(tr_i)
                for k in range(0, len(perm), bs):
                    b = perm[k:k + bs]
                    if len(b) < 2:
                        continue
                    loss = nn.functional.mse_loss(m(X[b]), yt[b])
                    opt.zero_grad(); loss.backward()
                    nn.utils.clip_grad_norm_(m.parameters(), 5.0)
                    opt.step()
                    if sched.last_epoch < sched.total_steps - 1:
                        sched.step()
                if ep % 3 == 2 or ep == epochs - 1:
                    m.eval()
                    with torch.no_grad():
                        p = m(X[va_i]).numpy()
                    mse = ((p - y[va_i]) ** 2).mean()
                    if mse < best - 1e-5:
                        best, patience = mse, 0
                        best_state = {k2: v.clone() for k2, v in m.state_dict().items()}
                    else:
                        patience += 1
                        if patience >= 5:
                            break
            m.load_state_dict(best_state); m.eval()
            with torch.no_grad():
                oof[va_i] = m(X[va_i]).numpy()
            print(f"    seed{si} fold {fold}: "
                  f"r2={cv.score(y_raw[va_i], oof[va_i]*sd+mu)['r2']:.4f}", flush=True)
        oof_seeds.append(oof)

    pred = np.mean(oof_seeds, axis=0) * sd + mu
    r2 = cv.score(y_raw, pred)["r2"]
    print(f"  {target}: CNN OOF r2 = {r2:.4f}")
    np.save(C.RUNS / f"_cnn{n_folds}s{len(seeds)}_oof_{target}.npy", pred)
    return r2


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--seeds", type=int, default=1)
    a = ap.parse_args()
    seed_list = tuple(C.SEED + 100 * i for i in range(a.seeds))
    print(f"=== SMILES CNN ({a.epochs} ep, {a.folds} folds, {a.seeds} seed(s)) ===")
    res = {t: run_target(t, epochs=a.epochs, n_folds=a.folds, seeds=seed_list)
           for t in C.TARGET_TYPES}
    print(f"\n  official = {np.mean(list(res.values())):.4f}")
