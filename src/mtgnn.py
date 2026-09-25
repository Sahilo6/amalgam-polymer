"""Multi-task GNN across the electronic/optical cluster.

Six of the seven Round 3 targets are frontier-orbital quantities and correlate
strongly on molecules carrying both labels (egc-egb +0.93, eps-nc +0.92,
egc-nc -0.85). Five of them have ~220 training rows. One shared encoder with a
head per target lets the 220-row targets borrow the representation learned
from egc's 2,028 rows.

This is the opposite of Round 1, where multi-task across tg and egc lost 0.023:
those two share no physics. tg stays standalone here.

Fold alignment: each target keeps the SAME per-target fold split as the
single-task runs (cv.make_folds on that target's rows), so its OOF vector lines
up row-for-row with the single-task OOF and the two can be blended.
"""
import argparse

import numpy as np
import torch
import torch.nn as nn

from . import config as C
from . import cv
from .data import load_raw, subset
from .gnn import MPNN, N_FEAT, collate, mol_to_graph
import src.gnn as G

torch.set_num_threads(1)


class MTMPNN(nn.Module):
    """The Round 1 encoder with one linear head per target."""

    def __init__(self, targets, hidden=128, layers=3, dropout=0.1):
        super().__init__()
        base = MPNN(hidden=hidden, layers=layers, dropout=dropout)
        self.inp, self.msg, self.upd, self.norm = base.inp, base.msg, base.upd, base.norm
        self.trunk = nn.Sequential(nn.Linear(hidden * 2, hidden), nn.ReLU(), nn.Dropout(dropout))
        self.heads = nn.ModuleDict({t: nn.Linear(hidden, 1) for t in targets})
        self.targets = list(targets)

    def embed(self, x, edge, batch, n_graphs, edge_attr=None):
        h = torch.relu(self.inp(x))
        src, dst = edge
        if edge_attr is None:
            edge_attr = torch.zeros(len(src), G.N_EDGE)
        for msg, upd, norm in zip(self.msg, self.upd, self.norm):
            agg = torch.zeros_like(h).index_add_(0, dst, msg(torch.cat([h[src], edge_attr], dim=1)))
            h = norm(upd(agg, h))
        mean = torch.zeros(n_graphs, h.shape[1]).index_add_(0, batch, h)
        cnt = torch.zeros(n_graphs).index_add_(0, batch, torch.ones(len(batch))).clamp(min=1).unsqueeze(1)
        mx = torch.full((n_graphs, h.shape[1]), -1e9).index_reduce_(0, batch, h, "amax", include_self=True)
        return self.trunk(torch.cat([mean / cnt, mx], 1))

    def forward(self, x, edge, batch, n_graphs, edge_attr, task_idx):
        """task_idx: LongTensor of head index per graph. Returns one value per graph."""
        z = self.embed(x, edge, batch, n_graphs, edge_attr)
        outs = torch.stack([self.heads[t](z).squeeze(-1) for t in self.targets], 1)   # (n, T)
        return outs.gather(1, task_idx.unsqueeze(1)).squeeze(1)


def run(targets, epochs=120, hidden=128, bs=64, lr=1e-3, n_folds=5, seeds=(C.SEED,), tag="mt",
        init=None):
    """`init`: path to a pretrained encoder state (from src/pretrain.py). Loaded
    into every fold's model before fine-tuning; heads start fresh."""
    G.PLAIN = True
    init_state = torch.load(init) if init else None
    tr, te = load_raw()
    T = {t: i for i, t in enumerate(targets)}

    # one flat table of electronic rows, each carrying its target's fold id
    rows = []
    for t in targets:
        d = subset(tr, t)
        fold_of = np.empty(len(d), int)
        for k, (_, va) in enumerate(cv.make_folds(len(d), n_splits=n_folds)):
            fold_of[va] = k
        mu, sd = d.target.mean(), d.target.std()
        for i, (s, y) in enumerate(zip(d.smiles, d.target)):
            rows.append((s, t, T[t], (y - mu) / sd, y, fold_of[i], i, mu, sd))
    smi = [r[0] for r in rows]
    graphs = [mol_to_graph(s) for s in smi]
    ok = np.array([g is not None for g in graphs])
    task = torch.tensor([r[2] for r in rows]); yz = torch.tensor([r[3] for r in rows], dtype=torch.float32)
    fold = np.array([r[5] for r in rows]); y_raw = np.array([r[4] for r in rows])
    mu = np.array([r[7] for r in rows]); sd = np.array([r[8] for r in rows])
    print(f"  {len(rows)} rows over {len(targets)} targets; graphs ok: {ok.sum()}")

    oof_seeds = []
    for si, base_seed in enumerate(seeds):
        oof = np.full(len(rows), np.nan)
        for k in range(n_folds):
            tr_i = np.where((fold != k) & ok)[0]; va_i = np.where((fold == k) & ok)[0]
            torch.manual_seed(base_seed + k); rng = np.random.default_rng(base_seed + k)
            model = MTMPNN(targets, hidden=hidden)
            if init_state is not None:
                missing, unexpected = model.load_state_dict(init_state, strict=False)
                assert not unexpected, unexpected
            opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
            sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=epochs * max(1, len(tr_i) // bs))
            best, best_state, patience = 1e9, None, 0
            for ep in range(epochs):
                model.train(); perm = rng.permutation(tr_i)
                for b0 in range(0, len(perm), bs):
                    b = perm[b0:b0 + bs]
                    if len(b) < 2:
                        continue
                    x, e, bt, n, ea = collate(graphs, b)
                    loss = nn.functional.mse_loss(model(x, e, bt, n, ea, task[b]), yz[b])
                    opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 5.0); opt.step()
                    if sched.last_epoch < sched.total_steps - 1:
                        sched.step()
                if ep % 5 == 4 or ep == epochs - 1:
                    model.eval()
                    with torch.no_grad():
                        x, e, bt, n, ea = collate(graphs, va_i)
                        p = model(x, e, bt, n, ea, task[va_i]).numpy()
                    mse = ((p - yz[va_i].numpy()) ** 2).mean()
                    if mse < best - 1e-5:
                        best, patience = mse, 0
                        best_state = {kk: v.clone() for kk, v in model.state_dict().items()}
                    else:
                        patience += 1
                        if patience >= 6:
                            break
            model.load_state_dict(best_state); model.eval()
            with torch.no_grad():
                x, e, bt, n, ea = collate(graphs, va_i)
                oof[va_i] = model(x, e, bt, n, ea, task[va_i]).numpy()
            print(f"    seed{si} fold {k}: done", flush=True)
        oof_seeds.append(oof)

    oof = np.nanmean(np.vstack(oof_seeds), 0) * sd + mu
    res = {}
    for t in targets:
        m = np.array([r[1] == t for r in rows])
        idx = np.array([r[6] for r in rows])[m]
        pred = np.full(m.sum(), np.nan); pred[idx] = oof[m]
        pred = np.nan_to_num(pred, nan=np.nanmean(y_raw[m]))
        yt = np.full(m.sum(), np.nan); yt[idx] = y_raw[m]
        res[t] = cv.score(yt, pred)["r2"]
        np.save(C.RUNS / f"_{tag}_oof_{t}.npy", pred)
        print(f"  {t:4s} multi-task r2 = {res[t]:.4f}")
    print(f"\n  mean over cluster = {np.mean(list(res.values())):.4f}")
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=120)
    ap.add_argument("--seeds", type=int, default=1)
    ap.add_argument("--tag", default="mt")
    ap.add_argument("--init", default=None, help="pretrained encoder .pt")
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--folds", type=int, default=C.N_FOLDS)
    a = ap.parse_args()
    print(f"=== multi-task GNN over {C.ELECTRONIC} ({a.epochs} ep, {a.seeds} seed(s)) ===")
    run(C.ELECTRONIC, epochs=a.epochs, seeds=tuple(C.SEED + 100 * i for i in range(a.seeds)), tag=a.tag,
        init=a.init, lr=a.lr, n_folds=a.folds)
