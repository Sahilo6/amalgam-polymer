"""Message-passing GNN, plain PyTorch (no torch-geometric on the Kaggle image).

The one predictor that sees something the trees cannot: raw molecular topology,
rather than topology hashed into fingerprint bits. Trained from scratch, since
pretrained weights are banned (rule 6.2.1).

Expectation is modest. ExtraTrees is a genuinely different algorithm scoring
0.9001 solo and added exactly 0.0000 to the blend, which suggests diversity is
not our bottleneck. This is worth one day because it is the last idea of a
different KIND, not because it is likely to win.
"""
import argparse

import numpy as np
import torch
import torch.nn as nn
from rdkit import Chem, RDLogger

from . import config as C
from . import cv, polymer
from .data import load_raw, subset

RDLogger.DisableLog("rdApp.*")
torch.manual_seed(C.SEED)

# Single-threaded on purpose. Parallel index_add_/index_reduce_ sum atomics in
# varying order, which made two identical runs differ by up to 0.18 per
# prediction. Rule 7.2 has the hosts re-run this notebook and voids the
# submission if the score does not reproduce, so exact determinism is required.
# These graphs are small enough that threading bought nothing: 1.7s vs 1.8s.
torch.set_num_threads(1)

ATOMS = [6, 7, 8, 9, 14, 15, 16, 17, 35, 53, 0]   # 0 = the '*' attachment point
HYB = [Chem.HybridizationType.SP, Chem.HybridizationType.SP2,
       Chem.HybridizationType.SP3, Chem.HybridizationType.SP3D]


BOND_TYPES = [Chem.BondType.SINGLE, Chem.BondType.DOUBLE,
              Chem.BondType.TRIPLE, Chem.BondType.AROMATIC]


def atom_features(a, poly=None):
    """Per-atom features. `poly` carries the polymer context for this molecule:
    (backbone atom set, attachment indices, distance-to-attachment array).

    Telling the trees about backbone structure was the single biggest win of the
    competition (+0.0056). The GNN was never told, so it saw these as ordinary
    molecules rather than chain repeat units.
    """
    z = a.GetAtomicNum()
    f = [1.0 if z == x else 0.0 for x in ATOMS] + [1.0 if z not in ATOMS else 0.0]
    f += [1.0 if a.GetHybridization() == h else 0.0 for h in HYB]
    f += [a.GetDegree() / 4.0, a.GetFormalCharge(), float(a.GetIsAromatic()),
          float(a.IsInRing()), a.GetTotalNumHs() / 4.0, a.GetMass() / 200.0]

    if poly is None:
        f += [0.0, 0.0, 0.0, 0.0, 0.0]
    else:
        bb, attach, dist = poly
        i = a.GetIdx()
        f += [
            1.0 if i in bb else 0.0,            # on the chain backbone
            1.0 if i in attach else 0.0,        # is an attachment point itself
            min(dist[i], 20.0) / 20.0,          # graph distance to nearest '*'
            float(a.GetIsAromatic() and a.IsInRing()),
            min(_ring_size(a), 8) / 8.0,
        ]
    return f


def _ring_size(a):
    if not a.IsInRing():
        return 0
    m = a.GetOwningMol()
    sizes = [len(r) for r in m.GetRingInfo().AtomRings() if a.GetIdx() in r]
    return min(sizes) if sizes else 0


def bond_features(b):
    f = [1.0 if b.GetBondType() == t else 0.0 for t in BOND_TYPES]
    f += [float(b.GetIsConjugated()), float(b.IsInRing())]
    return f


N_FEAT = 22 + 5
N_EDGE = len(BOND_TYPES) + 2


PLAIN = False     # when True, omit polymer features so this matches the notebook


def _polymer_context(m):
    if PLAIN:
        return None
    """Backbone set, attachment indices and per-atom distance to the nearest
    attachment point. Reuses the ring-complete backbone from src/polymer.py."""
    try:
        attach = set(polymer.attachment_points(m))
        bb = polymer.backbone_atoms_ring_complete(m)
        bb = bb if bb is not None else set()
    except Exception:
        return None
    if not attach:
        return None
    n = m.GetNumAtoms()
    dist = np.full(n, 99.0)
    dmat = Chem.GetDistanceMatrix(m)
    for i in range(n):
        dist[i] = min(dmat[i][j] for j in attach)
    return bb, attach, dist


def mol_to_graph(smi):
    m = Chem.MolFromSmiles(smi)
    if m is None:
        return None
    poly = _polymer_context(m)
    x = torch.tensor([atom_features(a, poly) for a in m.GetAtoms()], dtype=torch.float32)
    src, dst, ef = [], [], []
    for b in m.GetBonds():
        i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        bf = bond_features(b)
        src += [i, j]; dst += [j, i]; ef += [bf, bf]
    if not src:
        src, dst, ef = [0], [0], [[0.0] * N_EDGE]
    return (x, torch.tensor([src, dst], dtype=torch.long),
            torch.tensor(ef, dtype=torch.float32))


class MPNN(nn.Module):
    def __init__(self, hidden=128, layers=3, dropout=0.1):
        super().__init__()
        self.inp = nn.Linear(N_FEAT, hidden)
        # messages now condition on the bond as well as the neighbour atom
        self.msg = nn.ModuleList(nn.Linear(hidden + N_EDGE, hidden) for _ in range(layers))
        self.upd = nn.ModuleList(nn.GRUCell(hidden, hidden) for _ in range(layers))
        self.norm = nn.ModuleList(nn.LayerNorm(hidden) for _ in range(layers))
        self.head = nn.Sequential(nn.Linear(hidden * 2, hidden), nn.ReLU(),
                                  nn.Dropout(dropout), nn.Linear(hidden, 1))

    def forward(self, x, edge, batch, n_graphs, edge_attr=None):
        h = torch.relu(self.inp(x))
        src, dst = edge
        if edge_attr is None:
            edge_attr = torch.zeros(len(src), N_EDGE)
        for msg, upd, norm in zip(self.msg, self.upd, self.norm):
            agg = torch.zeros_like(h).index_add_(
                0, dst, msg(torch.cat([h[src], edge_attr], dim=1)))
            h = norm(upd(agg, h))
        # concat mean and max pooling per graph
        mean = torch.zeros(n_graphs, h.shape[1], device=h.device).index_add_(0, batch, h)
        cnt = torch.zeros(n_graphs, device=h.device).index_add_(
            0, batch, torch.ones(len(batch), device=h.device)).clamp(min=1).unsqueeze(1)
        mean = mean / cnt
        mx = torch.full((n_graphs, h.shape[1]), -1e9, device=h.device)
        mx = mx.index_reduce_(0, batch, h, "amax", include_self=True)
        return self.head(torch.cat([mean, mx], 1)).squeeze(-1)


def collate(graphs, idx):
    xs, edges, efs, batch, off = [], [], [], [], 0
    for k, i in enumerate(idx):
        x, e, ef = graphs[i]
        xs.append(x); edges.append(e + off); efs.append(ef)
        batch.append(torch.full((len(x),), k, dtype=torch.long))
        off += len(x)
    return (torch.cat(xs), torch.cat(edges, 1), torch.cat(batch), len(idx),
            torch.cat(efs))


TAG = ""          # set from the CLI so variants cannot overwrite each other


def run_target(target, epochs=120, hidden=128, bs=64, lr=1e-3, n_folds=5,
               seeds=(C.SEED,), layers=3, dropout=0.1):
    """Train over one or more seeds and average the predictions.

    Seed averaging did nothing for the GBDTs (+0.0000 measured), but neural nets
    have far higher variance across initialisations, so this is a different
    proposition rather than a re-test of a dead idea.
    """
    tr, te = load_raw()
    tr_t = subset(tr, target)
    graphs = [mol_to_graph(s) for s in tr_t.smiles]
    ok = np.array([g is not None for g in graphs])
    y_raw = tr_t.target.values
    mu, sd = y_raw[ok].mean(), y_raw[ok].std()
    y = (y_raw - mu) / sd

    oof_seeds = []
    for seed_i, base_seed in enumerate(seeds):
      oof = np.full(len(y), np.nan)
      for fold, (tr_i, va_i) in enumerate(cv.make_folds(len(y), n_splits=n_folds)):
        tr_i = tr_i[ok[tr_i]]; va_i_ok = va_i[ok[va_i]]
        # Re-seed per fold. Rule 7.2 has the hosts re-run this notebook and voids
        # the submission if the score does not reproduce, so every source of
        # randomness (weight init, batch shuffling, dropout) must be pinned.
        torch.manual_seed(base_seed + fold)
        rng = np.random.default_rng(base_seed + fold)
        model = MPNN(hidden=hidden, layers=layers, dropout=dropout)
        opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
        sched = torch.optim.lr_scheduler.OneCycleLR(
            opt, max_lr=lr, total_steps=epochs * max(1, len(tr_i) // bs))
        yt = torch.tensor(y, dtype=torch.float32)
        best, best_state, patience = 1e9, None, 0
        for ep in range(epochs):
            model.train()
            perm = rng.permutation(tr_i)
            for k in range(0, len(perm), bs):
                b = perm[k:k + bs]
                if len(b) < 2:
                    continue
                x, e, bt, n, ea = collate(graphs, b)
                loss = nn.functional.mse_loss(model(x, e, bt, n, ea), yt[b])
                opt.zero_grad(); loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                opt.step()
                if sched.last_epoch < sched.total_steps - 1:
                    sched.step()
            if ep % 5 == 4 or ep == epochs - 1:
                model.eval()
                with torch.no_grad():
                    x, e, bt, n, ea = collate(graphs, va_i_ok)
                    p = model(x, e, bt, n, ea).numpy()
                mse = ((p - y[va_i_ok]) ** 2).mean()
                if mse < best - 1e-5:
                    best, best_state, patience = mse, {k: v.clone() for k, v in model.state_dict().items()}, 0
                else:
                    patience += 1
                    if patience >= 6:
                        break
        model.load_state_dict(best_state)
        model.eval()
        with torch.no_grad():
            x, e, bt, n, ea = collate(graphs, va_i_ok)
            oof[va_i_ok] = model(x, e, bt, n, ea).numpy()
        print(f"    seed{seed_i} fold {fold}: "
              f"r2={cv.score(y_raw[va_i_ok], oof[va_i_ok]*sd+mu)['r2']:.4f}", flush=True)
      oof_seeds.append(oof)

    oof = np.nanmean(np.vstack(oof_seeds), axis=0)
    pred = oof * sd + mu
    m = ~np.isnan(pred)
    r2 = cv.score(y_raw[m], pred[m])["r2"]
    print(f"  {target}: GNN OOF r2 = {r2:.4f}")
    # fold count in the filename: a 3-fold run once silently overwrote the
    # 5-fold predictions the blend depended on.
    np.save(C.RUNS / f"_gnn{n_folds}{TAG}_oof_{target}.npy",
            np.nan_to_num(pred, nan=y_raw.mean()))
    return r2


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=120)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--seeds", type=int, default=1)
    ap.add_argument("--tag", default="")
    ap.add_argument("--plain", action="store_true",
                    help="omit polymer node/edge features (matches the notebook)")
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--layers", type=int, default=3)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--dropout", type=float, default=0.1)
    a = ap.parse_args()
    PLAIN = a.plain
    globals()["PLAIN"] = a.plain
    TAG = a.tag
    globals()["TAG"] = a.tag
    seed_list = tuple(C.SEED + 100 * i for i in range(a.seeds))
    print(f"=== GNN (feats={N_FEAT}+{N_EDGE}e, {a.epochs} ep, {a.folds} folds, "
          f"{a.seeds} seed(s), tag='{a.tag}') ===")
    res = {t: run_target(t, epochs=a.epochs, n_folds=a.folds, seeds=seed_list,
                         hidden=a.hidden, layers=a.layers, lr=a.lr, bs=a.bs,
                         dropout=a.dropout)
           for t in C.TARGET_TYPES}
    print(f"\n  official = {np.mean(list(res.values())):.4f}")
