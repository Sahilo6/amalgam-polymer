"""Self-supervised pretraining of the GNN encoder on PI1M.

Masked-atom prediction: hide the identity of 15% of atoms (atomic-number one-hot,
hybridisation and mass are zeroed; degree, charge, aromaticity, ring membership
and H-count stay) and train the encoder to recover the atomic number from the
node embedding. No labels needed, so all 995,799 PI1M repeat units are usable.

Rule 6.2.4 bans uploading weights, so in the notebook this runs from scratch
before fine-tuning. Locally it exists to measure whether it helps at all.
"""
import argparse
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from . import config as C
import src.gnn as G
from .gnn import ATOMS, MPNN, collate, mol_to_graph

torch.set_num_threads(1)
N_CLASSES = len(ATOMS) + 1          # ATOMS one-hot + "other"
MASK_COLS = list(range(0, 16)) + [21]   # atom one-hot (12) + hybridisation (4) + mass


class MaskedAtomModel(nn.Module):
    def __init__(self, hidden=128, layers=3):
        super().__init__()
        base = MPNN(hidden=hidden, layers=layers)
        self.inp, self.msg, self.upd, self.norm = base.inp, base.msg, base.upd, base.norm
        self.node_head = nn.Linear(hidden, N_CLASSES)

    def node_embed(self, x, edge, edge_attr):
        h = torch.relu(self.inp(x))
        src, dst = edge
        for msg, upd, norm in zip(self.msg, self.upd, self.norm):
            agg = torch.zeros_like(h).index_add_(0, dst, msg(torch.cat([h[src], edge_attr], 1)))
            h = norm(upd(agg, h))
        return h

    def encoder_state(self):
        return {k: v for k, v in self.state_dict().items() if not k.startswith("node_head")}


def pretrain(n_mols=100_000, epochs=20, bs=128, lr=1e-3, seed=C.SEED, out=None):
    torch.manual_seed(seed); rng = np.random.default_rng(seed)
    G.PLAIN = True
    df = pd.read_csv("/Users/sahilsadhwani/Downloads/PI1M.csv")
    smi = df.SMILES.sample(n=n_mols, random_state=seed).tolist()
    t0 = time.time()
    graphs = [g for g in (mol_to_graph(s) for s in smi) if g is not None]
    print(f"  {len(graphs)} graphs built in {time.time()-t0:.0f}s")
    # class target per atom = index into ATOMS, or N_CLASSES-1 for other
    targets = []
    for x, _, _ in graphs:
        oh = x[:, :len(ATOMS)]
        cls = torch.where(oh.sum(1) > 0, oh.argmax(1), torch.full((len(x),), len(ATOMS)))
        targets.append(cls)

    model = MaskedAtomModel()
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=epochs * max(1, len(graphs) // bs))
    idx_all = np.arange(len(graphs))
    for ep in range(epochs):
        model.train(); perm = rng.permutation(idx_all); tot, n, correct, seen = 0.0, 0, 0, 0
        for b0 in range(0, len(perm), bs):
            b = perm[b0:b0 + bs]
            x, e, bt, ng, ea = collate(graphs, b)
            y = torch.cat([targets[i] for i in b])
            mask = torch.rand(len(x)) < 0.15
            if mask.sum() == 0:
                continue
            xm = x.clone(); xm[mask.nonzero().squeeze(1)[:, None], torch.tensor(MASK_COLS)] = 0.0
            logits = model.node_head(model.node_embed(xm, e, ea))[mask]
            loss = nn.functional.cross_entropy(logits, y[mask])
            opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 5.0); opt.step()
            if sched.last_epoch < sched.total_steps - 1:
                sched.step()
            tot += loss.item() * mask.sum().item(); n += mask.sum().item()
            correct += (logits.argmax(1) == y[mask]).sum().item(); seen += mask.sum().item()
        print(f"  epoch {ep+1:2d}/{epochs}  masked-atom loss {tot/n:.3f}  acc {correct/seen:.3f}  ({time.time()-t0:.0f}s)", flush=True)
    out = out or (C.RUNS / f"pretrained_{n_mols//1000}k_{epochs}ep.pt")
    torch.save(model.encoder_state(), out)
    print(f"  encoder saved -> {out}")
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--mols", type=int, default=100_000)
    ap.add_argument("--epochs", type=int, default=20)
    a = ap.parse_args()
    print(f"=== masked-atom pretraining on {a.mols:,} PI1M repeat units, {a.epochs} epochs ===")
    pretrain(n_mols=a.mols, epochs=a.epochs)
