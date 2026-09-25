"""Polymer invariance, measured.

Two levels. (1) SMILES-string invariance: the same repeat unit written as a
different valid SMILES must give the same prediction. (2) Cut-point invariance:
the same polymer written from a different cut of the chain (*CC(*)c1ccccc1 vs
*C(c1ccccc1)C*) must give the same prediction. Canonicalisation handles (1)
only; (2) is the polymer-specific case.

Reports the maximum deviation, at the feature level (trees) and the prediction
level (GNN), so the report can state a number rather than a claim.
"""
import argparse

import numpy as np
import pandas as pd
import torch
from rdkit import Chem, RDLogger

from . import config as C, cv, features, polymer
import src.gnn as G
from .gnn import MPNN, collate, mol_to_graph

RDLogger.DisableLog("rdApp.*")
torch.set_num_threads(1)


def random_smiles(smi, k, seed=C.SEED):
    m = Chem.MolFromSmiles(smi)
    out, rng = set(), np.random.default_rng(seed)
    for _ in range(k * 4):
        s = Chem.MolToSmiles(m, doRandom=True, canonical=False)
        if s != smi:
            out.add(s)
        if len(out) >= k:
            break
    return sorted(out)


def rotations(smi):
    """Equivalent repeat units obtained by cutting the chain at a different
    backbone bond. Built from the dimer: any backbone bond of the dimer that lies
    between the two original units is a valid cut."""
    m = Chem.MolFromSmiles(smi)
    d = polymer.make_dimer(m)
    if d is None:
        return []
    path = polymer.backbone_path(d)
    if path is None:
        return []
    n_unit = m.GetNumAtoms() - 2          # heavy atoms per unit, excluding the two *
    outs = set()
    # walk the dimer backbone; cut at each bond, keep a fragment with n_unit atoms
    for i in range(1, len(path) - 1):
        a, b = path[i], path[i + 1]
        bond = d.GetBondBetweenAtoms(a, b)
        if bond is None or bond.IsInRing():
            continue
        frag = Chem.FragmentOnBonds(d, [bond.GetIdx()], addDummies=True)
        for piece in Chem.GetMolFrags(frag, asMols=True, sanitizeFrags=True):
            if sum(1 for at in piece.GetAtoms() if at.GetAtomicNum() > 0) == n_unit \
               and sum(1 for at in piece.GetAtoms() if at.GetAtomicNum() == 0) == 2:
                for at in piece.GetAtoms():
                    at.SetAtomMapNum(0); at.SetIsotope(0)
                outs.add(Chem.MolToSmiles(piece))
    outs.discard(Chem.MolToSmiles(m))
    return sorted(outs)


def feature_deviation(smis_by_mol):
    """Max abs difference in tree features across variants of the same molecule."""
    worst = 0.0
    for variants in smis_by_mol:
        rows = []
        for s in variants:
            r = features._one(s)
            if r is None:
                continue
            rows.append(np.nan_to_num(np.array(r[0], dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0))
        if len(rows) < 2:
            continue
        A = np.vstack(rows)
        worst = max(worst, float(np.abs(A - A[0]).max()))
    return worst


def gnn_deviation(smis_by_mol, target="egc"):
    """Train one GNN fold on `target`, predict every variant, report max spread."""
    G.PLAIN = True
    tr, _ = __import__("src.data", fromlist=["load_raw"]).load_raw()
    d = tr[tr.target_type == target].reset_index(drop=True)
    y = d.target.values; mu, sd = y.mean(), y.std()
    graphs = [mol_to_graph(s) for s in d.smiles]
    tr_i, _ = next(iter(cv.make_folds(len(y), n_splits=5)))
    torch.manual_seed(C.SEED); rng = np.random.default_rng(C.SEED)
    m = MPNN(); opt = torch.optim.AdamW(m.parameters(), lr=1e-3)
    yt = torch.tensor((y - mu) / sd, dtype=torch.float32)
    for ep in range(15):
        perm = rng.permutation(tr_i)
        for k in range(0, len(perm), 64):
            b = perm[k:k + 64]
            if len(b) < 2:
                continue
            x, e, bt, n, ea = collate(graphs, b)
            loss = torch.nn.functional.mse_loss(m(x, e, bt, n, ea), yt[b]); opt.zero_grad(); loss.backward(); opt.step()
    m.eval(); worst = 0.0
    with torch.no_grad():
        for variants in smis_by_mol:
            gs = [mol_to_graph(s) for s in variants]; gs = [g for g in gs if g is not None]
            if len(gs) < 2:
                continue
            x, e, bt, n, ea = collate(gs, range(len(gs)))
            p = m(x, e, bt, n, ea).numpy() * sd + mu
            worst = max(worst, float(np.abs(p - p[0]).max()))
    return worst, sd


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=150)
    ap.add_argument("--k", type=int, default=5)
    a = ap.parse_args()
    te = pd.read_csv(C.TEST_CSV)
    smis = te.smiles.sample(n=a.n, random_state=C.SEED).tolist()

    print(f"=== (1) SMILES-string invariance: {a.n} test molecules x {a.k} randomised SMILES each ===")
    string_sets = [[s] + random_smiles(s, a.k) for s in smis]
    nvar = sum(len(v) - 1 for v in string_sets)
    print(f"  {nvar} alternative SMILES generated")
    print(f"  tree-feature max deviation : {feature_deviation(string_sets):.2e}")
    dev, sd = gnn_deviation(string_sets)
    print(f"  GNN prediction max deviation: {dev:.2e} eV  ({dev/sd*100:.4f}% of target sd)")

    print(f"\n=== (2) cut-point invariance: same polymer, chain cut at a different bond ===")
    rot_sets = [[s] + rotations(s) for s in smis]
    have = [v for v in rot_sets if len(v) > 1]
    print(f"  {len(have)} of {a.n} molecules admit an alternative cut ({sum(len(v)-1 for v in have)} rotations)")
    if have:
        fd = feature_deviation(have)
        gd, sd = gnn_deviation(have)
        print(f"  tree-feature max deviation : {fd:.3f}   <- NOT invariant if > 0")
        print(f"  GNN prediction max deviation: {gd:.3f} eV ({gd/sd*100:.1f}% of target sd)")
        ex = have[0]; print(f"  example: {ex[0]}\n       vs  {ex[1]}")
