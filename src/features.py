"""SMILES -> feature matrix, computed once per unique SMILES and cached.

Every block is computed once into a single parquet; experiments then select
blocks by column prefix rather than recomputing. Adding a block therefore costs
one featurization pass, and ablating costs nothing.

Polymer SMILES here are repeat units carrying exactly two '*' attachment
points. Dummy atoms break Gasteiger-charge and BCUT descriptors, so every
molecule is featurized twice: as given, and with '*' capped to carbon.
"""
import argparse
import warnings

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors, MACCSkeys
from rdkit.Chem import rdFingerprintGenerator as rfg

from . import config as C
from . import polymer
from .data import all_smiles

warnings.filterwarnings("ignore")
RDLogger.DisableLog("rdApp.*")

CACHE = C.FEAT / "features_v7.parquet"

# block name -> column prefix. Selection happens by prefix.
BLOCKS = {
    "desc_raw": "draw_",
    "desc_cap": "dcap_",
    "morgan2": "mg2_",
    "morgan3": "mg3_",
    "maccs": "maccs_",
    "avalon": "avl_",
    "rdkfp": "rdk_",
    "atompair": "ap_",
    "torsion": "tt_",
    "polymer": "poly_",
    "dimer": "dim_",
    "conj": "conj_",
    "decomp_bb": "bbd_",
    "decomp_sc": "scd_",
    "decomp_fp": "bbfp_",
    "bare_bb": "bare_",
    "large_sc": "lsc_",
    "dimer_bb": "dbb_",
}

# The current promoted feature set: what the shipped notebook uses.
# Keep this in sync with the notebook so ablations measure marginal value on
# top of what we actually run, not on top of an outdated baseline.
BASELINE_BLOCKS = ["desc_raw", "desc_cap", "morgan2", "maccs", "avalon", "polymer",
                   "decomp_bb"]

FP_SIZE = 2048
AVALON_SIZE = 1024

_GENS = None


def _gens():
    """Built lazily per process: generator objects are not picklable."""
    global _GENS
    if _GENS is None:
        _GENS = {
            "morgan2": rfg.GetMorganGenerator(radius=2, fpSize=FP_SIZE),
            "morgan3": rfg.GetMorganGenerator(radius=3, fpSize=FP_SIZE),
            "rdkfp": rfg.GetRDKitFPGenerator(fpSize=FP_SIZE),
            "atompair": rfg.GetAtomPairGenerator(fpSize=FP_SIZE),
            "torsion": rfg.GetTopologicalTorsionGenerator(fpSize=FP_SIZE),
        }
    return _GENS


def cap_wildcards(mol):
    """Replace '*' dummy atoms with carbon so charge descriptors resolve."""
    rw = Chem.RWMol(mol)
    for atom in rw.GetAtoms():
        if atom.GetAtomicNum() == 0:
            atom.SetAtomicNum(6)
            atom.SetNoImplicit(False)
            atom.SetNumExplicitHs(0)
    m = rw.GetMol()
    try:
        Chem.SanitizeMol(m)
    except Exception:
        return None
    return m


def _one(smi):
    """Feature vector for a single SMILES. Returns (values, names)."""
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return None
    # Canonical atom order before anything else. Several descriptors are
    # order-dependent (tied shortest paths round a ring, Gasteiger iteration
    # with dummy atoms, Ipc rounding); re-parsing from canonical SMILES makes
    # every feature a deterministic function of the molecule. Measured: tree
    # predictions moved up to 3% of target sd across randomised SMILES before
    # this, and exactly 0 after.
    mol = Chem.MolFromSmiles(Chem.MolToSmiles(mol))
    if mol is None:
        return None
    capped = cap_wildcards(mol)
    fp_mol = capped if capped is not None else mol
    g = _gens()
    vals, names = [], []

    d_raw = Descriptors.CalcMolDescriptors(mol)
    vals += list(d_raw.values()); names += [f"draw_{k}" for k in d_raw]

    if capped is not None:
        d_cap = Descriptors.CalcMolDescriptors(capped)
        vals += list(d_cap.values()); names += [f"dcap_{k}" for k in d_cap]
    else:
        vals += [np.nan] * len(d_raw); names += [f"dcap_{k}" for k in d_raw]

    for block, prefix in (("morgan2", "mg2_"), ("morgan3", "mg3_"),
                          ("rdkfp", "rdk_"), ("atompair", "ap_"),
                          ("torsion", "tt_")):
        v = g[block].GetCountFingerprintAsNumPy(fp_mol).astype(np.float32)
        vals += v.tolist(); names += [f"{prefix}{i}" for i in range(len(v))]

    maccs = np.array(MACCSkeys.GenMACCSKeys(fp_mol), dtype=np.float32)
    vals += maccs.tolist(); names += [f"maccs_{i}" for i in range(len(maccs))]

    from rdkit.Avalon import pyAvalonTools
    avl = np.array(pyAvalonTools.GetAvalonCountFP(fp_mol, nBits=AVALON_SIZE).ToList(),
                   dtype=np.float32)
    vals += avl.tolist(); names += [f"avl_{i}" for i in range(len(avl))]

    # Polymer-aware blocks are computed on the RAW mol: they need the '*'
    # attachment points that capping would destroy.
    poly_names, poly_vals = polymer.features(mol)
    vals += poly_vals; names += poly_names

    cvals, cnames = polymer.conjugation_features(mol)
    vals += cvals; names += cnames

    dvals, dnames = polymer.decomposition_features(mol)
    vals += dvals; names += dnames

    evals, enames = polymer.extended_features(mol)
    vals += evals; names += enames

    return vals, names


def build(force=False, n_jobs=-1):
    """Build (or load) the cached feature table indexed by SMILES."""
    if CACHE.exists() and not force:
        return pd.read_parquet(CACHE)

    smiles = all_smiles()
    print(f"featurizing {len(smiles)} unique SMILES across {len(BLOCKS)} blocks ...")
    out = Parallel(n_jobs=n_jobs, verbose=1)(delayed(_one)(s) for s in smiles)

    failed = [s for s, o in zip(smiles, out) if o is None]
    if failed:
        print(f"WARNING: {len(failed)} SMILES failed to parse")
    names = next(o[1] for o in out if o is not None)
    rows = [o[0] if o is not None else [np.nan] * len(names) for o in out]

    X = pd.DataFrame(rows, columns=names, index=smiles, dtype=np.float32)
    X = X.replace([np.inf, -np.inf], np.nan)
    # Drop constant columns; LightGBM keeps NaN natively so we do not impute.
    X = X[X.columns[X.nunique(dropna=False) > 1]]
    X.to_parquet(CACHE)
    print(f"features: {X.shape[0]} rows x {X.shape[1]} cols -> {CACHE}")
    for b, p in BLOCKS.items():
        print(f"    {b:10s} {p:7s} {sum(c.startswith(p) for c in X.columns):5d} cols")
    return X


# RDKit's Ipc grows exponentially with molecule size (values above 1e13 are
# routine) and its floating-point rounding depends on atom ordering, so two
# SMILES for the same molecule give slightly different Ipc. That noise crossed
# tree split thresholds and moved predictions by up to 3% of target sd across
# randomised SMILES. Dropped from every block: invariance is a scored theme.
NON_INVARIANT = ("_Ipc",)


def select(X, blocks):
    """Columns belonging to the named blocks, in BLOCKS order."""
    unknown = set(blocks) - set(BLOCKS)
    if unknown:
        raise KeyError(f"unknown blocks {unknown}; have {list(BLOCKS)}")
    prefixes = tuple(BLOCKS[b] for b in blocks)
    cols = [c for c in X.columns if c.startswith(prefixes) and not c.endswith(NON_INVARIANT)]
    return X[cols]


def for_rows(df, X=None, blocks=None):
    """Align the feature table to a dataframe's smiles column."""
    if X is None:
        X = build()
    if blocks is not None:
        X = select(X, blocks)
    return X.reindex(df.smiles.values).reset_index(drop=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    if a.smoke:
        smi = all_smiles()[:50]
        res = [_one(s) for s in smi]
        nfail = sum(r is None for r in res)
        print(f"smoke: {len(smi)} smiles, {nfail} failures, {len(res[0][0])} features each")
    else:
        build(force=a.force)
