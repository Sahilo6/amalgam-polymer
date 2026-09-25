"""Polymer-aware features.

Generic descriptors treat each row as a small molecule. They are repeat units
with exactly two '*' attachment points, and two physical facts follow:

  Tg  is governed by chain stiffness  -> the backbone path between the
      attachment points, its rotatable bonds, rings and side-chain bulk.
  Egc is governed by conjugation ALONG THE CHAIN -> which a single repeat unit
      cannot express, so descriptors are also computed on a dimer built by
      joining two copies at their attachment points.

Backbone length alone correlates +0.47 with Tg, which no existing block
captures directly.
"""
import numpy as np
from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors, rdMolDescriptors, rdmolops

RDLogger.DisableLog("rdApp.*")


def attachment_points(mol):
    return [a.GetIdx() for a in mol.GetAtoms() if a.GetAtomicNum() == 0]


def backbone_path(mol):
    """Atom indices on the shortest path between the two attachment points."""
    d = attachment_points(mol)
    if len(d) != 2:
        return None
    try:
        return list(rdmolops.GetShortestPath(mol, d[0], d[1]))
    except Exception:
        return None


def make_dimer(mol):
    """Two repeat units joined head-to-tail, leaving two attachment points."""
    a, b = Chem.Mol(mol), Chem.Mol(mol)
    da, db = attachment_points(a), attachment_points(b)
    if len(da) != 2 or len(db) != 2:
        return None
    a.GetAtomWithIdx(da[0]).SetAtomMapNum(1)
    a.GetAtomWithIdx(da[1]).SetAtomMapNum(2)
    b.GetAtomWithIdx(db[0]).SetAtomMapNum(2)
    b.GetAtomWithIdx(db[1]).SetAtomMapNum(3)
    try:
        d = rdmolops.molzip(a, b)
        Chem.SanitizeMol(d)
        return d
    except Exception:
        return None


def _longest_true_run(flags):
    best = cur = 0
    for f in flags:
        cur = cur + 1 if f else 0
        best = max(best, cur)
    return best


def backbone_scalars(mol):
    """Topology of the chain itself, separated from its side groups."""
    names = ["bb_len", "bb_frac", "side_atoms", "side_frac", "bb_rot", "bb_ring_atoms",
             "bb_aromatic", "bb_hetero", "bb_conj_bonds", "bb_conj_frac",
             "bb_max_conj_run", "bb_sp3_frac", "n_side_chains", "max_side_len",
             "side_mass_frac", "bb_double_bonds", "flexibility", "bb_branch_pts"]
    path = backbone_path(mol)
    if path is None:
        return [np.nan] * len(names), names

    n_atoms = mol.GetNumAtoms()
    bb = set(path)
    bb_atoms = [mol.GetAtomWithIdx(i) for i in path]

    # bonds along the backbone, in order
    bonds = [mol.GetBondBetweenAtoms(path[i], path[i + 1]) for i in range(len(path) - 1)]
    bonds = [b for b in bonds if b is not None]
    conj = [b.GetIsConjugated() for b in bonds]

    rot = sum(1 for b in bonds
              if b.GetBondType() == Chem.BondType.SINGLE
              and not b.IsInRing()
              and b.GetBeginAtom().GetDegree() > 1 and b.GetEndAtom().GetDegree() > 1)

    # side chains: connected components once the backbone is removed
    side_sizes, seen = [], set()
    for a in bb_atoms:
        for nb in a.GetNeighbors():
            if nb.GetIdx() in bb or nb.GetIdx() in seen:
                continue
            stack, comp = [nb.GetIdx()], set()
            while stack:
                cur = stack.pop()
                if cur in comp or cur in bb:
                    continue
                comp.add(cur)
                stack += [x.GetIdx() for x in mol.GetAtomWithIdx(cur).GetNeighbors()]
            seen |= comp
            if comp:
                side_sizes.append(len(comp))

    side_atoms = n_atoms - len(path)
    bb_mass = sum(a.GetMass() for a in bb_atoms)
    tot_mass = sum(a.GetMass() for a in mol.GetAtoms()) or 1.0

    vals = [
        len(path),
        len(path) / n_atoms if n_atoms else np.nan,
        side_atoms,
        side_atoms / n_atoms if n_atoms else np.nan,
        rot,
        sum(1 for a in bb_atoms if a.IsInRing()),
        sum(1 for a in bb_atoms if a.GetIsAromatic()),
        sum(1 for a in bb_atoms if a.GetAtomicNum() not in (6, 0)),
        sum(conj),
        (sum(conj) / len(conj)) if conj else np.nan,
        _longest_true_run(conj),
        (sum(1 for a in bb_atoms if a.GetHybridization() == Chem.HybridizationType.SP3)
         / len(path)) if path else np.nan,
        len(side_sizes),
        max(side_sizes) if side_sizes else 0,
        1.0 - bb_mass / tot_mass,
        sum(1 for b in bonds if b.GetBondType() == Chem.BondType.DOUBLE),
        rdMolDescriptors.CalcNumRotatableBonds(mol) / max(n_atoms, 1),
        sum(1 for a in bb_atoms if a.GetDegree() > 2),
    ]
    return vals, names


def dimer_descriptors(mol):
    """RDKit descriptors on the dimer: captures chain periodicity that a single
    repeat unit cannot express (conjugation length above all)."""
    d = make_dimer(mol)
    ref = Descriptors.CalcMolDescriptors(Chem.MolFromSmiles("C"))
    names = [f"dim_{k}" for k in ref]
    if d is None:
        return [np.nan] * len(names), names
    try:
        vals = list(Descriptors.CalcMolDescriptors(d).values())
    except Exception:
        return [np.nan] * len(names), names
    return vals, names


def features(mol):
    """Both polymer blocks for one molecule: (values, names)."""
    v1, n1 = backbone_scalars(mol)
    v2, n2 = dimer_descriptors(mol)
    return [f"poly_{n}" for n in n1] + n2, v1 + v2


# --- conjugation across the chain -------------------------------------------
# Targeted at the hardest egc rows: short, rigid, conjugated backbones where the
# band gap is set by conjugation length. A single repeat unit understates that
# length, because in the real polymer conjugation continues across the linkage.
# These features ask directly whether it does.

CONJ_NAMES = ["mono_conj_run", "dimer_conj_run", "trimer_conj_run",
              "conj_extends_d", "conj_extends_t", "conj_growth_ratio",
              "attach_aromatic", "attach_sp2", "attach_conj",
              "dimer_arom_rings", "arom_ring_growth", "dimer_max_conj_frac"]


def _max_conj_run_backbone(mol):
    """Longest run of consecutive conjugated bonds along the backbone."""
    path = backbone_path(mol)
    if path is None or len(path) < 2:
        return 0
    bonds = [mol.GetBondBetweenAtoms(path[i], path[i + 1]) for i in range(len(path) - 1)]
    return _longest_true_run([b.GetIsConjugated() for b in bonds if b is not None])


def make_trimer(mol):
    d = make_dimer(mol)
    if d is None:
        return None
    a, b = Chem.Mol(d), Chem.Mol(mol)
    da, db = attachment_points(a), attachment_points(b)
    if len(da) != 2 or len(db) != 2:
        return None
    a.GetAtomWithIdx(da[0]).SetAtomMapNum(1); a.GetAtomWithIdx(da[1]).SetAtomMapNum(2)
    b.GetAtomWithIdx(db[0]).SetAtomMapNum(2); b.GetAtomWithIdx(db[1]).SetAtomMapNum(3)
    try:
        t = rdmolops.molzip(a, b); Chem.SanitizeMol(t); return t
    except Exception:
        return None


def conjugation_features(mol):
    """(values, names) - how conjugation behaves when the chain is extended."""
    mono = _max_conj_run_backbone(mol)
    dim, tri = make_dimer(mol), make_trimer(mol)
    d_run = _max_conj_run_backbone(dim) if dim is not None else np.nan
    t_run = _max_conj_run_backbone(tri) if tri is not None else np.nan

    # If conjugation stopped at the linkage the dimer run would be ~the monomer's.
    # Exceeding it means the chain conjugates through, which lowers the band gap.
    extends_d = (d_run - mono) if dim is not None else np.nan
    extends_t = (t_run - mono) if tri is not None else np.nan
    growth = (d_run / mono) if (dim is not None and mono > 0) else np.nan

    ats = [mol.GetAtomWithIdx(i) for i in attachment_points(mol)]
    nbrs = [n for a in ats for n in a.GetNeighbors()]
    attach_arom = sum(1 for n in nbrs if n.GetIsAromatic())
    attach_sp2 = sum(1 for n in nbrs if n.GetHybridization() == Chem.HybridizationType.SP2)
    attach_conj = sum(1 for a in ats for b in a.GetBonds() if b.GetIsConjugated())

    mono_rings = rdMolDescriptors.CalcNumAromaticRings(mol)
    if dim is not None:
        d_rings = rdMolDescriptors.CalcNumAromaticRings(dim)
        ring_growth = d_rings - 2 * mono_rings      # >0 means a ring formed across the join
        d_bonds = dim.GetNumBonds() or 1
        d_conj_frac = sum(1 for b in dim.GetBonds() if b.GetIsConjugated()) / d_bonds
    else:
        d_rings = ring_growth = d_conj_frac = np.nan

    vals = [mono, d_run, t_run, extends_d, extends_t, growth,
            attach_arom, attach_sp2, attach_conj,
            d_rings, ring_growth, d_conj_frac]
    return vals, [f"conj_{n}" for n in CONJ_NAMES]


# --- backbone / side-chain decomposition -------------------------------------
# The 18 backbone scalars were the biggest win so far, but they only summarise
# the chain. The model never sees the backbone's actual STRUCTURE. Fingerprinting
# the backbone and side chains separately is new information, not a
# recombination of what the whole-molecule features already encode.

_DECOMP_FP_BITS = 1024
_DFPGEN = None


def _decomp_gen():
    global _DFPGEN
    if _DFPGEN is None:
        from rdkit.Chem import rdFingerprintGenerator as _rfg
        _DFPGEN = _rfg.GetMorganGenerator(radius=2, fpSize=_DECOMP_FP_BITS)
    return _DFPGEN


def backbone_atoms_ring_complete(mol):
    """Backbone path expanded so any ring it passes through is included whole.

    Chemically right - a phenylene in the chain is a backbone unit, not a side
    group - and it keeps the extracted fragment sanitisable. Naive path-only
    extraction leaves broken aromatic rings and fails on ~70% of molecules.
    """
    path = backbone_path(mol)
    if path is None:
        return None
    bb = set(path)
    rings = mol.GetRingInfo().AtomRings()
    changed = True
    while changed:
        changed = False
        for ring in rings:
            r = set(ring)
            if bb & r and not r <= bb:
                bb |= r
                changed = True
    return bb


def _submol(mol, keep):
    rw = Chem.RWMol(mol)
    for i in sorted((a.GetIdx() for a in mol.GetAtoms() if a.GetIdx() not in keep),
                    reverse=True):
        rw.RemoveAtom(i)
    m = rw.GetMol()
    try:
        Chem.SanitizeMol(m)
        return m
    except Exception:
        return None


def decomposition_features(mol):
    """(values, names): descriptors on the backbone and side chains separately,
    plus a Morgan fingerprint of the backbone alone."""
    ref = Descriptors.CalcMolDescriptors(Chem.MolFromSmiles("C"))
    names = ([f"bbd_{k}" for k in ref] + [f"scd_{k}" for k in ref]
             + [f"bbfp_{i}" for i in range(_DECOMP_FP_BITS)])
    n_desc = len(ref)

    keep = backbone_atoms_ring_complete(mol)
    if keep is None:
        return [np.nan] * len(names), names

    bb = _submol(mol, keep)
    side_idx = set(range(mol.GetNumAtoms())) - keep
    sc = _submol(mol, side_idx) if side_idx else None

    def desc(m):
        if m is None:
            return [np.nan] * n_desc
        try:
            return list(Descriptors.CalcMolDescriptors(m).values())
        except Exception:
            return [np.nan] * n_desc

    vals = desc(bb) + desc(sc)
    if bb is not None:
        vals += _decomp_gen().GetCountFingerprintAsNumPy(bb).astype(np.float32).tolist()
    else:
        vals += [np.nan] * _DECOMP_FP_BITS
    return vals, names


# --- extended chain decomposition -------------------------------------------
# Backbone descriptors gave +0.0025 and backbone scalars +0.0056, while every
# generic-chemistry idea gave ~0. These push further into the same family.
#
#   bare_   the chain with side groups replaced by hydrogen, i.e. the backbone
#           as an actual molecule rather than a fragment with dangling valences
#   lsc_    the LARGEST side chain on its own. Merging all side groups into one
#           fragment (decomp_sc) gave nothing, plausibly because a big pendant
#           group and several small ones average out to look identical.
#   dbb_    descriptors of the DIMER's backbone, where chain periodicity shows
#           up in a way one repeat unit cannot express.

EXT_FP_BITS = 0   # descriptors only: the backbone fingerprint measured -0.0005


def bare_backbone(mol):
    """Backbone with every side group replaced by hydrogen."""
    keep = backbone_atoms_ring_complete(mol)
    if keep is None:
        return None
    rw = Chem.RWMol(mol)
    for i in sorted((a.GetIdx() for a in mol.GetAtoms() if a.GetIdx() not in keep),
                    reverse=True):
        rw.RemoveAtom(i)
    m = rw.GetMol()
    for a in m.GetAtoms():
        a.SetNoImplicit(False)
        a.SetNumExplicitHs(0)
    try:
        Chem.SanitizeMol(m)
        return m
    except Exception:
        return None


def largest_side_chain(mol):
    """The biggest pendant group, as its own molecule."""
    keep = backbone_atoms_ring_complete(mol)
    if keep is None:
        return None
    side = set(range(mol.GetNumAtoms())) - keep
    if not side:
        return None
    seen, comps = set(), []
    for idx in side:
        if idx in seen:
            continue
        stack, comp = [idx], set()
        while stack:
            cur = stack.pop()
            if cur in comp or cur not in side:
                continue
            comp.add(cur)
            stack += [n.GetIdx() for n in mol.GetAtomWithIdx(cur).GetNeighbors()]
        seen |= comp
        comps.append(comp)
    if not comps:
        return None
    return _submol(mol, max(comps, key=len))


def dimer_backbone(mol):
    """Backbone of the dimer: the chain over two repeat units."""
    d = make_dimer(mol)
    if d is None:
        return None
    keep = backbone_atoms_ring_complete(d)
    return _submol(d, keep) if keep is not None else None


def extended_features(mol):
    ref = Descriptors.CalcMolDescriptors(Chem.MolFromSmiles("C"))
    n = len(ref)
    names = ([f"bare_{k}" for k in ref] + [f"lsc_{k}" for k in ref]
             + [f"dbb_{k}" for k in ref])

    def desc(m):
        if m is None:
            return [np.nan] * n
        try:
            return list(Descriptors.CalcMolDescriptors(m).values())
        except Exception:
            return [np.nan] * n

    return desc(bare_backbone(mol)) + desc(largest_side_chain(mol)) \
        + desc(dimer_backbone(mol)), names
