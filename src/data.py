"""Loading + the grouping rule that keeps CV honest."""
import pandas as pd
from . import config as C


def load_raw():
    tr = pd.read_csv(C.TRAIN_CSV)
    te = pd.read_csv(C.TEST_CSV)
    return tr, te


def all_smiles():
    tr, te = load_raw()
    return pd.Index(pd.concat([tr.smiles, te.smiles]).unique(), name="smiles")


def subset(df, target_type):
    """Rows for one property, index reset so positions line up with features."""
    return df[df.target_type == target_type].reset_index(drop=True)
