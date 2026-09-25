"""Uniform wrappers so cv.run_cv can drive any library the same way."""
import numpy as np

from . import config as C

# Tree models take NaN natively; the linear model needs it filled.
_LGB_BASE = dict(
    n_estimators=4000, learning_rate=0.03, num_leaves=64, min_child_samples=20,
    colsample_bytree=0.6, subsample=0.8, subsample_freq=1,
    reg_lambda=1.0, verbose=-1, n_jobs=-1,
)
_XGB_BASE = dict(
    n_estimators=4000, learning_rate=0.03, max_depth=7, min_child_weight=5,
    colsample_bytree=0.5, subsample=0.8, reg_lambda=1.0,
    tree_method="hist", n_jobs=-1, early_stopping_rounds=200,
)
_CAT_BASE = dict(
    iterations=4000, learning_rate=0.03, depth=8, l2_leaf_reg=3.0,
    loss_function="RMSE", verbose=0, allow_writing_files=False,
)


class LGBMWrap:
    def __init__(self, seed=C.SEED, **kw):
        import lightgbm as lgb
        p = {**_LGB_BASE, "random_state": seed, **kw}
        self.m = lgb.LGBMRegressor(**p)
        self._lgb = lgb

    def fit_fold(self, Xtr, ytr, Xva, yva):
        self.m.fit(Xtr, ytr, eval_set=[(Xva, yva)], eval_metric="rmse",
                   callbacks=[self._lgb.early_stopping(200, verbose=False)])
        return self

    def predict(self, X):
        return self.m.predict(X)


class XGBWrap:
    def __init__(self, seed=C.SEED, **kw):
        import xgboost as xgb
        self.m = xgb.XGBRegressor(**{**_XGB_BASE, "random_state": seed, **kw})

    def fit_fold(self, Xtr, ytr, Xva, yva):
        self.m.fit(Xtr, ytr, eval_set=[(Xva, yva)], verbose=False)
        return self

    def predict(self, X):
        return self.m.predict(X)


class CatWrap:
    def __init__(self, seed=C.SEED, **kw):
        from catboost import CatBoostRegressor
        self.m = CatBoostRegressor(**{**_CAT_BASE, "random_seed": seed, **kw})

    def fit_fold(self, Xtr, ytr, Xva, yva):
        self.m.fit(np.nan_to_num(Xtr), ytr,
                   eval_set=(np.nan_to_num(Xva), yva),
                   early_stopping_rounds=200)
        return self

    def predict(self, X):
        return self.m.predict(np.nan_to_num(X))


class RidgeWrap:
    """Reproduces the organizer baseline's family; adds blend diversity."""

    def __init__(self, seed=C.SEED, alpha=10.0, **kw):
        from sklearn.linear_model import Ridge
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
        from sklearn.impute import SimpleImputer
        self.m = make_pipeline(
            SimpleImputer(strategy="median"),
            StandardScaler(),
            Ridge(alpha=alpha, random_state=seed),
        )

    def fit_fold(self, Xtr, ytr, Xva, yva):
        self.m.fit(Xtr, ytr)
        return self

    def predict(self, X):
        return self.m.predict(X)


REGISTRY = {"lgbm": LGBMWrap, "xgb": XGBWrap, "cat": CatWrap, "ridge": RidgeWrap}


def make(name, seed=C.SEED, **kw):
    if name not in REGISTRY:
        raise KeyError(f"unknown model {name!r}; have {list(REGISTRY)}")
    return lambda: REGISTRY[name](seed=seed, **kw)
