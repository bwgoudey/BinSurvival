from typing import Callable, Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
from SurvivalEVAL import SurvivalEvaluator
import lifelines.datasets


def _load_dataset(name: str) -> pd.DataFrame:
    loader_name = f"load_{name}"
    if not hasattr(lifelines.datasets, loader_name):
        raise ValueError(f"Unknown dataset: {name}")
    loader = getattr(lifelines.datasets, loader_name)
    df = loader()
    # common column names
    for time_col in ["time", "T", "duration"]:
        if time_col in df.columns:
            t_col = time_col
            break
    else:
        raise ValueError("Dataset must contain a time column")
    for event_col in ["event", "E", "status"]:
        if event_col in df.columns:
            e_col = event_col
            break
    else:
        raise ValueError("Dataset must contain an event column")
    df = df.rename(columns={t_col: "time", e_col: "event"})
    return df


def _shift_baseline(df: pd.DataFrame, pct: float) -> pd.DataFrame:
    if pct <= 0:
        return df.copy()
    events = df.loc[df["event"] == 1, "time"].sort_values()
    if len(events) == 0:
        return df.copy()
    idx = max(int(np.ceil(pct * len(events))) - 1, 0)
    shift = events.iloc[idx]
    out = df.copy()
    out["time"] = np.clip(out["time"] - shift, a_min=0, a_max=None)
    return out


def evaluate_models(
    models: Dict[str, object],
    dataset: str,
    resampler: Callable[[pd.DataFrame], Iterable[Tuple[np.ndarray, np.ndarray]]],
    prev_case_pct: float = 0.0,
) -> Tuple[Dict[str, List[Dict[str, float]]], Dict[str, List[np.ndarray]]]:
    """Evaluate multiple survival models on a dataset.

    Parameters
    ----------
    models : mapping of model name to fitted-like object
        Each model must implement ``fit(X, y)`` and ``predict_proba(X, times)``.
    dataset : str
        Name of dataset available through :mod:`lifelines.datasets`.
    resampler : callable
        Function returning iterable of (train_idx, test_idx) arrays.
    prev_case_pct : float, optional
        Proportion of events to treat as prevalent cases by shifting the baseline
        time.

    Returns
    -------
    metrics : dict[str, list[dict[str, float]]]
        Evaluation metrics per fold.
    predictions : dict[str, list[np.ndarray]]
        Predicted survival curves per fold.
    """
    df = _load_dataset(dataset)
    metrics: Dict[str, List[Dict[str, float]]] = {name: [] for name in models}
    preds: Dict[str, List[np.ndarray]] = {name: [] for name in models}

    for train_idx, test_idx in resampler(df):
        train_df = _shift_baseline(df.iloc[train_idx], prev_case_pct)
        test_df = _shift_baseline(df.iloc[test_idx], prev_case_pct)

        X_train = train_df.drop(columns=["time", "event"])
        y_train = train_df[["time", "event"]]
        X_test = test_df.drop(columns=["time", "event"])
        y_test = test_df[["time", "event"]]

        for name, model in models.items():
            model.fit(X_train, y_train)
            times = np.unique(y_test["time"])
            surv = model.predict_proba(X_test, times)
            ev = SurvivalEvaluator(
                surv,
                times,
                y_test["time"],
                y_test["event"],
                y_train["time"],
                y_train["event"],
            )
            res = {
                "concordance": ev.concordance(),
                "ibs": ev.integrated_brier_score(),
            }
            metrics[name].append(res)
            preds[name].append(surv)
    return metrics, preds
