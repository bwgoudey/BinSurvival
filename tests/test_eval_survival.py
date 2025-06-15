import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

from coxbin.eval_survival import _load_dataset, _shift_baseline, evaluate_models

class DummyModel:
    def fit(self, X, y):
        # store mean time to make a simple exponential model
        self.mean_time_ = y['time'].mean()
    def predict_proba(self, X, times):
        t = np.asarray(times)
        surv = np.exp(-t / self.mean_time_)
        return np.tile(surv, (len(X), 1))


def resampler(df):
    kf = KFold(n_splits=2, shuffle=False)
    for train_idx, test_idx in kf.split(df):
        yield train_idx, test_idx


def test_load_dataset():
    df = _load_dataset('lung')
    assert {'time', 'event'}.issubset(df.columns)
    assert len(df) > 0


def test_shift_baseline():
    df = pd.DataFrame({'time': [1, 2, 3, 4], 'event': [1, 0, 1, 0]})
    shifted = _shift_baseline(df, 0.5)
    assert shifted['time'].iloc[0] == 0
    assert shifted['time'].iloc[2] == 2


def test_evaluate_models():
    metrics, preds = evaluate_models({'dummy': DummyModel()}, 'lung', resampler, prev_case_pct=0.1)
    assert 'dummy' in metrics
    assert len(metrics['dummy']) == 2
    assert len(preds['dummy']) == 2
    assert preds['dummy'][0].shape[0] > 0
