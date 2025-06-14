import numpy as np
import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from lifelines.utils import concordance_index

from coxbin.TorchCoxMulti import TorchCoxMulti


def _cindex_score(estimator, X, y):
    times = y[:, 0]
    events = y[:, 1]
    risk = estimator.predict(X)
    return concordance_index(times, -risk, events)


def test_torchcoxmulti_fit_predict():
    X = pd.DataFrame({'smoke': [1, 0, 0, 1, 1, 0, 0, 0]})
    y = pd.DataFrame({'time': [1, 3, 6, 10, -1, 0, -0.5, 1], 'event': [1, 1, 0, 1, 1, 0, 1, 0]})
    model = TorchCoxMulti(lr=1.0, Xnames=['smoke'])
    model.fit(X, y)
    assert model.beta is not None
    assert not model.basehaz.empty
    X_pos = X[y['time'] > 0]
    times = y.loc[y['time'] > 0, 'time']
    prob = model.predict_proba(X_pos, times)
    assert prob.shape[0] == len(times)


def test_torchcoxmulti_pipeline_integration():
    X = pd.DataFrame({'smoke': [1, 0, 1, 0, 1, 0, 1, 0, 1, 0]})
    y = pd.DataFrame({'time': [1, 3, 2, 4, 5, 6, 1.5, 2.5, -1, 0],
                      'event': [1, 1, 1, 0, 1, 0, 1, 0, 1, 0]})
    pipe = make_pipeline(StandardScaler(), TorchCoxMulti(lr=1.0, Xnames=['smoke']))
    scores = cross_val_score(
        pipe, X, y[['time', 'event']].values, cv=2, scoring=_cindex_score
    )
    assert scores.shape[0] == 2
    assert not np.isnan(scores).any()
