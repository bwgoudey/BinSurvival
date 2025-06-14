import numpy as np
import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from lifelines.utils import concordance_index

from coxbin.TorchCox import TorchCox


def _cindex_score(estimator, X, y):
    times = y[:, 0]
    events = y[:, 1]
    risk = estimator.predict(X)
    return concordance_index(times, -risk, events)


def test_torchcox_fit_predict():
    X = pd.DataFrame({'smoke': [1, 0, 0, 1]})
    y = pd.DataFrame({'time': [1, 3, 6, 10], 'event': [1, 1, 0, 1]})
    model = TorchCox(lr=1.0, Xnames=['smoke'])
    model.fit(X, y)
    assert model.beta is not None
    np.testing.assert_allclose(
        model.beta.detach().numpy()[0], np.log(2) / 2, rtol=1e-4, atol=1e-4
    )
    prob = model.predict_proba(X, y['time'])
    assert prob.shape[0] == len(X)
    risk = model.predict(X)
    assert risk.shape[0] == len(X)


def test_torchcox_pipeline_integration():
    X = pd.DataFrame({'smoke': [1, 0, 0, 1, 1, 0, 0, 1]})
    y = pd.DataFrame({'time': [1, 3, 6, 10, 4, 2, 8, 5], 'event': [1, 1, 0, 1, 0, 1, 1, 0]})
    pipe = make_pipeline(StandardScaler(), TorchCox(lr=1.0, Xnames=['smoke']))
    scores = cross_val_score(
        pipe, X, y[['time', 'event']].values, cv=2, scoring=_cindex_score
    )
    assert scores.shape[0] == 2
    assert not np.isnan(scores).any()
