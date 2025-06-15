import numpy as np
import pandas as pd
import pytest

@pytest.fixture(scope="session")
def lung_df():
    rng = np.random.default_rng(0)
    n = 200
    beta = np.array([0.5, -0.3])
    X = rng.normal(size=(n, 2))
    linpred = X @ beta
    prev = rng.random(n) < 0.2
    times = rng.exponential(scale=np.exp(-linpred))
    times[prev] = 0.0
    status = np.ones(n, dtype=int)
    df = pd.DataFrame({"time": times, "status": status, "age": X[:, 0], "sex": X[:, 1]})
    return df
