import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from lifelines import CoxPHFitter
from coxbin.TorchCoxMulti import TorchCoxMulti

XCOLS = ["age", "sex"]
TNAME = "time"
DNAME = "status"

def test_logistic_equivalence(lung_df):
    torch.manual_seed(0)
    np.random.seed(0)
    model = TorchCoxMulti(alpha=1, random_state=0)
    model.fit(lung_df, Xnames=XCOLS, tname=TNAME, dname=DNAME, basehaz=False)
    beta = model.beta.detach().numpy()

    y = (lung_df[TNAME] < 1).astype(float).values
    lr = LogisticRegression(fit_intercept=False, solver="lbfgs")
    lr.fit(lung_df[XCOLS].values, y)
    np.testing.assert_allclose(beta, lr.coef_[0], rtol=1e-4, atol=2e-2)

def test_cox_equivalence(lung_df):
    torch.manual_seed(0)
    np.random.seed(0)
    model = TorchCoxMulti(alpha=0, random_state=0)
    model.fit(lung_df, Xnames=XCOLS, tname=TNAME, dname=DNAME, basehaz=False)
    beta = model.beta.detach().numpy()

    df = lung_df[lung_df[TNAME] > 0]
    cph = CoxPHFitter().fit(df[[TNAME, DNAME] + XCOLS], duration_col=TNAME, event_col=DNAME)
    ref = cph.params_.loc[XCOLS].values
    np.testing.assert_allclose(beta, ref, rtol=1e-4, atol=1e-3)

def test_alpha_interpolation(lung_df):
    torch.manual_seed(0)
    np.random.seed(0)
    model = TorchCoxMulti(alpha=0.5, random_state=0)
    model.fit(lung_df, Xnames=XCOLS, tname=TNAME, dname=DNAME, basehaz=False)
    mix_loss = model.compute_loss(lung_df)

    # compute individual losses with current coefficients
    incident_df, logistic_df, logistic_y = model.preprocess_data(lung_df)
    tensor, event_tens, num_tied = model.compute_cox_components(incident_df)
    logistic_X, logistic_y = model.compute_logistic_components(logistic_df, logistic_y)
    cox_loss = model.get_cox_loss(tensor, event_tens, num_tied, model.beta).item()
    log_loss = model.get_logistic_loss(logistic_X, logistic_y, model.beta).item()
    expected = 0.5 * log_loss + 0.5 * cox_loss
    assert np.isclose(mix_loss, expected)
