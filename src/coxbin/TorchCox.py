import pandas as pd
from sklearn.base import BaseEstimator
import torch
from torch import nn
from torch import optim
import numpy as np

torch.autograd.set_detect_anomaly(True)


class TorchCox(BaseEstimator):
    """Fit a Cox proportional hazards model."""

    def __init__(
        self, lr=1.0, random_state=None, *, Xnames=None, tname="time", dname="event"
    ):
        self.random_state = random_state
        self.lr = lr
        self.Xnames = Xnames
        self.tname = tname
        self.dname = dname

    def _padToMatch2d(self, inputtens, targetshape):
        target = torch.full(targetshape, fill_value=-1e3)  # torch.zeros(*targetshape)
        target[: inputtens.shape[0], : inputtens.shape[1]] = inputtens
        return target

    def get_loss(self, tensor, event_tens, num_tied, beta):
        loss_event = torch.einsum("ik,k->i", event_tens, beta)

        XB = torch.einsum("ijk,k->ij", tensor, beta)
        loss_atrisk = -num_tied * torch.logsumexp(XB, dim=1)

        loss = torch.sum(loss_event + loss_atrisk)

        return -loss

    def fit(self, X, y, basehaz=True):
        """Fit the model.

        Parameters
        ----------
        X : array-like or DataFrame of shape (n_samples, n_features)
            The covariate matrix.
        y : array-like or DataFrame of shape (n_samples, 2)
            Structured array containing the survival time and event indicator.
        basehaz : bool, default=True
            Whether to compute and store the baseline cumulative hazard.
        """

        if isinstance(X, pd.DataFrame):
            if self.Xnames is None:
                self.Xnames = list(X.columns)
            X_df = X.reset_index(drop=True)
        else:
            if self.Xnames is None:
                raise ValueError("Xnames must be provided when X is not a DataFrame")
            X_df = pd.DataFrame(X, columns=self.Xnames)

        if isinstance(y, pd.DataFrame):
            t = y[self.tname].reset_index(drop=True)
            d = y[self.dname].reset_index(drop=True)
        else:
            t = pd.Series(y[:, 0])
            d = pd.Series(y[:, 1])

        df = pd.concat(
            [t.rename(self.tname), d.rename(self.dname), X_df.reset_index(drop=True)],
            axis=1,
        )

        # self.random_state_ = check_random_state(self.random_state)
        beta = nn.Parameter(torch.zeros(len(self.Xnames))).float()

        optimizer = optim.LBFGS([beta], lr=self.lr)

        inputdf = df[[self.tname, self.dname, *self.Xnames]].sort_values(
            [self.dname, self.tname], ascending=[False, True]
        )

        tiecountdf = (
            inputdf.loc[inputdf[self.dname] == 1, :]
            .groupby([self.tname])
            .size()
            .reset_index(name="tiecount")
        )
        num_tied = torch.from_numpy(tiecountdf.tiecount.values).int()

        tensin = torch.from_numpy(
            inputdf[[self.tname, self.dname, *self.Xnames]].values
        )

        # Get unique event times
        tensin_events = torch.unique(tensin[tensin[:, 1] == 1, 0])

        # For each unique event stack another matrix with event at the top, and all at risk entries below
        tensor = torch.stack(
            [
                self._padToMatch2d(tensin[tensin[:, 0] >= eventtime, :], tensin.shape)
                for eventtime in tensin_events
            ]
        )

        assert all(tensor[:, 0, 1] == 1)

        # One actually has to sum over the covariates which have a tied event time in the Breslow correction method!
        # See page 33 here: https://www.math.ucsd.edu/~rxu/math284/slect5.pdf
        event_tens = torch.stack(
            [tensor[i, : num_tied[i], 2:].sum(dim=0) for i in range(tensor.shape[0])]
        )

        # Drop time and status columns as no longer required
        tensor = tensor[:, :, 2:]

        def closure():
            optimizer.zero_grad()
            loss = self.get_loss(tensor, event_tens, num_tied, beta)
            # print(loss)
            loss.backward()
            return loss

        optimizer.step(closure)

        self.beta = beta
        print(self.beta.detach().numpy())

        # Compute baseline hazard during fit() to avoid having to save dataset to memory in TorchCox() objects, so it
        #  can then later be calculated if basehaz() is called.
        if basehaz:
            t, _ = torch.sort(torch.from_numpy(inputdf[self.tname].values))
            t_uniq = torch.unique(t)

            h0 = []
            for time in t_uniq:
                value = 1 / torch.sum(
                    torch.exp(
                        torch.einsum(
                            "ij,j->i",
                            torch.from_numpy(
                                inputdf.loc[
                                    inputdf[self.tname] >= time.numpy(), self.Xnames
                                ].values
                            ).float(),
                            self.beta,
                        )
                    )
                )
                h0.append({"time": time.numpy(), "h0": value.detach().numpy()})

            h0df = pd.DataFrame(h0)
            h0df["H0"] = h0df.h0.cumsum()

            self.basehaz = h0df

    def predict_proba(self, X, times):
        """Predict survival probabilities at the given times."""

        if isinstance(X, pd.DataFrame):
            X_mat = X[self.Xnames].values
        else:
            X_mat = X

        betas = self.beta.detach().numpy()
        H0 = np.asarray(
            [self.basehaz.loc[self.basehaz.time <= t, "H0"].iloc[-1] for t in times]
        )

        S = np.exp(-np.exp(np.dot(X_mat, betas)) * H0)
        S = np.clip(S, 0, 1)
        return S

    def predict(self, X):
        """Predict the relative risk (partial hazard)."""
        if isinstance(X, pd.DataFrame):
            X_mat = X[self.Xnames].values
        else:
            X_mat = X
        beta = self.beta.detach().numpy()
        return np.exp(np.dot(X_mat, beta))
