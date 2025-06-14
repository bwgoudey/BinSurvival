import pandas as pd
from sklearn.base import BaseEstimator
import torch
from torch import nn
from torch import optim
import numpy as np

torch.autograd.set_detect_anomaly(True)


class TorchCoxMulti(BaseEstimator):
    """Fit a Cox model with an additional logistic regression loss for left-censored observations."""

    def __init__(
        self,
        lr=1.0,
        alpha=0.5,
        random_state=None,
        *,
        Xnames=None,
        tname="time",
        dname="event",
    ):
        self.random_state = random_state
        self.lr = lr
        self.alpha = alpha  # Balance between Cox loss and logistic loss
        self.Xnames = Xnames
        self.tname = tname
        self.dname = dname

    def _padToMatch2d(self, inputtens, targetshape):
        target = torch.full(targetshape, fill_value=-1e3)
        target[: inputtens.shape[0], : inputtens.shape[1]] = inputtens
        return target

    def get_cox_loss(self, tensor, event_tens, num_tied, beta):
        """Compute the Cox regression loss."""
        loss_event = torch.einsum("ik,k->i", event_tens, beta)
        XB = torch.einsum("ijk,k->ij", tensor, beta)
        loss_atrisk = -num_tied * torch.logsumexp(XB, dim=1)
        loss = torch.sum(loss_event + loss_atrisk)
        return -loss

    def get_logistic_loss(self, X, y, beta):
        """Compute the logistic regression loss for prevalent cases."""
        logits = X @ beta
        loss_fn = nn.BCEWithLogitsLoss()
        loss = loss_fn(logits, y)
        return loss

    def preprocess_data(self, df):
        """Preprocess the data and separate it for Cox and logistic regression."""
        # Data for Cox regression (incident cases)
        incident_df = df[df[self.tname] > 0]

        # Data for logistic regression (prevalent cases and controls)
        # Cases: time < 1
        # Controls: time >= 1
        cases_df = df[df[self.tname] < 1]
        controls_df = df[df[self.tname] >= 1]

        # Combine cases and controls for logistic regression
        logistic_df = pd.concat([cases_df, controls_df], ignore_index=True)
        logistic_y = np.where(
            logistic_df[self.tname] < 1, 1.0, 0.0
        )  # 1 for cases, 0 for controls

        return incident_df, logistic_df, logistic_y

    def compute_cox_components(self, incident_df):
        """Compute components required for the Cox regression loss."""
        inputdf = incident_df[[self.tname, self.dname, *self.Xnames]].sort_values(
            [self.dname, self.tname], ascending=[False, True]
        )

        tiecountdf = (
            inputdf.loc[inputdf[self.dname] == 1]
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

        # For each unique event, stack a matrix with event at the top and all at-risk entries below
        tensor_list = []
        for eventtime in tensin_events:
            at_risk = tensin[tensin[:, 0] >= eventtime]
            padded = self._padToMatch2d(at_risk, tensin.shape)
            tensor_list.append(padded)
        tensor = torch.stack(tensor_list)

        # Ensure the first entry is an event
        assert all(tensor[:, 0, 1] == 1)

        # Sum over the covariates for tied event times
        event_tens = torch.stack(
            [tensor[i, : num_tied[i], 2:].sum(dim=0) for i in range(tensor.shape[0])]
        )

        # Drop time and status columns
        tensor = tensor[:, :, 2:]

        return tensor, event_tens, num_tied

    def compute_logistic_components(self, logistic_df, logistic_y):
        """Compute components required for the logistic regression loss."""
        logistic_X = torch.from_numpy(logistic_df[self.Xnames].values).float()
        logistic_y = torch.from_numpy(logistic_y).float()
        return logistic_X, logistic_y

    def get_loss(self, tensor, event_tens, num_tied, logistic_X, logistic_y, beta):
        """Compute the combined loss function."""
        cox_loss = self.get_cox_loss(tensor, event_tens, num_tied, beta)
        logistic_loss = self.get_logistic_loss(logistic_X, logistic_y, beta)
        loss = (1 - self.alpha) * cox_loss + self.alpha * logistic_loss
        return loss

    def fit(self, X, y, basehaz=True):
        """Fit the model using (X, y) style arguments."""

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

        beta = nn.Parameter(torch.zeros(len(self.Xnames))).float()
        optimizer = optim.LBFGS([beta], lr=self.lr)

        # Preprocess data
        incident_df, logistic_df, logistic_y = self.preprocess_data(df)

        # Compute Cox components
        tensor, event_tens, num_tied = self.compute_cox_components(incident_df)

        # Compute Logistic components
        logistic_X, logistic_y = self.compute_logistic_components(
            logistic_df, logistic_y
        )

        def closure():
            optimizer.zero_grad()
            loss = self.get_loss(
                tensor, event_tens, num_tied, logistic_X, logistic_y, beta
            )
            loss.backward()
            return loss

        optimizer.step(closure)

        self.beta = beta
        # print("Estimated coefficients:", self.beta.detach().numpy())

        # Compute baseline hazard if required
        if basehaz:
            t, _ = torch.sort(torch.from_numpy(incident_df[self.tname].values))
            t_uniq = torch.unique(t)

            h0 = []
            for time in t_uniq:
                risk_set = incident_df[incident_df[self.tname] >= time.item()]
                X_risk = torch.from_numpy(risk_set[self.Xnames].values).float()
                exp_X_beta = torch.exp(X_risk @ self.beta)
                value = 1 / torch.sum(exp_X_beta)
                h0.append({"time": time.item(), "h0": value.item()})

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

    def compute_loss(self, df):
        """Compute the combined loss on new data."""
        # Preprocess the data
        incident_df, logistic_df, logistic_y = self.preprocess_data(df)

        # Compute Cox components
        tensor, event_tens, num_tied = self.compute_cox_components(incident_df)

        # Compute Logistic components
        logistic_X, logistic_y = self.compute_logistic_components(
            logistic_df, logistic_y
        )

        # Compute the combined loss
        loss = self.get_loss(
            tensor, event_tens, num_tied, logistic_X, logistic_y, self.beta
        )
        return loss.item()
