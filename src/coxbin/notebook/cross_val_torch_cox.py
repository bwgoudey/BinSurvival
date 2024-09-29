import torch
import numpy as np
import pandas as pd
from lifelines.utils import concordance_index
from lifelines.datasets import load_lung
from torchcox import TorchCoxMulti        
from lifelines.datasets import load_lung
from sklearn.model_selection import RepeatedKFold

def evaluate_model_on_fold(model, train_idx, test_idx, df, Xnames, tname, dname, cutoff_time):
    """Fit model on training data and evaluate on test data."""
    train_df = df.iloc[train_idx]
    test_df = df.iloc[test_idx]

    # Fit the model on training data
    model.fit(train_df, Xnames=Xnames, tname=tname, dname=dname)

    # Predict survival probabilities on the test data
    survival_prob = model.predict_proba(test_df, Xnames=Xnames, tname=tname)
    
    # Evaluate using Harrell's C-index
    test_times = test_df[tname].values
    test_events = test_df[dname].values
    c_index = concordance_index(test_times, -survival_prob, test_events)
    
    return c_index


def tune_alpha(train_df, Xnames, tname, dname, alphas, n_inner_folds=3):
    """Tune alpha using inner cross-validation and return the best alpha."""
    best_alpha = None
    best_c_index = -np.inf

    rkf_inner = RepeatedKFold(n_splits=n_inner_folds, n_repeats=1, random_state=42)
    
    for alpha in alphas:
        avg_c_index = 0
        
        # Initialize the model with the current alpha
        model = TorchCoxMulti.TorchCoxMulti(alpha=alpha)
        
        for train_idx, test_idx in rkf_inner.split(train_df):
            c_index = evaluate_model_on_fold(model, train_idx, test_idx, train_df, Xnames, tname, dname, cutoff_time=None)
            avg_c_index += c_index
        
        avg_c_index /= n_inner_folds
        
        if avg_c_index > best_c_index:
            best_c_index = avg_c_index
            best_alpha = alpha
    
    return best_alpha


def evaluate_fold(df, train_idx, test_idx, Xnames, tname, dname, alphas, cutoff_time, fold):
    """Evaluate models on a single fold."""
    all_c_indices = []
    
    # Split the data for this fold
    train_df = df.iloc[train_idx]
    test_df = df.iloc[test_idx]
    
    # Tune alpha on the training data using inner CV
    best_alpha = tune_alpha(train_df, Xnames, tname, dname, alphas)
    
    # Evaluate the models with different alpha values
    for alpha in alphas:
        model = TorchCoxMulti(alpha=alpha)
        c_index = evaluate_model_on_fold(
            model, train_idx, test_idx, df, Xnames, tname, dname, cutoff_time
        )
        print(f"Fold {fold + 1}, alpha={alpha}, C-index={c_index:.4f}")
        all_c_indices.append((fold + 1, alpha, c_index))
    
    return all_c_indices