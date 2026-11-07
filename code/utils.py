import numpy as np
import pandas as pd
import random
import os
from sklearn.preprocessing import LabelEncoder

from sktime.transformations.panel.rocket import MiniRocketMultivariate


def preprocess_data(X_train, Y_train, X_test, Y_test):
    """
    Normalizes the shape of X_train / X_test to always be 3D:
        (n_samples, n_dims, n_timestamps)

    NOTE: this function no longer builds a MiniRocket transformer.
    In the vertical FL setting each client must build and fit its OWN
    transformer on its OWN channel shard (after split_data_vertical),
    so a single "global" transformer fit before splitting doesn't make
    sense. Transformer creation now lives inside each per-family module
    (fed2rc_vertical.py, feddict_vertical.py, ...).
    """
    if X_train.ndim == 2:
        # (n_samples, n_timestamps) -> (n_samples, 1, n_timestamps)
        X_train = X_train.reshape(-1, 1, X_train.shape[1])
        X_test = X_test.reshape(-1, 1, X_test.shape[1])
    elif X_train.ndim != 3:
        raise ValueError(f"Unexpected X_train shape: {X_train.shape}")

    print(f"Data shape after preprocessing: {X_train.shape} "
          f"(n_samples, n_dims, n_timestamps)")

    return X_train, Y_train, X_test, Y_test


def make_client_transformer(n_kernels, seed):
    """
    Every client (in every family that needs a ROCKET-style feature
    extractor) should call this to get its OWN transformer, fit only
    on its OWN channel shard. MiniRocketMultivariate is used
    unconditionally because it handles the n_dims == 1 case natively,
    unlike the old if/else on MiniRocket vs MiniRocketMultivariate.
    """
    return MiniRocketMultivariate(num_kernels=n_kernels, random_state=int(seed))


def safe_convert_labels(y_train, y_test):
    """
    Encodes labels consistently across train/test.
    Fixed: LabelEncoder instance was referenced (`LE`) but never created.
    """
    le = LabelEncoder()
    try:
        # try numeric-looking labels first (handles '1.0', '2', ecc.)
        y_train_safe = y_train.astype(float).astype(int).astype(str)
        y_test_safe = y_test.astype(float).astype(int).astype(str)
        y_train_enc = le.fit_transform(y_train_safe)
        y_test_enc = le.transform(y_test_safe)
        return y_train_enc, y_test_enc
    except (ValueError, TypeError):
        # non-numeric labels -> encode as strings directly
        y_train_safe = y_train.astype(str)
        y_test_safe = y_test.astype(str)
        y_train_enc = le.fit_transform(y_train_safe)
        y_test_enc = le.transform(y_test_safe)
        return y_train_enc, y_test_enc


def split_data_vertical(X_train, Y_train, X_test, Y_test, n_clients, rng=None, shuffle_dims=False):
    """
    Splits a multivariate time series dataset into n_clients vertical
    shards: every client keeps ALL samples but only a subset of the
    channels/dimensions. Labels are shared by all clients (only the
    active party actually needs them for training/eval, but we keep
    them available on every client for simplicity/simulation purposes).

    Parameters
    ----------
    X_train, X_test : np.ndarray, shape (n_samples, n_dims, n_timestamps)
    n_clients : int
    rng : np.random.RandomState, optional
        Used only if shuffle_dims=True, to decide the (fixed, reused for
        train/test) permutation of channels assigned to each client.
    shuffle_dims : bool, default False
        If True, channels are randomly permuted before being split into
        contiguous blocks (useful later to test sensitivity to channel
        grouping / correlated-channel-per-client scenarios).
    """
    n_samples, n_dims, n_timestamps = X_train.shape
    if n_clients <= 0 or n_clients > n_dims:
        raise ValueError("Number of clients must be greater than 0 and less than or equal to the number of dimensions.")

    dim_order = np.arange(n_dims)
    if shuffle_dims:
        if rng is None:
            raise ValueError("shuffle_dims=True requires an rng (np.random.RandomState).")
        dim_order = rng.permutation(n_dims)

    dims_per_client = n_dims // n_clients
    remainder = n_dims % n_clients

    s_X_train, s_X_test = [], []
    s_Y_train = [Y_train] * n_clients  # all clients share the same labels
    s_Y_test = [Y_test] * n_clients

    client_dim_indices = []  # keep track of which original channels went to which client
    start = 0
    for i in range(n_clients):
        end = start + dims_per_client + (1 if i < remainder else 0)
        idx = dim_order[start:end]
        client_dim_indices.append(idx)
        s_X_train.append(X_train[:, idx, :])
        s_X_test.append(X_test[:, idx, :])
        start = end

    return s_X_train, s_Y_train, s_X_test, s_Y_test, client_dim_indices


# ----------------------------------------------------------------------
# Helper: convert 2D numpy array to nested pandas DataFrame
# ----------------------------------------------------------------------
def to_nested_dataframe(X):
    X_series = [pd.Series(x) for x in X]
    return pd.DataFrame({"X": X_series})