"""
Fed2RC - Vertical variant.

Replaces Arsenal in HIVE-COTE 2.0. Faithful port of the original
(horizontal) Fed2RC's closed-form federated ridge regression, adapted
to the vertical (channel-split) setting -- see federated_ridge.py's
module docstring for the full rationale (what carries over unchanged
from the original vs. what had to be adapted for vertical, and why).

Feature extraction here: each client fits its OWN MiniRocketMultivariate
on its OWN channel shard (see federated_ridge.py's docstring for why the
original's shared-transformer bias-averaging trick doesn't transfer to
vertical -- each client necessarily has a different input width). The
ridge classifier itself is the shared closed_form_federated_ridge
helper, identical across fed2rc/fedst/feddict.

TODO (secure aggregation): the F_train/F_test concatenation below stands
in for a real secure-aggregation protocol (SMPC/HE on the Gram-matrix
cross terms). Fine for prototyping accuracy numbers, not yet
privacy-preserving.
"""

import numpy as np

from utils import make_client_transformer
from sklearn.model_selection import StratifiedKFold, KFold


def closed_form_federated_ridge(F_train, Y_train, F_test, n_classes,
                                 seed, n_cv_folds, lambda_grid=None):
    """
    Parameters
    ----------
    F_train : np.ndarray, shape (n_train_samples, n_features_total)
        Already-aggregated (e.g. vertically concatenated) feature matrix.
    Y_train : array-like, shape (n_train_samples,)
        Integer class labels.
    F_test : np.ndarray, shape (n_test_samples, n_features_total)
    n_classes : int
    seed : int
    n_cv_folds : int
        Number of sample-folds for the leave-one-fold-out lambda search.
    lambda_grid : array-like, optional
        Candidate regularization values. Defaults to
        np.logspace(-3, 0, 100), exactly as in the original Fed2RC code.

    Returns
    -------
    train_proba : np.ndarray, shape (n_train_samples, n_classes)
    test_proba : np.ndarray, shape (n_test_samples, n_classes)
    final_lambda : float
    """
    if lambda_grid is None:
        lambda_grid = np.logspace(-3, 0, 100)

    Y_train = np.asarray(Y_train).astype(int)
    n_features = F_train.shape[1]
    Y_onehot = np.eye(n_classes)[Y_train]

    A_global = F_train.T @ F_train
    b_global = F_train.T @ Y_onehot

    # --- leave-one-sample-fold-out CV for lambda ---
    try:
        kf = StratifiedKFold(n_splits=n_cv_folds, shuffle=True, random_state=seed)
        fold_splits = list(kf.split(F_train, Y_train))
    except ValueError:
        # some class too small to stratify into n_cv_folds -> plain KFold
        kf = KFold(n_splits=n_cv_folds, shuffle=True, random_state=seed)
        fold_splits = list(kf.split(F_train))

    fold_best_lambdas, fold_best_accs = [], []
    identity = np.eye(n_features)
    for _, fold_idx in fold_splits:
        F_fold = F_train[fold_idx]
        Y_fold = Y_train[fold_idx]
        Y_fold_onehot = Y_onehot[fold_idx]

        A_fold = F_fold.T @ F_fold
        b_fold = F_fold.T @ Y_fold_onehot
        A_minus_fold = A_global - A_fold
        b_minus_fold = b_global - b_fold

        best_lambda, best_acc = 0.0, 0.0
        for cand_lambda in lambda_grid:
            A_cand = A_minus_fold + cand_lambda * identity
            try:
                W_cand = np.linalg.solve(A_cand, b_minus_fold)
            except np.linalg.LinAlgError:
                continue
            W_cand = W_cand / (np.linalg.norm(W_cand, axis=0, keepdims=True) + 1e-10)
            fold_pred = np.argmax(F_fold @ W_cand, axis=1)
            fold_acc = (fold_pred == Y_fold).mean()
            if fold_acc > best_acc or (fold_acc == best_acc and cand_lambda > best_lambda):
                best_lambda, best_acc = cand_lambda, fold_acc

        fold_best_lambdas.append(best_lambda * best_acc)
        fold_best_accs.append(best_acc)

    if np.sum(fold_best_accs) > 0:
        final_lambda = float(np.sum(fold_best_lambdas) / np.sum(fold_best_accs))
    else:
        final_lambda = float(np.median(lambda_grid))  # degenerate fallback

    # --- final closed-form solve on the FULL aggregate statistics ---
    A_final = A_global + final_lambda * identity
    W_final = np.linalg.solve(A_final, b_global)
    W_final = W_final / (np.linalg.norm(W_final, axis=0, keepdims=True) + 1e-10)

    def _to_proba(scores):
        exp = np.exp(scores - scores.max(axis=1, keepdims=True))
        return exp / exp.sum(axis=1, keepdims=True)

    train_proba = _to_proba(F_train @ W_final)
    test_proba = _to_proba(F_test @ W_final)

    return train_proba, test_proba, final_lambda


def train_predict(s_X_train, s_Y_train, s_X_test, s_Y_test,
                   n_classes, n_clients, n_rounds, seed,
                   n_kernels=84, n_cv_folds=None, lambda_grid=None, **kwargs):
    """
    n_cv_folds : int, optional
        Number of sample-folds for the lambda-selection CV. Defaults to
        n_clients, matching the original's one-fold-per-client count.
    lambda_grid : array-like, optional
        Candidate regularization values. Defaults to
        np.logspace(-3, 0, 100), exactly as in the original.
    """
    if n_cv_folds is None:
        n_cv_folds = n_clients

    Y_train = s_Y_train[0]  # shared across clients
    Y_test = s_Y_test[0]

    # --- each client fits its OWN transformer on its OWN channel shard ---
    train_blocks, test_blocks = [], []
    for c in range(n_clients):
        transformer = make_client_transformer(n_kernels=n_kernels, seed=seed + c)
        F_train_c = np.asarray(transformer.fit_transform(s_X_train[c]))
        F_test_c = np.asarray(transformer.transform(s_X_test[c]))
        train_blocks.append(F_train_c)
        test_blocks.append(F_test_c)

    # --- SIMULATED AGGREGATION (see module docstring TODO) ---
    F_train = np.concatenate(train_blocks, axis=1)
    F_test = np.concatenate(test_blocks, axis=1)

    train_proba, test_proba, final_lambda = closed_form_federated_ridge(
        F_train, Y_train, F_test, n_classes, seed, n_cv_folds, lambda_grid
    )

    info = {
        "n_kernels_per_client": n_kernels,
        "n_cv_folds": n_cv_folds,
        "final_lambda": final_lambda,
    }
    return train_proba, test_proba, info