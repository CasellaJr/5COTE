"""
Federated CAWPE (Cross-validation Accuracy Weighted Probabilistic Ensemble)

Combines the probability outputs of the (up to 4) family models using
weights based on each family's own accuracy, in the spirit of the
original CAWPE (Large, Lines & Bagnall 2019).

Design note (per-client-family assignment protocol)
-----------------------------------------------------
In this codebase each family is trained ONCE, locally, on a single
assigned client's own channel shard (no cross-client feature
concatenation at training time -- see vertical.py for the assignment).
The accuracy used for weighting is NOT computed by this module: it is
supplied by the caller as `weight_accuracies`, one float per family,
computed as the mean of that family's accuracy on (a) its own client's
test set and (b) the SAME already-fitted model evaluated on every OTHER
client's test-set channels (same samples, different columns) -- see
vertical.py's main loop for how these are gathered via each family
module's `extra_test_X` argument. This function is agnostic to how the
accuracy was computed; it only combines probabilities using the given
weights.

Privacy note: this step only needs each family's accuracy numbers (not
raw features) and the (already shared, not channel-specific) labels,
available at the label-holding/orchestrating party without any extra
disclosure of raw features.
"""

import numpy as np


def federated_cawpe(family_names, weight_accuracies, test_probas, alpha=4.0):
    """
    Parameters
    ----------
    family_names : list[str]
        Names of the families that were successfully trained (same order
        as weight_accuracies / test_probas).
    weight_accuracies : list[float]
        Precomputed accuracy used as the CAWPE weighting signal for each
        family (see module docstring for how vertical.py computes this).
    test_probas : list[np.ndarray]
        Each element: (n_test_samples, n_classes) probabilities from one
        family's model, evaluated on its OWN client's test set -- this is
        what actually gets combined into the final ensemble prediction.
    alpha : float, default 4.0
        Exponent applied to accuracy when computing weights, as in the
        original CAWPE paper (higher alpha -> more weight concentrated
        on the best-performing family).

    Returns
    -------
    final_test_pred : np.ndarray, shape (n_test_samples,)
    combined_test_proba : np.ndarray, shape (n_test_samples, n_classes)
    weights : dict[str, float]
        Normalized per-family weights, for reporting/diagnostics.
    """
    if len(test_probas) == 0:
        raise ValueError("No family produced results -- nothing to combine.")
    if len(weight_accuracies) != len(test_probas):
        raise ValueError("weight_accuracies and test_probas must have the same length.")

    accs = np.array(weight_accuracies, dtype=float)

    raw_weights = accs ** alpha
    if raw_weights.sum() == 0:
        # degenerate case: all families at 0 accuracy -> fall back to uniform
        raw_weights = np.ones_like(raw_weights)
    weights = raw_weights / raw_weights.sum()

    combined_test_proba = np.zeros_like(test_probas[0], dtype=float)
    for w, proba in zip(weights, test_probas):
        combined_test_proba += w * proba

    final_test_pred = np.argmax(combined_test_proba, axis=1)
    weight_dict = {name: float(w) for name, w in zip(family_names, weights)}

    return final_test_pred, combined_test_proba, weight_dict