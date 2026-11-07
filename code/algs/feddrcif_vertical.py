"""
Fed-DrCIF - Vertical variant.

Replaces DrCIF (interval-based) in HIVE-COTE 2.0 -- the last of the four
HC2 components without a federated counterpart anywhere in the
literature (horizontal or vertical), closing the loop on this
framework's four families.

Design: TREE POOLING (default aggregation="tree_pooling")
-------------------------------------------------------------
Unlike fed2rc/fedst/feddict (which concatenate per-sample DERIVED
FEATURES from every client into one shared ridge classifier -- the
"SIMULATED AGGREGATION" caveat repeated across this codebase), DrCIF is
a tree ENSEMBLE (Time Series Forest / CIF style), and trees are
naturally poolable objects: once fitted, a tree is a compact, already-
aggregated artifact (split thresholds, leaf class distributions) that
reveals nothing about individual training samples the way a raw
per-sample feature vector does. This lets Fed-DrCIF use a structurally
different (and more private) aggregation paradigm than the other three
families in this codebase:

  1. LOCAL tree growing (no communication): each client c, using ONLY
     its own channels, grows T/n_clients trees. For each tree: sample
     random (channel, representation, start, length) intervals across
     THREE representations (raw, first difference, periodogram -- the
     "Diverse Representation" DrCIF is named for), extract interval
     features (see the feature-fidelity note below), bootstrap-sample
     rows (bagging), fit a DecisionTreeClassifier.
  2. Tree pooling (the ONLY exchange): each client's fitted trees (plus
     the interval definitions needed to re-extract features at
     inference time) are pooled into one global forest. No raw data, no
     per-sample feature vectors, ever cross a client boundary.
  3. Inference: each tree is evaluated using only its owning client's
     channels; all trees' probabilities are averaged (standard bagging).

This is the direct interval-forest analogue of FedAvg-ZeroPad sharing
weights instead of data -- a fourth distinct aggregation paradigm
alongside feature-concatenation (fed2rc/fedst/feddict), split-learning
embeddings (fedinception), and weight-averaging (fedavg_zeropad).

Second mode for comparison: aggregation="concat"
------------------------------------------------------
Same interval-feature extraction, but following the SAME
concatenate-then-shared-classifier pattern as fed2rc/fedst/feddict: each
client extracts one block of interval features, blocks are concatenated,
ONE shared ExtraTreesClassifier is fit on top. This CAN exploit splits
that combine features from different clients' channels (impossible for
tree_pooling, where a tree only ever sees its own client's channels),
at the cost of requiring per-sample feature vectors to leave each
client -- a clean two-way ablation: how much does forbidding
cross-client feature interactions in the trees actually cost, in
exchange for never sharing per-sample features at all?

Feature-fidelity note (IMPORTANT)
-------------------------------------
Real DrCIF uses the catch22 feature set. This environment has no
network access to verify pycatch22's exact API, so the DEFAULT feature
extractor here is a small set of fully-vectorized, hand-verified
interval statistics (mean, std, slope, skewness, kurtosis, lag-1
autocorrelation, min, max, IQR) -- NOT catch22, labeled as such
throughout. If pycatch22 is importable, set use_pycatch22=True to try
it (best-effort; falls back to the verified stats with a warning if the
import or call fails) -- confirm pycatch22's exact function signature
before trusting that path for a paper's real experiments.

Performance note: default n_trees_total (100) and interval counts are
set lighter than DrCIF's usual defaults (500 trees) given how much
runtime has mattered throughout this codebase -- raise them once you've
confirmed the accuracy/speed trade-off you want.
"""

import warnings

import numpy as np
from scipy import signal, stats
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.tree import DecisionTreeClassifier

try:
    import pycatch22
    _HAS_PYCATCH22 = True
except ImportError:
    _HAS_PYCATCH22 = False


# ----------------------------------------------------------------------
# Representations
# ----------------------------------------------------------------------

def _compute_representations(X_client):
    """
    X_client: (n_samples, n_channels, ts_length).
    Returns {"raw": ..., "diff": ..., "periodogram": ...}, each
    (n_samples, n_channels, repr_length) -- the "Diverse Representation"
    DrCIF is named for.
    """
    raw = X_client
    diff = np.diff(X_client, axis=2)
    _, pxx = signal.periodogram(X_client, axis=2)
    return {"raw": raw, "diff": diff, "periodogram": pxx}


# ----------------------------------------------------------------------
# Interval feature extraction
# ----------------------------------------------------------------------

def _fallback_interval_features_batch(data):
    """
    data: (n_samples, length). Fully vectorized across samples.
    NOT catch22 -- see module docstring's feature-fidelity note.
    Returns (n_samples, 9).
    """
    n_samples, length = data.shape
    mean = data.mean(axis=1)
    std = data.std(axis=1)

    if length >= 2:
        t = np.arange(length)
        t_mean = t.mean()
        denom = np.sum((t - t_mean) ** 2)
        slope = ((data - mean[:, None]) @ (t - t_mean)) / denom if denom > 1e-12 else np.zeros(n_samples)
        centered = data - mean[:, None]
        num = np.sum(centered[:, :-1] * centered[:, 1:], axis=1)
        den = np.sum(centered ** 2, axis=1)
        autocorr = np.where(den > 1e-12, num / den, 0.0)
    else:
        slope = np.zeros(n_samples)
        autocorr = np.zeros(n_samples)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # scipy warns on near-constant intervals; handled by nan_to_num below
        skew = stats.skew(data, axis=1) if length >= 3 else np.zeros(n_samples)
        kurt = stats.kurtosis(data, axis=1) if length >= 4 else np.zeros(n_samples)

    mn = data.min(axis=1)
    mx = data.max(axis=1)
    q75, q25 = np.percentile(data, [75, 25], axis=1)
    iqr = q75 - q25

    feats = np.stack([mean, std, slope, skew, kurt, autocorr, mn, mx, iqr], axis=1)
    return np.nan_to_num(feats, nan=0.0, posinf=0.0, neginf=0.0)


def _pycatch22_features_batch(data):
    """
    Best-effort pycatch22 path -- NOT verified against the real API in
    this environment (no network access). Falls back to
    _fallback_interval_features_batch with a one-time warning if the
    call fails; confirm the exact pycatch22 signature before trusting
    this for real experiments.
    """
    try:
        feats = np.array([pycatch22.catch22_all(row.tolist())["values"] for row in data])
        return feats
    except Exception as e:  # noqa: BLE001 -- best-effort fallback, see docstring
        warnings.warn(f"pycatch22 call failed ({e!r}); falling back to the verified non-catch22 "
                       f"interval statistics for this batch. See module docstring.")
        return _fallback_interval_features_batch(data)


def _interval_features_batch(data, use_pycatch22):
    if use_pycatch22 and _HAS_PYCATCH22:
        return _pycatch22_features_batch(data)
    return _fallback_interval_features_batch(data)


# ----------------------------------------------------------------------
# Interval sampling
# ----------------------------------------------------------------------

def _sample_intervals(n_channels, repr_lengths, n_intervals_per_repr, min_interval_length, rng):
    """Returns list of (channel, representation, start, length)."""
    intervals = []
    for repr_name, length in repr_lengths.items():
        if length < min_interval_length:
            continue
        for _ in range(n_intervals_per_repr):
            channel = rng.randint(0, n_channels)
            interval_len = rng.randint(min_interval_length, length + 1)
            start = rng.randint(0, length - interval_len + 1)
            intervals.append((channel, repr_name, start, interval_len))
    return intervals


def _extract_features(representations, intervals, use_pycatch22):
    """representations: dict from _compute_representations. Returns
    (n_samples, n_intervals * n_features_per_interval)."""
    blocks = []
    for channel, repr_name, start, length in intervals:
        data = representations[repr_name][:, channel, start:start + length]
        blocks.append(_interval_features_batch(data, use_pycatch22))
    return np.concatenate(blocks, axis=1)


# ----------------------------------------------------------------------
# Tree pooling
# ----------------------------------------------------------------------

def _build_and_score_client_forest(X_train_c, Y_train, X_test_c, n_trees, n_intervals_per_repr,
                                    min_interval_length, max_depth, use_pycatch22, seed, n_classes):
    """
    Grows n_trees decision trees LOCALLY on client c's own channels (no
    cross-client communication) and immediately scores train/test with
    each tree as it's built -- train features are computed ONCE per
    tree (reused for both fitting and the train-probability
    contribution) instead of being recomputed in a separate later pass.
    Returns (train_proba_sum, test_proba_sum, n_trees_built).
    """
    n_samples, n_channels, ts_length = X_train_c.shape
    rng = np.random.RandomState(seed)
    reps_train = _compute_representations(X_train_c)
    reps_test = _compute_representations(X_test_c)
    repr_lengths = {name: arr.shape[2] for name, arr in reps_train.items()}

    train_proba_sum = np.zeros((X_train_c.shape[0], n_classes))
    test_proba_sum = np.zeros((X_test_c.shape[0], n_classes))
    n_built = 0

    for t in range(n_trees):
        intervals = _sample_intervals(n_channels, repr_lengths, n_intervals_per_repr,
                                       min_interval_length, rng)
        if not intervals:
            continue
        F_train = _extract_features(reps_train, intervals, use_pycatch22)  # computed ONCE

        bootstrap_idx = rng.randint(0, n_samples, n_samples)  # bagging, with replacement
        clf = DecisionTreeClassifier(criterion="entropy", max_depth=max_depth, random_state=seed + t)
        clf.fit(F_train[bootstrap_idx], Y_train[bootstrap_idx])

        F_test = _extract_features(reps_test, intervals, use_pycatch22)
        for F, out in ((F_train, train_proba_sum), (F_test, test_proba_sum)):
            proba = clf.predict_proba(F)
            for i, cls in enumerate(clf.classes_):
                out[:, cls] += proba[:, i]
        n_built += 1

    return train_proba_sum, test_proba_sum, n_built


def train_predict(s_X_train, s_Y_train, s_X_test, s_Y_test,
                   n_classes, n_clients, n_rounds, seed,
                   aggregation="tree_pooling",
                   n_trees_total=100, n_intervals_per_repr=None,
                   min_interval_length=3, max_depth=None,
                   use_pycatch22=False, **kwargs):
    """
    aggregation : {"tree_pooling", "concat"}, default "tree_pooling"
        See module docstring for the two designs.
    n_trees_total : int, default 100
        Lighter than DrCIF's usual 500 -- see performance note.
    n_intervals_per_repr : int, optional
        Random intervals sampled PER representation (raw/diff/
        periodogram) PER tree (tree_pooling) or for the whole client
        (concat). Defaults to round(sqrt(ts_length)), the usual
        TSF/CIF convention.
    use_pycatch22 : bool, default False
        See the feature-fidelity note in the module docstring --
        defaults to the verified non-catch22 fallback statistics.
    """
    Y_train = np.asarray(s_Y_train[0]).astype(int)  # shared across clients
    Y_test = np.asarray(s_Y_test[0]).astype(int)

    if aggregation == "tree_pooling":
        trees_per_client = max(1, n_trees_total // n_clients)
        train_proba = np.zeros((len(Y_train), n_classes))
        test_proba = np.zeros((len(Y_test), n_classes))
        total_trees = 0

        for c in range(n_clients):
            ts_length = s_X_train[c].shape[2]
            n_int = n_intervals_per_repr or max(1, round(np.sqrt(ts_length)))
            train_sum_c, test_sum_c, n_built = _build_and_score_client_forest(
                s_X_train[c], Y_train, s_X_test[c], trees_per_client, n_int,
                min_interval_length, max_depth, use_pycatch22, seed + c * 1000, n_classes
            )
            train_proba += train_sum_c
            test_proba += test_sum_c
            total_trees += n_built

        train_proba /= max(total_trees, 1)
        test_proba /= max(total_trees, 1)

        info = {
            "aggregation": "tree_pooling",
            "n_trees_total": total_trees,
            "trees_per_client": trees_per_client,
            "n_intervals_per_repr": n_intervals_per_repr,
            "min_interval_length": min_interval_length,
            "max_depth": max_depth,
            "use_pycatch22": use_pycatch22 and _HAS_PYCATCH22,
        }

    elif aggregation == "concat":
        train_blocks, test_blocks = [], []
        for c in range(n_clients):
            ts_length = s_X_train[c].shape[2]
            n_int = n_intervals_per_repr or max(1, round(np.sqrt(ts_length)))
            rng_c = np.random.RandomState(seed + c)
            reps_train = _compute_representations(s_X_train[c])
            reps_test = _compute_representations(s_X_test[c])
            repr_lengths = {name: arr.shape[2] for name, arr in reps_train.items()}
            intervals = _sample_intervals(s_X_train[c].shape[1], repr_lengths, n_int,
                                           min_interval_length, rng_c)
            train_blocks.append(_extract_features(reps_train, intervals, use_pycatch22))
            test_blocks.append(_extract_features(reps_test, intervals, use_pycatch22))

        # --- SIMULATED AGGREGATION --- (same caveat as fed2rc/fedst/feddict)
        F_train = np.concatenate(train_blocks, axis=1)
        F_test = np.concatenate(test_blocks, axis=1)

        clf = ExtraTreesClassifier(n_estimators=n_trees_total, criterion="entropy", random_state=seed)
        clf.fit(F_train, Y_train)

        train_proba = np.zeros((len(Y_train), n_classes))
        test_proba = np.zeros((len(Y_test), n_classes))
        for F, out in ((F_train, train_proba), (F_test, test_proba)):
            proba = clf.predict_proba(F)
            for i, cls in enumerate(clf.classes_):
                out[:, cls] = proba[:, i]

        info = {
            "aggregation": "concat",
            "n_trees_total": n_trees_total,
            "n_intervals_per_repr": n_intervals_per_repr,
            "min_interval_length": min_interval_length,
            "use_pycatch22": use_pycatch22 and _HAS_PYCATCH22,
        }
    else:
        raise ValueError(f"aggregation must be 'tree_pooling' or 'concat', got {aggregation!r}")

    return train_proba, test_proba, info