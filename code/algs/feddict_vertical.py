"""
FedDict - Vertical variant (v2).

Replaces TDE (Temporal Dictionary Ensemble) in HIVE-COTE 2.0. Rewritten
to be faithful to the original (horizontal) FedDict algorithm after a
detailed correction -- supersedes an earlier version that used SAX
(time-domain PAA symbolization) instead of SFA (frequency-domain DFT
symbolization), and was missing coefficient selection, equi-width
binning, multi-resolution windows, and bigrams entirely.

Original algorithm (per the paper, as described to me)
---------------------------------------------------------
For a series T_j, the final feature vector concatenates D per-window-
length representations:
    X_j = (+)_{d=1}^{D} Dict(SFA(T_j, w_d, l_d, 4; theta_d))
For each window length w_d (l_d = word length, chosen uniformly at
random from {4, 5, 6} using w_d itself as the random seed; alphabet
size fixed at 4):
  1. Slide a window of length w_d over the (z-normalized) series,
     compute the DFT of each window -> w_d raw Fourier coefficients.
  2. Compute, per coefficient, the ANOVA F-value against the class
     labels, USING ALL CLIENTS (in the original horizontal setting this
     means aggregating per-class sum / sum-of-squares / count across
     clients, since different clients hold different SAMPLES -- the
     F-value is exactly decomposable into these sums, so it can be
     computed federatedly without ever sharing raw per-sample
     coefficient values).
  3. Keep the top l_d coefficients by F-value.
  4. Discretize each of those l_d coefficients into `alphabet_size`
     equal-WIDTH bins, with bin edges taken from the min/max of the
     SELECTED coefficient's own observed values (not a theoretical
     distribution) -- sidesteps estimating the global coefficient
     distribution.
  5. Each window becomes an l_d-symbol word; build a dictionary of
     UNIGRAM (word) and BIGRAM (consecutive word pair) counts.

Why the federated ANOVA-selection step becomes a LOCAL computation here
--------------------------------------------------------------------------
In vertical, clients hold the SAME samples and DIFFERENT CHANNELS -- the
opposite partitioning from horizontal. Step 2 needs "all clients" in the
original specifically because no single horizontal client's local
sample subset gives the TRUE (whole-dataset) per-class sum / sum-of-
squares / count for a coefficient; pooling across clients is what makes
the F-value match centralized. In vertical, each client already has ALL
the samples for its own channel, so its local ANOVA F-value computation
already IS the true one for that channel's coefficients -- aggregating
across clients would add nothing. Same story for step 4's min/max: a
client's own local min/max over its own samples already IS the global
min/max for that channel, since no other client holds any of those same
samples' values for that channel. This is the same pattern already
noted for FedST's shapelet search (fedst_vertical.py) and Fed2RC's
original horizontal bias-averaging trick: horizontal-specific
coordination mechanisms that exist to pool information across disjoint
SAMPLE partitions collapse to a purely local step once every client
already holds every sample. The only place aggregation genuinely still
happens here is the same "SIMULATED AGGREGATION" concatenation shared by
every family in this codebase (see fed2rc_vertical.py) -- turning each
client's already-local dictionary features into one joint feature
matrix for the ridge classifier.

Window-length grid
--------------------
The exact grid of w_d values used in the original paper wasn't
specified to me -- `window_lengths` defaults to a small spread across
the series length (see _default_window_lengths). Override it explicitly
once the paper's actual grid is known.

Classifier
-----------
Plain sklearn RidgeClassifierCV on top (per the decision to keep the
closed-form federated ridge in federated_ridge.py scoped to fed2rc
only) -- this file is about the FEATURE EXTRACTION being faithful to
FedDict, not about the classifier.

Performance / memory note
---------------------------
D window lengths x every channel x (DFT + ANOVA F-test over
n_samples*n_windows rows), unigrams+bigrams. Feature extraction is now
built directly as sparse matrices via vectorized numpy (see
_word_ids_from_coeffs / _bag_counts_sparse) -- no Python string or
Counter object is ever created per sample, which used to make this
extremely memory/time-hungry on datasets with many samples (measured:
OOM-killed, exit code -9, on InsectWingbeat's 25000 samples x 200
channels with the earlier Counter-based version) or many channels
(measured: 707s on DuckDuckGeese's 1345 channels before the matrix
itself was made sparse). If you still hit memory pressure, the levers
that matter most are window_lengths (fewer/smaller values) and stride
(see their docstrings in train_predict) -- both directly shrink the
number of DFT windows processed per (channel, window_length) config.
"""

import warnings
from collections import Counter

import numpy as np
from scipy import sparse as sp
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_selection import f_classif
from sklearn.linear_model import RidgeClassifierCV
from sklearn.preprocessing import StandardScaler


def _default_window_lengths(length):
    """A small spread of window lengths across the series length --
    override via `window_lengths` once the paper's actual grid is known."""
    candidates = [8, 16] #, 24, 32]
    return sorted({w for w in candidates if w <= length} | {min(length, 8)})


def _extract_all_windows_coefficients(X_c_1channel, window_size, stride=1):
    """
    X_c_1channel: (n_samples, length) -- one channel.
    Returns (n_samples, n_windows, n_coeffs) raw DFT coefficients
    (real/imag interleaved, exactly `window_size` values) of every
    z-normalized window, fully vectorized.

    stride : int, default 1
        NOT part of the original paper's description (which slides
        densely, stride 1) -- added as a cost knob. n_windows, and hence
        the cost of every step downstream (DFT, ANOVA F-test, word/token
        counting), scales as ~1/stride. Set to 1 to match the paper
        exactly.
    """
    n_samples, length = X_c_1channel.shape
    window_size = min(window_size, length)
    windows = np.lib.stride_tricks.sliding_window_view(X_c_1channel, window_size, axis=1)
    if stride > 1:
        windows = windows[:, ::stride, :]
    # windows: (n_samples, n_windows, window_size)
    mu = windows.mean(axis=2, keepdims=True)
    sigma = windows.std(axis=2, keepdims=True)
    sigma_safe = np.where(sigma < 1e-8, 1.0, sigma)
    windows_norm = (windows - mu) / sigma_safe

    fft = np.fft.rfft(windows_norm, axis=2)  # (n_samples, n_windows, window_size//2 + 1) complex
    coeffs = np.empty((n_samples, fft.shape[1], 2 * fft.shape[2]))
    coeffs[..., 0::2] = fft.real
    coeffs[..., 1::2] = fft.imag
    return coeffs[..., :window_size]  # exactly window_size raw coefficients, as in the paper


def _fit_selection_and_bins(coeffs_train, Y_train, word_length, alphabet_size):
    """
    LOCAL ANOVA F-value coefficient selection + equi-width bin edges from
    the selected coefficients' own min/max -- see module docstring for
    why this needs no cross-client aggregation in the vertical setting.
    coeffs_train: (n_samples, n_windows, n_coeffs).
    """
    n_samples, n_windows, n_coeffs = coeffs_train.shape
    stacked = coeffs_train.reshape(-1, n_coeffs)
    y_repeated = np.repeat(Y_train, n_windows)  # every window inherits its sample's label

    # The imaginary part of a real signal's DC (0-Hz) DFT coefficient is
    # always exactly 0 by construction -- f_classif correctly flags it as
    # constant (NaN F-value); already handled by nan_to_num below, so
    # this specific, expected, already-handled warning is silenced here
    # rather than spamming the log on every (client, channel, w_d) call.
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Features .* are constant", category=UserWarning)
        warnings.filterwarnings("ignore", message="invalid value encountered in divide", category=RuntimeWarning)
        f_values, _ = f_classif(stacked, y_repeated)
    f_values = np.nan_to_num(f_values, nan=-np.inf)
    top_idx = np.argsort(f_values)[::-1][:word_length]

    selected = stacked[:, top_idx]
    mins = selected.min(axis=0)
    maxs = selected.max(axis=0)
    bin_edges = [np.linspace(mins[k], maxs[k], alphabet_size + 1)[1:-1] for k in range(word_length)]
    return top_idx, bin_edges


# ======================================================================
# SMALL/MEDIUM-DATASET PATH -- Counter/DictVectorizer based (v2 original)
# ======================================================================
# Kept, unmodified in behaviour, specifically so results already
# collected with this path (any dataset below the size threshold --
# see LARGE_DATASET_THRESHOLD / train_predict's large_dataset_threshold)
# remain exactly reproducible without rerunning. See the module
# docstring's Performance/memory note and the conversation this split
# was introduced in for why a second, memory-heavier path exists at all
# for large datasets.

def _words_from_coeffs_small(coeffs, top_idx, bin_edges, alphabet_size):
    """coeffs: (n_samples, n_windows, n_coeffs) -> list[list[str]], one word list per sample."""
    selected = coeffs[:, :, top_idx]  # (n_samples, n_windows, word_length)
    n_samples, n_windows, word_length = selected.shape
    symbols = np.zeros(selected.shape, dtype=int)
    for k in range(word_length):
        symbols[:, :, k] = np.clip(
            np.searchsorted(bin_edges[k], selected[:, :, k], side="right"), 0, alphabet_size - 1
        )
    return [
        ["".join(str(s) for s in symbols[i, w, :]) for w in range(n_windows)]
        for i in range(n_samples)
    ]


def _unigram_bigram_bag_small(words, numerosity_reduction=True):
    """Dictionary of unigram (word) and bigram (consecutive word pair) counts."""
    if numerosity_reduction:
        reduced, last = [], None
        for w in words:
            if w != last:
                reduced.append(w)
            last = w
        words = reduced
    bag = Counter(words)  # unigrams
    for i in range(len(words) - 1):
        bag[f"{words[i]}|{words[i + 1]}"] += 1  # bigrams: neighboring word pairs
    return bag


def _extract_client_features_small(s_X_train_c, s_X_test_c, Y_train, w_grid, alphabet_size,
                                    stride, numerosity_reduction, configs_used, client_idx):
    """One client's (F_train, F_test, vocab_size) via the Counter/DictVectorizer
    path -- see the section banner above for why this is kept at all."""
    n_train_c = s_X_train_c.shape[0]
    n_test_c = s_X_test_c.shape[0]
    dims_c = s_X_train_c.shape[1]

    client_train_bags = [Counter() for _ in range(n_train_c)]
    client_test_bags = [Counter() for _ in range(n_test_c)]

    for w_d in w_grid:
        rng_wd = np.random.RandomState(w_d)
        l_d = int(rng_wd.choice([4, 5, 6]))
        configs_used.append({"client": client_idx, "window_length": w_d, "word_length": l_d,
                              "extraction_path": "small"})

        for ch in range(dims_c):
            train_coeffs = _extract_all_windows_coefficients(s_X_train_c[:, ch, :], w_d, stride=stride)
            test_coeffs = _extract_all_windows_coefficients(s_X_test_c[:, ch, :], w_d, stride=stride)

            top_idx, bin_edges = _fit_selection_and_bins(train_coeffs, Y_train, l_d, alphabet_size)

            train_words = _words_from_coeffs_small(train_coeffs, top_idx, bin_edges, alphabet_size)
            test_words = _words_from_coeffs_small(test_coeffs, top_idx, bin_edges, alphabet_size)

            prefix = f"ch{ch}_w{w_d}_"
            for i, words in enumerate(train_words):
                for token, cnt in _unigram_bigram_bag_small(words, numerosity_reduction).items():
                    client_train_bags[i][prefix + token] += cnt
            for i, words in enumerate(test_words):
                for token, cnt in _unigram_bigram_bag_small(words, numerosity_reduction).items():
                    client_test_bags[i][prefix + token] += cnt

    vectorizer = DictVectorizer(sparse=True)
    F_train_c = vectorizer.fit_transform(client_train_bags)
    F_test_c = vectorizer.transform(client_test_bags)
    return F_train_c, F_test_c, len(vectorizer.vocabulary_)


# ======================================================================
# LARGE-DATASET PATH -- vectorized, integer word ids, no Python
# dict/Counter/string ever created. See module docstring's Performance/
# memory note. Verified (see the conversation this was checked in) to
# produce IDENTICAL unigram+bigram counts to the small path above,
# except unigrams are NOT restricted to the train-observed vocabulary
# (an intentional, checked-harmless simplification -- see
# _unigram_matrix's docstring).
# ======================================================================

def _word_ids_from_coeffs(coeffs, top_idx, bin_edges, alphabet_size):
    """
    coeffs: (n_samples, n_windows, n_coeffs) -> (n_samples, n_windows) int
    array of word ids in [0, alphabet_size**word_length). Same symbol
    computation as before, just encoded as a single integer (base-
    alphabet_size positional encoding of the symbol digits) instead of a
    joined string -- a bijection with the old string form, so grouping/
    counting behaviour is identical; avoids ever creating a Python string
    per window, which is what made the old Counter-based accumulation
    (a Counter per sample, string tokens) so memory/time-hungry on
    large-n_samples datasets (measured: OOM-killed on InsectWingbeat's
    25000 samples x 200 channels, see the conversation this replaced).
    """
    selected = coeffs[:, :, top_idx]  # (n_samples, n_windows, word_length)
    n_samples, n_windows, word_length = selected.shape
    symbols = np.zeros(selected.shape, dtype=np.int64)
    for k in range(word_length):
        symbols[:, :, k] = np.clip(
            np.searchsorted(bin_edges[k], selected[:, :, k], side="right"), 0, alphabet_size - 1
        )
    powers = alphabet_size ** np.arange(word_length - 1, -1, -1)  # base-alphabet_size positional weights
    return (symbols * powers).sum(axis=2)  # (n_samples, n_windows)


def _reduce_rows(word_ids, numerosity_reduction):
    """
    word_ids: (n_samples, n_windows) -> list of 1D int arrays, one per
    sample, the numerosity-reduced sequence of word ids (collapsing
    consecutive duplicates, exactly as the original per-sample Counter
    approach did before counting).
    """
    n_samples, n_windows = word_ids.shape
    out = []
    for i in range(n_samples):
        row = word_ids[i]
        if numerosity_reduction and n_windows > 1:
            keep = np.empty(n_windows, dtype=bool)
            keep[0] = True
            keep[1:] = row[1:] != row[:-1]
            out.append(row[keep])
        else:
            out.append(row)
    return out


def _unigram_matrix(reduced_rows, vocab_size, n_samples):
    """Sparse (n_samples, vocab_size) unigram counts -- vocab_size
    (alphabet_size**word_length, at most 4**6=4096) is small enough to
    use directly as the matrix width, no remapping needed.

    Design note: unlike bigrams (see _bigram_matrices_fit_transform),
    this does NOT restrict columns to unigrams observed in train --
    every theoretically possible symbol combination gets a column for
    both train and test. This is an intentional simplification, not an
    oversight: a column with ALL-ZERO values throughout training gets
    an effectively-zero ridge coefficient regardless (nothing to learn
    from a constant-zero feature), so including it changes predictions
    negligibly while avoiding the fit/transform bookkeeping bigrams
    needed. Verified by equivalence testing against the original train-
    vocabulary-only (DictVectorizer-based) behaviour -- see the
    conversation this was checked in."""
    rows_idx, cols_idx = [], []
    for i, reduced in enumerate(reduced_rows):
        rows_idx.append(np.full(len(reduced), i, dtype=np.int64))
        cols_idx.append(reduced)
    rows_idx = np.concatenate(rows_idx) if rows_idx else np.zeros(0, dtype=np.int64)
    cols_idx = np.concatenate(cols_idx) if cols_idx else np.zeros(0, dtype=np.int64)
    mat = sp.coo_matrix(
        (np.ones(len(rows_idx)), (rows_idx, cols_idx)), shape=(n_samples, vocab_size)
    ).tocsr()
    mat.sum_duplicates()
    return mat


def _raw_bigram_ids(reduced_rows, vocab_size):
    """Per-sample raw bigram id arrays (values in [0, vocab_size**2)) --
    one array per sample, from CONSECUTIVE pairs in the already-reduced
    sequence."""
    out = []
    for reduced in reduced_rows:
        if len(reduced) > 1:
            out.append(reduced[:-1].astype(np.int64) * vocab_size + reduced[1:].astype(np.int64))
        else:
            out.append(np.zeros(0, dtype=np.int64))
    return out


def _bigram_matrices_fit_transform(reduced_rows_train, reduced_rows_test, vocab_size,
                                    n_samples_train, n_samples_test):
    """
    Fits a compact bigram vocabulary from bigrams actually OBSERVED IN
    TRAIN ONLY, remapped to 0..K-1 (K = number distinct, via np.unique --
    no Python dict needed for this step), then builds train/test sparse
    matrices under that shared mapping.

    This is what keeps matrix width bounded by what's actually observed
    (typically orders of magnitude smaller than the theoretical
    vocab_size**2 -- e.g. word_length=6, alphabet_size=4 gives a
    16.7M-wide theoretical bigram space PER (channel, window_length)
    config; concatenated across channels/configs this blew up even as a
    nominally-sparse matrix, killing the process even on tiny synthetic
    test data -- see the conversation this fit/transform split was added
    to fix). Bigrams seen only in test are dropped (standard train-
    vocabulary-only, out-of-vocabulary handling -- the same behaviour
    DictVectorizer's fit_transform/transform gave before).
    """
    train_raw = _raw_bigram_ids(reduced_rows_train, vocab_size)
    concatenated = np.concatenate(train_raw) if train_raw else np.zeros(0, dtype=np.int64)
    unique_ids, inverse = np.unique(concatenated, return_inverse=True)
    n_vocab_bigram = len(unique_ids)

    train_rows = np.concatenate([np.full(len(raw), i, dtype=np.int64) for i, raw in enumerate(train_raw)]) \
        if train_raw else np.zeros(0, dtype=np.int64)
    train_mat = sp.coo_matrix(
        (np.ones(len(train_rows)), (train_rows, inverse)), shape=(n_samples_train, n_vocab_bigram)
    ).tocsr()
    train_mat.sum_duplicates()

    # Vectorized (binary-search) vocabulary lookup for test -- avoids a
    # Python dict-lookup loop over every individual bigram occurrence.
    test_raw = _raw_bigram_ids(reduced_rows_test, vocab_size)
    test_rows, test_cols = [], []
    for i, raw in enumerate(test_raw):
        if len(raw) == 0:
            continue
        idx = np.searchsorted(unique_ids, raw)
        idx_clipped = np.clip(idx, 0, max(n_vocab_bigram - 1, 0))
        matched = (n_vocab_bigram > 0) & (unique_ids[idx_clipped] == raw) if n_vocab_bigram > 0 else np.zeros(len(raw), dtype=bool)
        matched_cols = idx_clipped[matched]
        if len(matched_cols):
            test_rows.append(np.full(len(matched_cols), i, dtype=np.int64))
            test_cols.append(matched_cols)
    test_rows = np.concatenate(test_rows) if test_rows else np.zeros(0, dtype=np.int64)
    test_cols = np.concatenate(test_cols) if test_cols else np.zeros(0, dtype=np.int64)
    test_mat = sp.coo_matrix(
        (np.ones(len(test_rows)), (test_rows, test_cols)), shape=(n_samples_test, n_vocab_bigram)
    ).tocsr()
    test_mat.sum_duplicates()

    return train_mat, test_mat


def _extract_client_features_large(s_X_train_c, s_X_test_c, Y_train, w_grid, alphabet_size,
                                    stride, numerosity_reduction, configs_used, client_idx):
    """One client's (F_train, F_test, vocab_size) via the vectorized/sparse
    path -- see the section banner above for what it avoids and why."""
    dims_c = s_X_train_c.shape[1]
    n_train_c = s_X_train_c.shape[0]
    n_test_c = s_X_test_c.shape[0]

    client_train_blocks, client_test_blocks = [], []

    for w_d in w_grid:
        rng_wd = np.random.RandomState(w_d)
        l_d = int(rng_wd.choice([4, 5, 6]))
        vocab_size = alphabet_size ** l_d
        configs_used.append({"client": client_idx, "window_length": w_d, "word_length": l_d,
                              "vocab_size_unigram": vocab_size, "extraction_path": "large"})

        for ch in range(dims_c):
            train_coeffs = _extract_all_windows_coefficients(s_X_train_c[:, ch, :], w_d, stride=stride)
            test_coeffs = _extract_all_windows_coefficients(s_X_test_c[:, ch, :], w_d, stride=stride)

            top_idx, bin_edges = _fit_selection_and_bins(train_coeffs, Y_train, l_d, alphabet_size)

            train_word_ids = _word_ids_from_coeffs(train_coeffs, top_idx, bin_edges, alphabet_size)
            test_word_ids = _word_ids_from_coeffs(test_coeffs, top_idx, bin_edges, alphabet_size)

            train_reduced = _reduce_rows(train_word_ids, numerosity_reduction)
            test_reduced = _reduce_rows(test_word_ids, numerosity_reduction)

            train_uni = _unigram_matrix(train_reduced, vocab_size, n_train_c)
            test_uni = _unigram_matrix(test_reduced, vocab_size, n_test_c)
            train_bi, test_bi = _bigram_matrices_fit_transform(
                train_reduced, test_reduced, vocab_size, n_train_c, n_test_c
            )

            client_train_blocks.extend([train_uni, train_bi])
            client_test_blocks.extend([test_uni, test_bi])

    F_train_c = sp.hstack(client_train_blocks, format="csr")
    F_test_c = sp.hstack(client_test_blocks, format="csr")
    return F_train_c, F_test_c, F_train_c.shape[1]


# Default threshold (n_train_samples * dims_c, i.e. the quantity that
# actually drives both paths' cost) below which the small/exact path is
# used. Chosen to comfortably cover every dataset already run with the
# small path at the time this was added (e.g. DuckDuckGeese: ~40 train
# samples x up to 336 channels/client =~ 13,000) while routing datasets
# like InsectWingbeat (~20,000 train samples x 50 channels/client =~
# 1,000,000) to the large path. Override via train_predict's
# large_dataset_threshold if you add a dataset that falls awkwardly
# close to this line -- check which path it took in info["extraction_path_per_client"].
LARGE_DATASET_THRESHOLD = 50_000


def train_predict(s_X_train, s_Y_train, s_X_test, s_Y_test,
                   n_classes, n_clients, n_rounds, seed,
                   window_lengths=None, alphabet_size=4, stride=1,
                   numerosity_reduction=True, alphas=None,
                   large_dataset_threshold=LARGE_DATASET_THRESHOLD, **kwargs):
    """
    window_lengths : list[int], optional
        The D window lengths w_d (see module docstring -- defaults to a
        small spread across the series length; the paper's exact grid
        wasn't specified to me). BIGGEST cost knob: cost scales ~linearly
        with len(window_lengths) -- fewer/smaller values = faster.
    alphabet_size : int, default 4
        Fixed at 4 in the original paper. Affects vocabulary size
        (alphabet_size**word_length for unigrams, squared for bigrams)
        more than extraction cost.
    stride : int, default 1
        NOT in the original paper (which slides densely) -- added as a
        cost knob, see _extract_all_windows_coefficients. Second biggest
        lever: cost scales ~1/stride. Set to 1 for paper fidelity.
    numerosity_reduction : bool, default True
        Collapse consecutive identical words before counting (standard
        BOP/BOSS/SFA practice; not explicitly mentioned in what was
        described to me -- set False to drop it if the paper doesn't
        use it). Reduces vocabulary size / downstream ridge cost, not
        the DFT/ANOVA cost itself.
    large_dataset_threshold : int, default LARGE_DATASET_THRESHOLD (50,000)
        PER CLIENT, decides which extraction path runs: if
        n_train_samples * dims_c for that client exceeds this, the
        vectorized/sparse "large" path runs (see the section banner
        above _extract_client_features_large); otherwise the original
        Counter/DictVectorizer "small" path runs UNCHANGED, so results
        already collected on datasets below the threshold stay exactly
        reproducible without rerunning (see the conversation this switch
        was added in). Set to 0 to force "large" always, or np.inf to
        force "small" always (e.g. to reproduce an old result exactly on
        a dataset that would now cross the default threshold).
    """
    if alphas is None:
        alphas = np.logspace(-3, 3, 10)

    Y_train = np.asarray(s_Y_train[0]).astype(int)  # shared across clients
    Y_test = np.asarray(s_Y_test[0]).astype(int)

    train_blocks, test_blocks, vocab_sizes, configs_used = [], [], [], []
    extraction_path_per_client = []

    for c in range(n_clients):
        n_train_c = s_X_train[c].shape[0]
        dims_c = s_X_train[c].shape[1]
        length = s_X_train[c].shape[2]
        w_grid = window_lengths or _default_window_lengths(length)

        if n_train_c * dims_c > large_dataset_threshold:
            path = "large"
            F_train_c, F_test_c, vocab_c = _extract_client_features_large(
                s_X_train[c], s_X_test[c], Y_train, w_grid, alphabet_size,
                stride, numerosity_reduction, configs_used, c,
            )
        else:
            path = "small"
            F_train_c, F_test_c, vocab_c = _extract_client_features_small(
                s_X_train[c], s_X_test[c], Y_train, w_grid, alphabet_size,
                stride, numerosity_reduction, configs_used, c,
            )

        extraction_path_per_client.append(path)
        vocab_sizes.append(vocab_c)
        train_blocks.append(F_train_c)
        test_blocks.append(F_test_c)

    # --- SIMULATED AGGREGATION --- (same caveat as fed2rc/fedst)
    F_train = sp.hstack(train_blocks, format="csr")
    F_test = sp.hstack(test_blocks, format="csr")

    # with_mean=False is required for sparse input (centering would
    # densify an already huge matrix, defeating the point) -- standard
    # practice for bag-of-words-style sparse features; this only skips
    # mean-centering, variance scaling still applies.
    scaler = StandardScaler(with_mean=False, with_std=True)
    F_train = scaler.fit_transform(F_train)
    F_test = scaler.transform(F_test)

    clf = RidgeClassifierCV(alphas=alphas)
    clf.fit(F_train, Y_train)

    def _to_proba(scores):
        if scores.ndim == 1:  # binary case: decision_function returns 1D
            scores = np.stack([-scores, scores], axis=1)
        exp = np.exp(scores - scores.max(axis=1, keepdims=True))
        return exp / exp.sum(axis=1, keepdims=True)

    train_proba = _to_proba(clf.decision_function(F_train))
    test_proba = _to_proba(clf.decision_function(F_test))

    info = {
        "window_lengths": window_lengths or "default (see _default_window_lengths)",
        "alphabet_size": alphabet_size,
        "stride": stride,
        "numerosity_reduction": numerosity_reduction,
        "large_dataset_threshold": large_dataset_threshold,
        "extraction_path_per_client": extraction_path_per_client,
        "configs_used": configs_used,
        "vocab_size_per_client": vocab_sizes,
        "best_alpha": clf.alpha_,
    }
    return train_proba, test_proba, info