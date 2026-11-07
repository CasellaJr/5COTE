"""
FedAvg-ZeroPad - Vertical variant (deliberately weaker baseline).

Alternative to fedinception_vertical.py (split learning) for the 4th HC2
slot. Implements LITERAL FedAvg (weight averaging across clients, same
architecture for everyone) rather than split learning -- see the
conversation this module came out of for the full reasoning.

Encoder choice: the backbone inside each client's local model is
selectable via `encoder` ("inception", default, or "resnet1d" -- see
ENCODER_REGISTRY / resnet1d_encoder.py). Only the encoder changes;
everything below about zero-padding / FedAvg / evaluation is identical
regardless of which one is used.

Two sub-modes, chosen automatically based on whether n_dims is evenly
divisible by n_clients (matching the requested behaviour), with an
override available:

1. "zero_padded" (used whenever channels are NOT evenly divisible, or
   whenever `force_zero_padding=True`): every client's local model has
   in_channels = n_dims_total (the full original channel count). Each
   client's own channels are placed at their ORIGINAL global column
   positions; every other column is zero. This means weight index i in
   the first conv layer always refers to the SAME physical channel for
   every client, which is what makes weight-averaging across clients at
   least positionally coherent (though clients still never train on a
   jointly-populated example -- see caveat below).

2. "compact_no_pad" (used only when channels ARE evenly divisible, and
   force_zero_padding=False): no padding at all, every client's local
   model has in_channels = dims_per_client (identical across clients, so
   shapes match). Cheaper, but WORSE than (1): here weight index i is a
   different physical channel for every client (splits are contiguous
   blocks, not interleaved), so averaging is positionally incoherent on
   top of never seeing a joint example. Included because it was asked
   for explicitly as a baseline, not because it's recommended -- pass
   force_zero_padding=True to always use mode (1) instead.

Fundamental caveat shared by both modes (the actual reason this is
offered as a *baseline*, not a competitor to split learning): no single
training example, in either mode, ever has more than one client's real
channels non-zero at the same time. The averaged global model therefore
never learns genuine cross-channel interactions between different
clients' channels -- exactly the capability split learning (see
fedinception_vertical.py) is designed to preserve. Comparing this module
against fedinception_vertical.py head-to-head is the intended ablation.

Evaluation protocol: in "zero_padded" mode, train/test probabilities are
computed by feeding the TRUE jointly-populated sample (all real channels
at once, no padding) through the final global model -- the fairest
comparison point against the other (genuinely joint) families, though
note this is out-of-distribution relative to what the model saw during
training (every training batch was zero-padded/sparse). In
"compact_no_pad" mode there is no single joint input shape shared across
clients' models, so evaluation instead averages the SAME global model's
predictions over each client's own view of the test set (an ensemble at
prediction time).
"""

import copy

import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split

from algs.inception_encoder import FusionHead
from algs.encoder_registry import build_encoder


class FullClassifier(nn.Module):
    """A single, complete classifier (encoder + head) -- unlike
    fedinception_vertical's split design, here every client holds an
    identical COPY of this whole model, not just an encoder. `encoder`
    selects the backbone via encoder_registry.py's build_encoder --
    "inception" (default) or "resnet1d" (see resnet1d_encoder.py)."""

    def __init__(self, in_channels, n_classes, encoder="inception",
                 n_filters=32, depth=6, hidden_dim=64):
        super().__init__()
        self.encoder = build_encoder(encoder, in_channels=in_channels, n_filters=n_filters, depth=depth)
        self.head = FusionHead(self.encoder.embedding_dim, n_classes, hidden_dim=hidden_dim)

    def forward(self, x):
        return self.head(self.encoder(x))


def _build_zero_padded(s_X, n_dims_total, offsets, dims_per_client):
    n_clients = len(s_X)
    n_samples = s_X[0].shape[0]
    length = s_X[0].shape[2]
    padded_list = []
    for c in range(n_clients):
        padded = np.zeros((n_samples, n_dims_total, length), dtype=np.float32)
        padded[:, offsets[c]:offsets[c] + dims_per_client[c], :] = s_X[c]
        padded_list.append(padded)
    return padded_list


def train_predict(s_X_train, s_Y_train, s_X_test, s_Y_test,
                   n_classes, n_clients, n_rounds, seed,
                   encoder="inception", n_filters=32, depth=6, hidden_dim=64,
                   local_epochs=1, batch_size=16, lr=1e-3,
                   force_zero_padding=False, device=None, verbose=True,
                   early_stopping=True, patience=10, min_delta=1e-4,
                   val_fraction=0.15, **kwargs):
    """
    encoder : {"inception", "resnet1d"}, default "inception"
        Backbone used inside FullClassifier -- InceptionTimeEncoder (see
        fedinception_vertical.py) or ResNet1DEncoder (see
        resnet1d_encoder.py). n_filters/depth only apply to "inception".

    early_stopping / patience / min_delta / val_fraction
    -----------------------------------------------------
    Same idea as fedinception_vertical.py: a validation split is carved
    out of the TRAIN set only (never test), local training each round
    only ever samples from the remaining fit set, and training stops
    once the GLOBAL (post-FedAvg-aggregation) model's validation loss
    hasn't improved by `min_delta` for `patience` consecutive rounds --
    the best round's global state is restored before predicting.

    Validation loss is measured on the SAME distribution the model is
    actually trained on each round (each client's own padded/compact
    view of the held-out samples, averaged across clients) -- NOT the
    "true jointly-populated sample" view used by the final evaluation
    below. That final-eval view is deliberately out-of-distribution
    (see module docstring: it's the fairest comparison point against
    the other, genuinely joint families) and would give a noisy,
    not-very-meaningful early-stopping signal; tracking the actual
    training distribution instead gives a clean, expected loss curve.
    """
    torch.manual_seed(seed)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    Y_train = s_Y_train[0]  # shared across clients (see family_interface.py)

    dims_per_client = [x.shape[1] for x in s_X_train]
    n_dims_total = sum(dims_per_client)
    offsets = list(np.cumsum([0] + dims_per_client[:-1]))

    evenly_divisible = len(set(dims_per_client)) == 1
    use_padding = force_zero_padding or (not evenly_divisible)

    if use_padding:
        X_train_client = _build_zero_padded(s_X_train, n_dims_total, offsets, dims_per_client)
        X_test_client = _build_zero_padded(s_X_test, n_dims_total, offsets, dims_per_client)
        model_in_channels = n_dims_total
        eval_train = np.concatenate(s_X_train, axis=1).astype(np.float32)
        eval_test = np.concatenate(s_X_test, axis=1).astype(np.float32)
    else:
        X_train_client = [x.astype(np.float32) for x in s_X_train]
        X_test_client = [x.astype(np.float32) for x in s_X_test]
        model_in_channels = dims_per_client[0]
        eval_train = None  # no shared joint input shape in this mode, see docstring
        eval_test = None

    global_model = FullClassifier(
        in_channels=model_in_channels, n_classes=n_classes, encoder=encoder,
        n_filters=n_filters, depth=depth, hidden_dim=hidden_dim
    ).to(device)

    X_train_tensors = [torch.tensor(x, dtype=torch.float32) for x in X_train_client]
    Y_train_tensor = torch.tensor(Y_train, dtype=torch.long)
    n_train_total = X_train_tensors[0].shape[0]
    all_idx = np.arange(n_train_total)

    if early_stopping:
        try:
            fit_idx, val_idx = train_test_split(
                all_idx, test_size=val_fraction, random_state=seed, stratify=Y_train
            )
        except ValueError:
            # some class has too few samples to stratify -> plain random split
            fit_idx, val_idx = train_test_split(
                all_idx, test_size=val_fraction, random_state=seed
            )
    else:
        fit_idx, val_idx = all_idx, np.array([], dtype=int)

    n_fit = len(fit_idx)
    rng = np.random.RandomState(seed)
    criterion = nn.CrossEntropyLoss()

    best_val_loss = float("inf")
    best_state = None
    rounds_without_improvement = 0
    stopped_early_at = None

    for round_idx in range(n_rounds):
        local_state_dicts = []
        for c in range(n_clients):
            local_model = copy.deepcopy(global_model)
            optimizer = torch.optim.Adam(local_model.parameters(), lr=lr)
            local_model.train()
            for _ in range(local_epochs):
                perm = fit_idx[rng.permutation(n_fit)]
                for start in range(0, n_fit, batch_size):
                    idx = perm[start:start + batch_size]
                    xb = X_train_tensors[c][idx].to(device)
                    yb = Y_train_tensor[idx].to(device)
                    optimizer.zero_grad()
                    loss = criterion(local_model(xb), yb)
                    loss.backward()
                    optimizer.step()
            local_state_dicts.append({k: v.detach().clone() for k, v in local_model.state_dict().items()})

        # --- FedAvg aggregation: uniform average of local state dicts.
        # Uniform == sample-size-weighted here since every client sees the
        # same n_fit samples (same rows, different columns). ---
        avg_state = {}
        for key in local_state_dicts[0].keys():
            stacked = torch.stack([sd[key].float() for sd in local_state_dicts], dim=0)
            avg_state[key] = stacked.mean(dim=0).to(local_state_dicts[0][key].dtype)
        global_model.load_state_dict(avg_state)

        mode_str = "padded" if use_padding else "compact"

        if early_stopping and len(val_idx) > 0:
            global_model.eval()
            with torch.no_grad():
                val_losses = []
                for c in range(n_clients):
                    xb_val = X_train_tensors[c][val_idx].to(device)
                    yb_val = Y_train_tensor[val_idx].to(device)
                    val_losses.append(criterion(global_model(xb_val), yb_val).item())
                val_loss = float(np.mean(val_losses))

            if val_loss < best_val_loss - min_delta:
                best_val_loss = val_loss
                best_state = copy.deepcopy(global_model.state_dict())
                rounds_without_improvement = 0
            else:
                rounds_without_improvement += 1

            if verbose and ((round_idx + 1) % max(1, n_rounds // 10) == 0 or round_idx == 0):
                print(f"    [FedAvg-ZeroPad/{encoder}] round {round_idx + 1}/{n_rounds} "
                      f"({mode_str} mode) - val loss {val_loss:.4f} "
                      f"(best {best_val_loss:.4f}, patience {rounds_without_improvement}/{patience})")

            if rounds_without_improvement >= patience:
                stopped_early_at = round_idx + 1
                if verbose:
                    print(f"    [FedAvg-ZeroPad/{encoder}] early stopping at round {stopped_early_at} "
                          f"(no improvement for {patience} rounds)")
                break
        else:
            if verbose and ((round_idx + 1) % max(1, n_rounds // 10) == 0 or round_idx == 0):
                print(f"    [FedAvg-ZeroPad/{encoder}] round {round_idx + 1}/{n_rounds} done ({mode_str} mode)")

    if early_stopping and best_state is not None:
        global_model.load_state_dict(best_state)

    global_model.eval()
    with torch.no_grad():
        if use_padding:
            train_in = torch.tensor(eval_train, dtype=torch.float32).to(device)
            test_in = torch.tensor(eval_test, dtype=torch.float32).to(device)
            train_proba = torch.softmax(global_model(train_in), dim=1).cpu().numpy()
            test_proba = torch.softmax(global_model(test_in), dim=1).cpu().numpy()
        else:
            # No single joint input shape exists in compact mode: average
            # the SAME global model's predictions over each client's own
            # (dims_c-wide) view -- a test-time ensemble of one set of
            # weights applied to different channel groups in turn.
            train_probas_per_client, test_probas_per_client = [], []
            for c in range(n_clients):
                xb_train = torch.tensor(X_train_client[c], dtype=torch.float32).to(device)
                xb_test = torch.tensor(X_test_client[c], dtype=torch.float32).to(device)
                train_probas_per_client.append(torch.softmax(global_model(xb_train), dim=1).cpu().numpy())
                test_probas_per_client.append(torch.softmax(global_model(xb_test), dim=1).cpu().numpy())
            train_proba = np.mean(train_probas_per_client, axis=0)
            test_proba = np.mean(test_probas_per_client, axis=0)

    info = {
        "encoder": encoder,
        "n_rounds_requested": n_rounds,
        "stopped_early_at": stopped_early_at,
        "best_val_loss": best_val_loss if early_stopping else None,
        "local_epochs": local_epochs,
        "mode": "zero_padded" if use_padding else "compact_no_pad",
        "dims_per_client": dims_per_client,
    }
    return train_proba, test_proba, info