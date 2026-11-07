"""
FedInceptionTime - Vertical variant, split-learning design.

(Despite the module name kept for continuity, the encoder is now
pluggable -- see `encoder` below; "FedInceptionTime" is the default
config, not the only one this module can run.)

Fills the 4th HC2 slot (alternative to feddrcif_vertical.py) with a
genuinely end-to-end-trained deep model, using SPLIT LEARNING rather
than FedAvg -- see the conversation this module came out of for why
literal weight-averaging isn't well-posed here (client encoders operate
on different, non-comparable channel subsets; averaging their weights
has no principled interpretation, independent of whether n_dims is
evenly divisible by n_clients).

Architecture
------------
- Each client c owns its own encoder (built via encoder_registry.py's
  build_encoder -- InceptionTimeEncoder by default, or ResNet1DEncoder
  via `encoder="resnet1d"`, see resnet1d_encoder.py) that maps its own
  channel shard (dims_c, length) to a fixed-size embedding h_c. Which
  encoder is used doesn't change anything below -- the split-learning
  protocol only cares that every encoder exposes `.embedding_dim` and
  maps (batch, dims_c, length) -> (batch, embedding_dim).
- The active party owns a small FusionHead (2-layer MLP, see
  inception_encoder.py) that maps the CONCATENATION [h_1 | ... | h_n] to
  class logits, and is the only party that needs the labels.
- Forward: every client encodes its own shard -> embeddings are
  concatenated -> fusion head -> logits -> loss (active party only).
- Backward: gradients flow back through the fusion head (active party,
  local), then split at the concatenation point into per-client slices
  dL/dh_c, each routed back to update only that client's own encoder.

IMPORTANT -- single-process simulation caveat: this implementation runs
everything in one Python process and lets autograd build ONE graph
across all "parties", so calling `loss.backward()` automatically
computes the correct per-client gradients -- functionally equivalent to
the real multi-party protocol (server sends dL/dh_c back to client c,
client c backprops through its own encoder), but without actually
serializing/transmitting embeddings and gradients over a network. For a
true distributed deployment you'd cut the graph explicitly at
`torch.cat(embeddings, dim=1)`: each client sends `embeddings[c]`
(detached) forward, the active party sends `embeddings[c].grad` back,
and each client calls `.backward()` locally on its own encoder using
that received gradient as `grad_output`.

Privacy note: the active party sees per-sample embeddings (not raw
data), and each client only ever sees its own gradient slice -- better
than the plain feature concatenation used in fed2rc/fedst/feddict
(which requires a trusted aggregator to see full local feature blocks),
but not privacy-preserving in a formal sense: embeddings can still leak
information about the underlying signal. Worth flagging explicitly
alongside the other "SIMULATED AGGREGATION" caveats in this codebase.

Dependency note: this module needs PyTorch (not required by any other
module in this codebase). Install with e.g. `pip install torch`.
"""

import copy

import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split

from algs.inception_encoder import FusionHead
from algs.encoder_registry import build_encoder


class VerticalSplitModel(nn.Module):
    def __init__(self, client_in_channels, n_classes, encoder="inception",
                 n_filters=32, depth=6, hidden_dim=64):
        super().__init__()
        self.encoders = nn.ModuleList([
            build_encoder(encoder, in_channels=c, n_filters=n_filters, depth=depth)
            for c in client_in_channels
        ])
        total_embed_dim = sum(enc.embedding_dim for enc in self.encoders)
        self.head = FusionHead(total_embed_dim, n_classes, hidden_dim=hidden_dim)

    def forward(self, x_list):
        """x_list: list of tensors, one per client, each (batch, dims_c, length)."""
        embeddings = [enc(x_c) for enc, x_c in zip(self.encoders, x_list)]
        # --- THE "WIRE" BETWEEN CLIENTS AND ACTIVE PARTY ---
        # See module docstring: in a real deployment this concatenation
        # is where embeddings physically cross from client machines to
        # the active party, and where the backward pass would need to
        # explicitly send dL/d(embeddings[c]) back to client c.
        fused = torch.cat(embeddings, dim=1)
        return self.head(fused)


def train_predict(s_X_train, s_Y_train, s_X_test, s_Y_test,
                   n_classes, n_clients, n_rounds, seed,
                   encoder="inception", n_filters=32, depth=6, hidden_dim=64,
                   batch_size=16, lr=1e-3, device=None, verbose=True,
                   early_stopping=True, patience=10, min_delta=1e-4,
                   val_fraction=0.15, **kwargs):
    """
    encoder : {"inception", "resnet1d"}, default "inception"
        Per-client backbone, see encoder_registry.py / resnet1d_encoder.py.
        n_filters/depth only apply to "inception".
    early_stopping / patience / min_delta / val_fraction
    -----------------------------------------------------
    A validation split is carved out of the TRAIN set only (never the
    test set, to avoid leakage / implicit model selection on test data).
    If `early_stopping=True`, training stops once validation loss hasn't
    improved by at least `min_delta` for `patience` consecutive rounds,
    and the best-validation-loss weights are restored before predicting.
    Stratification by class is attempted first; if any class has too few
    samples to stratify, falls back to a plain random split.
    """
    torch.manual_seed(seed)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    Y_train = s_Y_train[0]  # shared across clients (see family_interface.py)

    client_in_channels = [s_X_train[c].shape[1] for c in range(n_clients)]

    if verbose:
        print(f"    [FedInceptionTime-split] encoder={encoder!r}, "
              f"client_in_channels={client_in_channels}")

    model = VerticalSplitModel(
        client_in_channels, n_classes, encoder=encoder,
        n_filters=n_filters, depth=depth, hidden_dim=hidden_dim
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    X_train_tensors = [torch.tensor(s_X_train[c], dtype=torch.float32) for c in range(n_clients)]
    X_test_tensors = [torch.tensor(s_X_test[c], dtype=torch.float32) for c in range(n_clients)]
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

    best_val_loss = float("inf")
    best_state = None
    rounds_without_improvement = 0
    stopped_early_at = None

    for round_idx in range(n_rounds):
        model.train()
        perm = fit_idx[rng.permutation(n_fit)]
        epoch_loss = 0.0
        for start in range(0, n_fit, batch_size):
            idx = perm[start:start + batch_size]
            batch_x = [X_train_tensors[c][idx].to(device) for c in range(n_clients)]
            batch_y = Y_train_tensor[idx].to(device)

            optimizer.zero_grad()
            logits = model(batch_x)
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * len(idx)
        epoch_loss /= n_fit

        if early_stopping and len(val_idx) > 0:
            model.eval()
            with torch.no_grad():
                val_x = [X_train_tensors[c][val_idx].to(device) for c in range(n_clients)]
                val_y = Y_train_tensor[val_idx].to(device)
                val_logits = model(val_x)
                val_loss = criterion(val_logits, val_y).item()

            if val_loss < best_val_loss - min_delta:
                best_val_loss = val_loss
                best_state = copy.deepcopy(model.state_dict())
                rounds_without_improvement = 0
            else:
                rounds_without_improvement += 1

            if verbose and ((round_idx + 1) % max(1, n_rounds // 10) == 0 or round_idx == 0):
                print(f"    [FedInceptionTime-split/{encoder}] round {round_idx + 1}/{n_rounds} "
                      f"- train loss {epoch_loss:.4f} - val loss {val_loss:.4f} "
                      f"(best {best_val_loss:.4f}, patience {rounds_without_improvement}/{patience})")

            if rounds_without_improvement >= patience:
                stopped_early_at = round_idx + 1
                if verbose:
                    print(f"    [FedInceptionTime-split/{encoder}] early stopping at round {stopped_early_at} "
                          f"(no improvement for {patience} rounds)")
                break
        else:
            if verbose and ((round_idx + 1) % max(1, n_rounds // 10) == 0 or round_idx == 0):
                print(f"    [FedInceptionTime-split/{encoder}] round {round_idx + 1}/{n_rounds} "
                      f"- train loss {epoch_loss:.4f}")

    if early_stopping and best_state is not None:
        model.load_state_dict(best_state)

    model.eval()
    with torch.no_grad():
        train_logits = model([t.to(device) for t in X_train_tensors])
        test_logits = model([t.to(device) for t in X_test_tensors])
        train_proba = torch.softmax(train_logits, dim=1).cpu().numpy()
        test_proba = torch.softmax(test_logits, dim=1).cpu().numpy()

    info = {
        "encoder": encoder,
        "n_rounds_requested": n_rounds,
        "stopped_early_at": stopped_early_at,
        "best_val_loss": best_val_loss if early_stopping else None,
        "embedding_dims_per_client": [enc.embedding_dim for enc in model.encoders],
    }
    return train_proba, test_proba, info