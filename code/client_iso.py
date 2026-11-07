#!/usr/bin/env python3
"""
"Single client, its own channels, nothing else" baseline.

Same vertical split as vertical.py (same n_clients, same seed -> same
channel assignment), but each family is evaluated ONCE PER CLIENT in
ISOLATION: train_predict is called with ONLY that client's own shard
(n_clients=1 from that family's point of view, so it never concatenates
anything with any other client). No cross-client aggregation happens at
all here -- this quantifies how much accuracy you'd get stuck with if
you ignored federation entirely and just used whichever channel subset
one client happens to hold, as the lower bound to compare against your
real (all-clients-combined) results.

Usage
-----
    python run_client_isolation.py --ds_name BasicMotions --n_clients 4 --fourth_family split
    python run_client_isolation.py --ds_name EigenWorms --families stc_fedst --max_shapelet_length 200
"""

import argparse
import random
import time

import numpy as np
import wandb
from aeon.datasets import load_classification
from sklearn.metrics import accuracy_score, f1_score

from utils import safe_convert_labels, preprocess_data, split_data_vertical
from cawpe import federated_cawpe

import algs.fed2rc_vertical as fed2rc_vertical
import algs.fedst_vertical as fedst_vertical
import algs.feddict_vertical as feddict_vertical
import algs.feddrcif_vertical as feddrcif_vertical
import algs.fedinception_vertical as fedinception_vertical
import algs.fedavg_zeropad_vertical as fedavg_zeropad_vertical

FAMILY_REGISTRY = {
    "arsenal_fed2rc": fed2rc_vertical,
    "stc_fedst": fedst_vertical,
    "tde_feddict": feddict_vertical,
    "drcif_feddrcif": feddrcif_vertical,
    "fedinception": fedinception_vertical,
    "fedavg_zeropad": fedavg_zeropad_vertical,
}
FOURTH_FAMILY_TO_KEY = {"feddrcif": "drcif_feddrcif", "split": "fedinception", "fedavg": "fedavg_zeropad"}
ENCODER_AWARE_FAMILIES = {"fedinception", "fedavg_zeropad"}


def build_active_families(fourth_family):
    families = ["arsenal_fed2rc", "stc_fedst", "tde_feddict"]
    if fourth_family == "none":
        return families
    return families + [FOURTH_FAMILY_TO_KEY[fourth_family]]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ds_name", type=str, required=True)
    parser.add_argument("--extract_path", type=str, default=None)
    parser.add_argument("--n_clients", type=int, default=4,
                         help="Real number of clients the channels are split across -- same as your "
                              "vertical.py run. Each one is then evaluated alone, in turn.")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--shuffle_dims", action="store_true")
    parser.add_argument("--fourth_family", type=str, default="split",
                         choices=["split", "fedavg", "feddrcif", "none"])
    parser.add_argument("--encoder", type=str, default="inception", choices=["inception", "resnet1d"])
    parser.add_argument("--families", type=str, default=None,
                         help="Comma-separated family keys, overrides --fourth_family (same as "
                              "vertical.py's --families).")
    parser.add_argument("--n_rounds", type=int, default=100)
    parser.add_argument("--n_kernels", type=int, default=84)
    parser.add_argument("--lambda_reg", type=float, default=0.01)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--max_shapelet_length", type=float, default=None)
    parser.add_argument("--time_contract_seconds", type=float, default=None)
    parser.add_argument("--cawpe_alpha", type=float, default=4.0,
                         help="CAWPE exponent for the per-client ensemble (same as vertical.py).")
    parser.add_argument("--wandb_project", type=str, default="5COTE", help="wandb project name.")
    parser.add_argument("--wandb_entity", type=str, default="mlgroup", help="wandb entity name.")
    parser.add_argument("--wandb_mode", type=str, default="online", choices=["online", "offline", "disabled"])
    args = parser.parse_args()
    print("Resolved CLI args:", vars(args))

    rng = np.random.RandomState(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    X_train, Y_train, _ = load_classification(args.ds_name, split="Train", return_metadata=True,
                                                extract_path=args.extract_path)
    X_test, Y_test, _ = load_classification(args.ds_name, split="Test", return_metadata=True,
                                               extract_path=args.extract_path)
    Y_train, Y_test = safe_convert_labels(Y_train, Y_test)
    X_train, Y_train, X_test, Y_test = preprocess_data(X_train, Y_train, X_test, Y_test)

    s_X_train, s_Y_train, s_X_test, s_Y_test, client_dim_indices = split_data_vertical(
        X_train, Y_train, X_test, Y_test, args.n_clients, rng=rng, shuffle_dims=args.shuffle_dims
    )
    for c, idx in enumerate(client_dim_indices):
        print(f"  client {c}: channels {list(idx)} ({len(idx)} total)")

    if args.families is not None:
        active_families = [f.strip() for f in args.families.split(",") if f.strip()]
    else:
        active_families = build_active_families(args.fourth_family)
    print("Active families:", active_families)

    run_name = f"{args.ds_name}_seed{args.seed}_client_isolation"
    wandb.init(
        project=args.wandb_project, entity=args.wandb_entity, mode=args.wandb_mode, name=run_name,
        config={
            "dataset": args.ds_name, "seed": args.seed, "n_clients": args.n_clients,
            "experiment_type": "client_isolation", "active_families": active_families,
            "fourth_family": args.fourth_family, "encoder": args.encoder,
            "channel_distribution": "shuffled" if args.shuffle_dims else "contiguous",
            "client_dim_indices": {str(c): [int(i) for i in idx] for c, idx in enumerate(client_dim_indices)},
            "n_dims_total": int(X_train.shape[1]), "ts_length": int(X_train.shape[2]),
        },
    )

    wandb_metrics = {}
    n_classes = len(np.unique(Y_train))
    # per-client storage, used to build one CAWPE ensemble PER CLIENT
    # (that client's own channels only, no cross-client information)
    iso_names = []                     # display names, in family order
    iso_test_proba = {}                # (display_name, client) -> test probabilities
    iso_test_acc = {}                  # (display_name, client) -> test accuracy (CAWPE weight)

    for family in active_families:
        module = FAMILY_REGISTRY[family]
        display_name = None

        for c in range(args.n_clients):
            call_kwargs = dict(
                n_classes=n_classes, n_clients=1, n_rounds=args.n_rounds,
                seed=args.seed + c, n_kernels=args.n_kernels, lambda_reg=args.lambda_reg,
                encoder=args.encoder, batch_size=args.batch_size,
                n_cv_folds=max(2, args.n_clients),
            )
            if args.device != "auto":
                call_kwargs["device"] = args.device
            if args.max_shapelet_length is not None:
                call_kwargs["max_shapelet_length"] = args.max_shapelet_length
            if args.time_contract_seconds is not None:
                call_kwargs["time_contract_seconds"] = args.time_contract_seconds

            t0 = time.perf_counter()
            # ONLY this client's shard, wrapped as the sole element -- the
            # family module never sees any other client's channels, so no
            # cross-client aggregation can happen (see module docstring).
            train_proba, test_proba, info = module.train_predict(
                [s_X_train[c]], [Y_train], [s_X_test[c]], [Y_test], **call_kwargs
            )
            elapsed = time.perf_counter() - t0

            if display_name is None:
                display_name = f"{family}[{info['encoder']}]" if "encoder" in info else family

            train_acc = accuracy_score(Y_train, np.argmax(train_proba, axis=1))
            train_f1 = f1_score(Y_train, np.argmax(train_proba, axis=1), average="macro")
            test_acc = accuracy_score(Y_test, np.argmax(test_proba, axis=1))
            test_f1 = f1_score(Y_test, np.argmax(test_proba, axis=1), average="macro")
            n_channels_c = len(client_dim_indices[c])
            iso_test_proba[(display_name, c)] = test_proba
            iso_test_acc[(display_name, c)] = test_acc

            print(f"  {display_name} / client {c} ({n_channels_c} channels): "
                  f"test_acc={test_acc:.4f} test_f1={test_f1:.4f} ({elapsed:.1f}s)")

            wandb_metrics.update({
                f"{display_name}/client{c}/train_acc": train_acc,
                f"{display_name}/client{c}/train_f1": train_f1,
                f"{display_name}/client{c}/test_acc": test_acc,
                f"{display_name}/client{c}/test_f1": test_f1,
                f"{display_name}/client{c}/time_seconds": elapsed,
                f"{display_name}/client{c}/n_channels": n_channels_c,
            })

        # Summary across clients for this family: best/worst/mean isolated
        # accuracy -- the "worst" number is the one to quote as the lower
        # bound against your real multi-client result for the same family.
        per_client_test_acc = [wandb_metrics[f"{display_name}/client{c}/test_acc"] for c in range(args.n_clients)]
        wandb_metrics[f"{display_name}/isolation_best_test_acc"] = max(per_client_test_acc)
        wandb_metrics[f"{display_name}/isolation_worst_test_acc"] = min(per_client_test_acc)
        wandb_metrics[f"{display_name}/isolation_mean_test_acc"] = float(np.mean(per_client_test_acc))
        per_client_test_f1 = [wandb_metrics[f"{display_name}/client{c}/test_f1"] for c in range(args.n_clients)]
        wandb_metrics[f"{display_name}/isolation_best_test_f1"] = max(per_client_test_f1)
        wandb_metrics[f"{display_name}/isolation_mean_test_f1"] = float(np.mean(per_client_test_f1))
        iso_names.append(display_name)

    # --- Per-client ENSEMBLE (same CAWPE as 5COTE, but each client alone) ---
    # Directly comparable with the federated ensemble: same families, same
    # weighting rule (test accuracy of each family on that client's model).
    if len(iso_names) > 1:
        ens_acc, ens_f1 = [], []
        for c in range(args.n_clients):
            pred, _, w = federated_cawpe(
                iso_names, [iso_test_acc[(n, c)] for n in iso_names],
                [iso_test_proba[(n, c)] for n in iso_names], alpha=args.cawpe_alpha,
            )
            a, f = accuracy_score(Y_test, pred), f1_score(Y_test, pred, average="macro")
            ens_acc.append(a); ens_f1.append(f)
            print(f"  ENSEMBLE / client {c}: test_acc={a:.4f} test_f1={f:.4f} weights={w}")
            wandb_metrics[f"isolation_ensemble/client{c}/test_acc"] = a
            wandb_metrics[f"isolation_ensemble/client{c}/test_f1"] = f
        # "best client" = best over clients of each metric separately (an
        # oracle choice made on the test set: it favours the baseline).
        wandb_metrics["isolation_ensemble/best_test_acc"] = max(ens_acc)
        wandb_metrics["isolation_ensemble/best_test_f1"] = max(ens_f1)
        wandb_metrics["isolation_ensemble/mean_test_acc"] = float(np.mean(ens_acc))
        wandb_metrics["isolation_ensemble/mean_test_f1"] = float(np.mean(ens_f1))

    wandb.log(wandb_metrics)
    wandb.finish()


if __name__ == "__main__":
    main()