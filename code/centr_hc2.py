#!/usr/bin/env python3
"""
Centralized (non-federated) HIVE-COTE 2.0 baseline, via sktime, logged to
wandb -- the upper-bound reference for your vertical-federated results:
same datasets/splits, but ONE model sees the full multivariate series,
no channel split, no federation. Lets you report "cost of verticality"
(centralized accuracy - your ensemble's accuracy).

Usage
-----
    python run_hivecote_centralized.py --datasets BasicMotions ArticularyWordRecognition --seeds 0 1 2 3 4
    python run_hivecote_centralized.py --datasets EigenWorms --time_limit_in_minutes 30
"""

import argparse
import time

import wandb
from aeon.datasets import load_classification
from sklearn.metrics import accuracy_score, f1_score
from sktime.classification.hybrid import HIVECOTEV2

from utils import safe_convert_labels, preprocess_data


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--datasets", nargs="+", required=True, help="UEA/UCR dataset names.")
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument("--extract_path", type=str, default=None,
                         help="Same meaning as in vertical.py -- set if a dataset name collides "
                              "between archives (e.g. Phoneme).")
    parser.add_argument("--n_jobs", type=int, default=-1, help="Passed to HIVECOTEV2 (-1 = all cores).")
    parser.add_argument("--time_limit_in_minutes", type=float, default=240,
                         help="HIVECOTEV2's own contract budget (0 = unbounded, the default -- can "
                              "be slow on larger datasets; set e.g. 30-60 to cap it, same spirit as "
                              "the paper's own contracted experiments).")
    parser.add_argument("--wandb_project", type=str, default="5COTE", help="wandb project name.")
    parser.add_argument("--wandb_entity", type=str, default="mlgroup", help="wandb entity name.")
    parser.add_argument("--wandb_mode", type=str, default="online", choices=["online", "offline", "disabled"])
    args = parser.parse_args()

    for ds_name in args.datasets:
        print(f"\n=== Loading {ds_name} ===")
        X_train, Y_train, _ = load_classification(ds_name, split="Train", return_metadata=True,
                                                    extract_path=args.extract_path)
        X_test, Y_test, _ = load_classification(ds_name, split="Test", return_metadata=True,
                                                   extract_path=args.extract_path)
        Y_train, Y_test = safe_convert_labels(Y_train, Y_test)
        X_train, Y_train, X_test, Y_test = preprocess_data(X_train, Y_train, X_test, Y_test)

        for seed in args.seeds:
            run_name = f"{ds_name}_seed{seed}_hivecote2_centralized"
            wandb.init(
                project=args.wandb_project, entity=args.wandb_entity, mode=args.wandb_mode, name=run_name,
                config={
                    "dataset": ds_name, "seed": seed,
                    "active_families": ["hivecote2_centralized"],
                    "experiment_type": "centralized_baseline",
                    "n_train_samples": X_train.shape[0], "n_test_samples": X_test.shape[0],
                    "n_dims_total": X_train.shape[1], "ts_length": X_train.shape[2],
                    "time_limit_in_minutes": args.time_limit_in_minutes,
                },
            )

            clf = HIVECOTEV2(stc_params={'n_shapelet_samples': 300},
                            drcif_params={'n_estimators': 50},
                            arsenal_params={'n_estimators': 10, 'num_kernels': 100},
                            tde_params={'n_parameter_samples': 100, 'max_ensemble_size': 25},
                            time_limit_in_minutes=args.time_limit_in_minutes,
                            n_jobs=args.n_jobs, random_state=seed)

            t0 = time.perf_counter()
            clf.fit(X_train, Y_train)
            train_pred = clf.predict(X_train)
            test_pred = clf.predict(X_test)
            elapsed = time.perf_counter() - t0

            train_acc = accuracy_score(Y_train, train_pred)
            train_f1 = f1_score(Y_train, train_pred, average="macro")
            test_acc = accuracy_score(Y_test, test_pred)
            test_f1 = f1_score(Y_test, test_pred, average="macro")

            print(f"[{ds_name} seed={seed}] test_acc={test_acc:.4f} test_f1={test_f1:.4f} "
                  f"({elapsed:.1f}s)")

            wandb.log({
                "hivecote2_centralized/train_acc": train_acc,
                "hivecote2_centralized/train_f1": train_f1,
                "hivecote2_centralized/test_acc": test_acc,
                "hivecote2_centralized/test_f1": test_f1,
                "hivecote2_centralized/time_seconds": elapsed,
            })
            wandb.finish()


if __name__ == "__main__":
    main()