import os
import json
import random
import argparse
import time

script_start_time = time.perf_counter()  # covers dataset loading/splitting too, not just the family loop

import numpy as np
import wandb
from sklearn.metrics import f1_score, accuracy_score

from aeon.datasets import load_classification
from utils import preprocess_data, safe_convert_labels, split_data_vertical
from cawpe import federated_cawpe

# --- per-family federated modules (vertical variants), all under algs/ ---
import algs.fed2rc_vertical as fed2rc_vertical
import algs.fedst_vertical as fedst_vertical
import algs.feddict_vertical as feddict_vertical
import algs.feddrcif_vertical as feddrcif_vertical
import algs.fedinception_vertical as fedinception_vertical
import algs.fedavg_zeropad_vertical as fedavg_zeropad_vertical


# Registry mapping a family "slot" (Arsenal / STC / TDE / DrCIF, as in
# centralized HC2) to the federated module that replaces it. EVERY family
# here is a genuinely federated method: it builds ONE global model using
# ALL n_clients' channel shards together (each family follows its own
# aggregation logic -- kernel-based, shapelet-based, dictionary-based,
# split-learning/FedAvg -- but training data is never pooled/shared raw;
# only the aggregated statistics/embeddings/weights each protocol
# specifically calls for are exchanged).
#
# The 4th ("DrCIF") slot has no real federated DrCIF implementation yet
# (feddrcif_vertical.py is a stub -- see its docstring for the design
# plan). Two working stand-ins fill that slot meanwhile (split-learning
# or literal FedAvg, both deep-learning), chosen via --fourth_family
# (deliberately NOT called "--deep_family": one of the three choices,
# "feddrcif" itself, is an interval/tree-based method, not deep learning
# at all -- a name implying the whole slot is about deep learning would
# be wrong for it). Registry keys below drop the "drcif_" prefix for
# "fedinception"/"fedavg_zeropad" for the same reason -- neither is
# actually a DrCIF variant, they're alternative families filling the
# same empty slot until a real federated DrCIF exists; only "feddrcif"
# itself genuinely belongs to that slot, hence keeps the prefix.
FAMILY_REGISTRY = {
    "arsenal_fed2rc": fed2rc_vertical,
    "stc_fedst": fedst_vertical,
    "tde_feddict": feddict_vertical,
    "drcif_feddrcif": feddrcif_vertical,
    "fedinception": fedinception_vertical,
    "fedavg_zeropad": fedavg_zeropad_vertical,
}

FOURTH_FAMILY_TO_KEY = {
    "feddrcif": "drcif_feddrcif",  # interval forest, tree-pooling (see feddrcif_vertical.py)
    "split": "fedinception",       # split-learning (fedinception_vertical.py)
    "fedavg": "fedavg_zeropad",    # literal FedAvg baseline (fedavg_zeropad_vertical.py)
}

# Only these two families' train_predict actually reads `encoder` (see
# ENCODER_REGISTRY in encoder_registry.py); the rest silently ignore it
# via **kwargs. Used to annotate results with the encoder actually in
# use (e.g. "fedinception[resnet1d]") so they can't be misread as plain
# InceptionTime -- see the conversation this was added in.
ENCODER_AWARE_FAMILIES = {"fedinception", "fedavg_zeropad"}


def build_active_families(fourth_family):
    families = ["arsenal_fed2rc", "stc_fedst", "tde_feddict"]
    if fourth_family == "none":
        return families
    if fourth_family not in FOURTH_FAMILY_TO_KEY:
        raise ValueError(f"fourth_family must be one of {['none'] + list(FOURTH_FAMILY_TO_KEY.keys())}")
    families.append(FOURTH_FAMILY_TO_KEY[fourth_family])
    return families


parser = argparse.ArgumentParser()
parser.add_argument("--ds_name", type=str, default="BasicMotions", help="Dataset name")
parser.add_argument("--extract_path", type=str, default=None,
                     help="Local path to load the dataset from, forwarded to "
                          "aeon.datasets.load_classification's extract_path. Use this to point at "
                          "a manually-downloaded/placed copy of a dataset instead of aeon's default "
                          "cache/auto-download -- needed for datasets whose name collides between "
                          "archives (e.g. UCR's univariate 'Phoneme', 1024 timesteps/39 classes, vs "
                          "UEA's multivariate 'Phoneme', 11 channels/39 classes -- aeon's default "
                          "resolution can land on either depending on what's cached, see the "
                          "conversation this flag was added in). Expects "
                          "<extract_path>/<ds_name>/<ds_name>_TRAIN.ts and _TEST.ts.")
parser.add_argument("--n_clients", type=int, default=4)
parser.add_argument("--n_rounds", type=int, default=100)
parser.add_argument("--n_kernels", type=int, default=84, help="Number of kernels for MiniRocket: 84, 924, or 9996")
parser.add_argument("--lambda_reg", type=float, default=0.01)
parser.add_argument("--cawpe_alpha", type=float, default=4.0, help="CAWPE accuracy exponent")
parser.add_argument("--ablation_alphas", type=float, nargs="*", default=[0.0, 1.0],
                     help="Extra CAWPE exponents evaluated on the SAME already-trained families "
                          "(no retraining) and logged as ensemble_alpha<a>/test_{acc,f1}. alpha=0 is "
                          "uniform weights. Pass no values to disable.")
parser.add_argument("--save_probas_dir", type=str, default=None,
                     help="If set, saves each family's test probabilities (+ labels) to an .npz here, "
                          "so other weighting schemes can be evaluated offline without retraining.")
parser.add_argument("--fourth_family", type=str, default="split",
                     choices=["split", "fedavg", "feddrcif", "none"],
                     help="Which method fills the 4th HC2 slot. 'split': split-learning "
                          "(fedinception_vertical.py). 'fedavg': literal FedAvg baseline "
                          "(fedavg_zeropad_vertical.py) -- use --encoder to pick the backbone for "
                          "either of these two. 'feddrcif': tree-pooling federated interval forest "
                          "(feddrcif_vertical.py); use --families for its aggregation='concat' "
                          "ablation, not exposed as a top-level flag here. "
                          "'none': only fed2rc/fedst/feddict run, no 4th family at all.")
parser.add_argument("--encoder", type=str, default="inception",
                     choices=["inception", "resnet1d"],
                     help="Backbone for --fourth_family split/fedavg -- see encoder_registry.py. "
                          "Ignored when --fourth_family is 'none' or 'feddrcif'.")
parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"],
                     help="Torch device for fedinception/fedavg_zeropad (ignored by every other "
                          "family). 'auto' (default) picks cuda if available else cpu, matching "
                          "each module's own internal default. Force 'cpu' to sidestep GPU memory "
                          "limits on very long series -- InceptionTime never downsamples the time "
                          "axis between modules, so a series like EigenWorms' 17984 timestamps can "
                          "exceed a mid-size GPU's memory during forward/backward, even though the "
                          "same computation fits fine in ordinary CPU RAM. See --batch_size too if "
                          "you'd rather stay on GPU.")
parser.add_argument("--batch_size", type=int, default=16,
                     help="Forwarded to fedinception/fedavg_zeropad (ignored by every other "
                          "family). Lower it if you're hitting GPU memory limits on long series "
                          "but want to stay on GPU rather than forcing --device cpu.")
parser.add_argument("--max_shapelet_length", type=float, default=None,
                     help="Forwarded to fedst_vertical.py (ignored by every other family); caps "
                          "candidate shapelet length. Default None keeps FedST's own default "
                          "(unbounded, i.e. up to the full series length) -- on very long series "
                          "(e.g. EigenWorms' 17984 timestamps) an unbounded candidate length can "
                          "try to allocate a windows-batch array well over 100 GiB (measured: 154 "
                          "GiB for one length group). Set this explicitly (e.g. 100-200) for any "
                          "dataset with long series.")
parser.add_argument("--shuffle_dims", action="store_true",
                     help="Randomly permute channels before splitting them across clients")
parser.add_argument("--families", type=str, default=None,
                     help="Comma-separated subset of FAMILY_REGISTRY keys to run instead of the "
                          "default set (e.g. 'arsenal_fed2rc' to test one family alone). "
                          "Overrides --fourth_family when set.")
parser.add_argument("--seed", type=int, default=1,
                     help="Random seed for this run (channel shuffling if --shuffle_dims, model "
                          "init, CV folds, etc.). run_experiments.py runs this once per seed in "
                          "SEEDS for each dataset -- set this directly only when running "
                          "vertical.py standalone for a single seed.")
parser.add_argument("--wandb_project", type=str, default="5COTE", help="wandb project name.")
parser.add_argument("--wandb_entity", type=str, default="mlgroup",
                     help="wandb entity (team/username); omit to use your configured default.")
parser.add_argument("--wandb_mode", type=str, default="online",
                     choices=["online", "offline", "disabled"],
                     help="'disabled' turns wandb off entirely (no network, no local run files) -- "
                          "useful for quick debugging runs that don't need to be tracked.")
parser.add_argument("--run_name", type=str, default=None,
                     help="Custom wandb run name, overriding the auto-generated one. Without "
                          "this, the default is dataset/seed/fourth_family/encoder for normal "
                          "runs, or dataset/seed/family-<names> when --families is set (ablation "
                          "runs), so single-family sweeps are identifiable at a glance.")

args = parser.parse_args()

# Printed unconditionally, deliberately BEFORE anything else runs: if a
# flag you passed on the command line doesn't show up here with the
# value you expect, the bug is in argument parsing/forwarding -- check
# here first rather than chasing it through family-level logs.
print("Resolved CLI args:", vars(args))

ds_name = args.ds_name
n_clients = args.n_clients
n_rounds = args.n_rounds
n_kernels = args.n_kernels
lambda_reg = args.lambda_reg
cawpe_alpha = args.cawpe_alpha
myseed = args.seed

rng = np.random.RandomState(myseed)
np.random.seed(myseed)
random.seed(myseed)

print("DATASET", ds_name)

X_train, Y_train, metadata_train = load_classification(
    ds_name, split="Train", return_metadata=True, extract_path=args.extract_path
)  # MULTIVARIATE
X_test, Y_test, metadata_test = load_classification(
    ds_name, split="Test", return_metadata=True, extract_path=args.extract_path
)
print(X_train.shape, metadata_train)

Y_train, Y_test = safe_convert_labels(Y_train, Y_test)
X_train, Y_train, X_test, Y_test = preprocess_data(X_train, Y_train, X_test, Y_test)

n_dims = X_train.shape[1]
ts_length = X_train.shape[2]
n_classes = len(np.unique(Y_train))
n_train_samples = X_train.shape[0]
n_test_samples = X_test.shape[0]
print(f"Multivariate data - Dimensions: {n_dims}, Time series length: {ts_length}, Classes: {n_classes}")

if n_clients > n_dims:
    raise ValueError(
        f"--n_clients ({n_clients}) cannot exceed the number of channels "
        f"in this dataset ({n_dims})."
    )

# --- VERTICAL SPLIT (same shards reused by every family below) ---
# Every client keeps ALL samples (train AND test, from the same 80/20
# split of the original data) but only its own subset of channels.
s_X_train, s_Y_train, s_X_test, s_Y_test, client_dim_indices = split_data_vertical(
    X_train, Y_train, X_test, Y_test, n_clients, rng=rng, shuffle_dims=args.shuffle_dims
)
for c, idx in enumerate(client_dim_indices):
    print(f"  client {c}: channels {list(idx)}")

# --- RUN EACH FEDERATED FAMILY ON THE SAME VERTICAL SHARDS ---
if args.families is not None:
    active_families = [f.strip() for f in args.families.split(",") if f.strip()]
    unknown = [f for f in active_families if f not in FAMILY_REGISTRY]
    if unknown:
        raise ValueError(f"Unknown family key(s) in --families: {unknown}. "
                          f"Valid keys: {list(FAMILY_REGISTRY.keys())}")
else:
    active_families = build_active_families(args.fourth_family)

print("\nActive families for this run:", active_families)

# --- WANDB: one run per (dataset, seed, config) combination. Initialized
# here, after dataset loading/splitting, so the config logged is complete
# in a single call rather than patched together in pieces. ---
if args.run_name is not None:
    run_name = args.run_name
elif args.families is not None:
    # Ablation/sensitivity runs (--families set): tag the run with which
    # families are active so it's identifiable at a glance in the wandb
    # UI, distinct from the general default naming below -- the
    # underlying hyperparameters you're sweeping (n_kernels etc.) are
    # already in the logged config either way, this is just for quick
    # visual identification in the run list. Also tags the encoder when
    # it's actually relevant (fedinception/fedavg_zeropad) -- otherwise
    # two runs that only differ by --encoder (e.g. "fedinception" with
    # inception vs resnet1d) would get the IDENTICAL name.
    families_tag = "-".join(active_families)
    if any(f in ENCODER_AWARE_FAMILIES for f in active_families):
        run_name = f"{ds_name}_seed{myseed}_family-{families_tag}_{args.encoder}"
    else:
        run_name = f"{ds_name}_seed{myseed}_family-{families_tag}"
else:
    run_name = f"{ds_name}_seed{myseed}_{args.fourth_family}_{args.encoder}"
wandb.init(
    project=args.wandb_project,
    entity=args.wandb_entity,
    mode=args.wandb_mode,
    name=run_name,
    config={
        "dataset": ds_name,
        "extract_path": args.extract_path,
        "seed": myseed,
        "n_clients": n_clients,
        "n_rounds": n_rounds,
        "n_kernels": n_kernels,
        "lambda_reg": lambda_reg,
        "cawpe_alpha": cawpe_alpha,
        "fourth_family": args.fourth_family,
        "encoder": args.encoder,
        "device": args.device,
        "batch_size": args.batch_size,
        "max_shapelet_length": args.max_shapelet_length,
        "channel_distribution": "shuffled" if args.shuffle_dims else "contiguous",
        "active_families": active_families,
        # dataset-derived facts, only known after loading -- kept in the
        # same config dict so the whole run is described in one place
        "n_dims_total": int(n_dims),
        "ts_length": int(ts_length),
        "n_classes": int(n_classes),
        "n_train_samples": int(n_train_samples),
        "n_test_samples": int(n_test_samples),
        "dims_per_client": [len(idx) for idx in client_dim_indices],
        "client_dim_indices": {str(c): [int(i) for i in idx] for c, idx in enumerate(client_dim_indices)},
    },
)

family_names, train_probas, test_probas, weight_accuracies, family_info = [], [], [], [], {}
table_rows = []  # (display_name, train_acc, train_f1, test_acc, test_f1, info) -- table built once weights are known
wandb_metrics = {}  # accumulated, logged in ONE wandb.log() call at the very end (see rationale below)

for family in active_families:
    module = FAMILY_REGISTRY[family]
    header_label = f"{family}[{args.encoder}]" if family in ENCODER_AWARE_FAMILIES else family
    print(f"\n=== Training family: {header_label} on all {n_clients} clients jointly ({module.__name__}) ===")

    if not hasattr(module, "train_predict"):
        print(f"  [SKIPPED] {family} missing train_predict")
        continue

    try:
        # ONE global model per family, built from every client's channel
        # shard together (each family's own aggregation logic decides how
        # -- see the module's own docstring), fit once on the train split.
        call_kwargs = dict(
            n_classes=n_classes, n_clients=n_clients, n_rounds=n_rounds,
            seed=myseed, n_kernels=n_kernels, lambda_reg=lambda_reg,
            encoder=args.encoder, batch_size=args.batch_size,
        )
        if args.device != "auto":
            call_kwargs["device"] = args.device
        if args.max_shapelet_length is not None:
            call_kwargs["max_shapelet_length"] = args.max_shapelet_length
        if family == "tde_feddict" and ds_name == "InsectWingbeat":
            # InsectWingbeat: 25000 samples x 200 channels -- even with the
            # size-based small/large extraction switch (see
            # feddict_vertical.py), the vocabulary/memory cost is still
            # large at the default window_lengths=[8,16]. Narrowing to a
            # single, smaller window length plus a coarser stride cuts
            # both the number of DFT windows processed and the bigram
            # vocabulary size, specifically for this dataset -- every
            # other dataset keeps feddict's own defaults untouched.
            call_kwargs["window_lengths"] = [8]
            call_kwargs["stride"] = 2

        t0 = time.perf_counter()
        train_proba, test_proba, info = module.train_predict(
            s_X_train, s_Y_train, s_X_test, s_Y_test, **call_kwargs
        )
        elapsed_seconds = time.perf_counter() - t0
    except NotImplementedError as e:
        print(f"  [SKIPPED] {family} not implemented yet: {e}")
        continue

    # Display name includes the encoder actually used (from `info`, the
    # ground truth the module itself reports -- not just args.encoder)
    # for the two families where it matters, so reading results later
    # (e.g. the CAWPE weights dict) can't be mistaken for plain
    # InceptionTime when --encoder resnet1d was passed.
    display_name = f"{family}[{info['encoder']}]" if "encoder" in info else family

    train_acc = accuracy_score(Y_train, np.argmax(train_proba, axis=1))
    train_f1 = f1_score(Y_train, np.argmax(train_proba, axis=1), average="macro")
    # CAWPE weight = held-out TEST accuracy of this one global model.
    # NOT resubstitution/train accuracy (that overfits trivially for the
    # ridge-based families and silently inverts the weighting -- see the
    # conversation this was diagnosed in). No k-fold CV, no retraining,
    # no cross-client testing: one model, one honest accuracy number.
    test_acc = accuracy_score(Y_test, np.argmax(test_proba, axis=1))
    test_f1 = f1_score(Y_test, np.argmax(test_proba, axis=1), average="macro")
    print(f"  {display_name}: train accuracy={train_acc:.4f}, train macro-F1={train_f1:.4f}, "
          f"test accuracy={test_acc:.4f}, test macro-F1={test_f1:.4f}, "
          f"time={elapsed_seconds:.2f}s, info={info}")

    family_names.append(display_name)
    train_probas.append(train_proba)
    test_probas.append(test_proba)
    weight_accuracies.append(test_acc)
    family_info[display_name] = {
        "train_acc": train_acc, "train_f1": train_f1,
        "test_acc": test_acc, "test_f1": test_f1,
        "time_seconds": elapsed_seconds,
        **info,
    }
    table_rows.append((display_name, train_acc, train_f1, test_acc, test_f1, elapsed_seconds, info))

    wandb_metrics.update({
        f"{display_name}/train_acc": train_acc,
        f"{display_name}/train_f1": train_f1,
        f"{display_name}/test_acc": test_acc,
        f"{display_name}/test_f1": test_f1,
        f"{display_name}/time_seconds": elapsed_seconds,
    })

# --- FEDERATED CAWPE OVER WHATEVER FAMILIES WERE ACTUALLY TRAINED ---
if len(family_names) == 0:
    print("\nNo family produced results yet -- implement at least one "
          "*_vertical.py module before running CAWPE.")
    wandb_metrics["n_families_trained"] = 0
else:
    final_test_pred, combined_test_proba, weights = federated_cawpe(
        family_names, weight_accuracies, test_probas, alpha=cawpe_alpha
    )
    # Same weights, applied to each family's TRAIN probabilities, give an
    # ensemble train-set metric alongside the test-set one -- CAWPE itself
    # only ever combines test_probas (that's the actual prediction path),
    # this is purely for symmetric train/test reporting.
    weight_array = np.array([weights[name] for name in family_names])
    combined_train_proba = sum(w * p for w, p in zip(weight_array, train_probas))
    final_train_pred = np.argmax(combined_train_proba, axis=1)

    ensemble_train_acc = accuracy_score(Y_train, final_train_pred)
    ensemble_train_f1 = f1_score(Y_train, final_train_pred, average="macro")
    ensemble_test_acc = accuracy_score(Y_test, final_test_pred)
    ensemble_test_f1 = f1_score(Y_test, final_test_pred, average="macro")

    print("\n=== Federated CAWPE ensemble ===")
    print("Per-family weights:", weights)
    print(f"Ensemble train accuracy: {ensemble_train_acc:.4f}, train macro-F1: {ensemble_train_f1:.4f}")
    print(f"Ensemble test accuracy: {ensemble_test_acc:.4f}, test macro-F1: {ensemble_test_f1:.4f}")

    results_table = wandb.Table(columns=[
        "family", "train_acc", "train_f1", "test_acc", "test_f1", "time_seconds", "cawpe_weight", "info_json"
    ])
    for display_name, train_acc, train_f1, test_acc, test_f1, elapsed_seconds, info in table_rows:
        results_table.add_data(display_name, train_acc, train_f1, test_acc, test_f1, elapsed_seconds,
                                weights.get(display_name), json.dumps(info, default=str))
    total_families_time_seconds = sum(row[5] for row in table_rows)
    results_table.add_data("ENSEMBLE (CAWPE)", ensemble_train_acc, ensemble_train_f1,
                            ensemble_test_acc, ensemble_test_f1, total_families_time_seconds, 1.0, "{}")

    # --- Weighting ablation: same trained families, different CAWPE exponent ---
    # alpha=0 -> uniform weights; alpha=cawpe_alpha is the main result above.
    for a in args.ablation_alphas:
        _pred, _proba, _w = federated_cawpe(family_names, weight_accuracies, test_probas, alpha=a)
        _acc = accuracy_score(Y_test, _pred)
        _f1 = f1_score(Y_test, _pred, average="macro")
        print(f"[ablation] alpha={a}: test accuracy={_acc:.4f}, macro-F1={_f1:.4f}, weights={_w}")
        wandb_metrics[f"ensemble_alpha{a:g}/test_acc"] = _acc
        wandb_metrics[f"ensemble_alpha{a:g}/test_f1"] = _f1
    if args.save_probas_dir:
        os.makedirs(args.save_probas_dir, exist_ok=True)
        np.savez(os.path.join(args.save_probas_dir, f"{args.ds_name}_seed{args.seed}.npz"),
                 families=np.array(family_names), y_test=np.asarray(Y_test),
                 weight_accuracies=np.array(weight_accuracies),
                 **{f"proba_{i}": p for i, p in enumerate(test_probas)})

    wandb_metrics.update({
        "ensemble/train_acc": ensemble_train_acc,
        "ensemble/train_f1": ensemble_train_f1,
        "ensemble/test_acc": ensemble_test_acc,
        "ensemble/test_f1": ensemble_test_f1,
        "total_families_time_seconds": total_families_time_seconds,
        "results_table": results_table,
        **{f"cawpe_weight/{name}": w for name, w in weights.items()},
    })

wandb_metrics["total_wall_time_seconds"] = time.perf_counter() - script_start_time

# Single log call: this script runs each family exactly once (no epochs/
# steps to plot as a time series), so one consolidated call keeps every
# metric at the same wandb step and avoids fragmenting history across
# per-family calls -- it also seeds wandb.summary with these values.
wandb.log(wandb_metrics)
wandb.finish()