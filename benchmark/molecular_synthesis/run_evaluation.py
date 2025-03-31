import os
import sys
import pickle
import argparse
import numpy as np
from tqdm import tqdm
from pathlib import Path
from arsenal.maths import logsumexp
from functools import lru_cache
from concurrent.futures import ProcessPoolExecutor, as_completed

from genlm_control.constant import EndOfSequence

project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from rdkit.Chem import QED
from rdkit import Chem
from rdkit import rdBase
import os, sys


class HiddenPrints:
    """Context manager to disable RDKit logs. By default all logs are disabled."""

    def __init__(
        self,
        mute_errors: bool = True,
        mute_warning: bool = True,
        mute_info: bool = True,
        mute_debug: bool = True,
    ):
        # Get current log state
        self.previous_status = self._get_log_status()

        # Init the desired log state to apply during in the context
        self.desired_status = {}
        self.desired_status["rdApp.error"] = not mute_errors
        self.desired_status["rdApp.warning"] = not mute_warning
        self.desired_status["rdApp.debug"] = not mute_debug
        self.desired_status["rdApp.info"] = not mute_info

    def _get_log_status(self):
        """Get the current log status of RDKit logs."""
        log_status = rdBase.LogStatus()
        log_status = {
            st.split(":")[0]: st.split(":")[1] for st in log_status.split("\n")
        }
        log_status = {
            k: True if v == "enabled" else False for k, v in log_status.items()
        }
        return log_status

    def _apply_log_status(self, log_status):
        """Apply an RDKit log status."""
        for k, v in log_status.items():
            if v is True:
                rdBase.EnableLog(k)
            else:
                rdBase.DisableLog(k)

    def __enter__(self):
        self._apply_log_status(self.desired_status)

    def __exit__(self, *args, **kwargs):
        self._apply_log_status(self.previous_status)


# class HiddenPrints:
#     def __enter__(self):
#         self._original_stdout = sys.stdout
#         sys.stdout = open(os.devnull, 'w')

#     def __exit__(self, exc_type, exc_val, exc_tb):
#         sys.stdout.close()
#         sys.stdout = self._original_stdout


def get_mol(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is not None:
        Chem.Kekulize(mol)
    return mol


@lru_cache
def cached_eval(mol):
    mol = get_mol(mol)
    if mol is None:
        return 0.0, 0.0
    return 1.0, QED.qed(mol=mol)


def posterior_weighted_eval(contexts, log_weights):
    weighted_acc = 0
    particle_results = []

    finished = [isinstance(context[-1], EndOfSequence) for context in contexts]
    log_weights = np.array([l if f else -np.inf for l, f in zip(log_weights, finished)])
    log_total = logsumexp(log_weights)
    log_normalized_weights = log_weights - log_total
    probs = np.exp(log_normalized_weights)

    assert len(contexts) == len(probs)

    valid_count = 0
    for context, p in zip(contexts, probs):
        ctx = context[:-1] if isinstance(context[-1], EndOfSequence) else context

        try:
            pred = b"".join(ctx).decode("utf-8")
        except UnicodeDecodeError:
            w_acc = 0
            particle_results.append((context, w_acc))
            continue

        if np.isnan(p):
            w_acc = 0
        else:
            mol = pred.strip()
            valid, acc = cached_eval(mol)
            valid_count += valid
            w_acc = p * acc

        weighted_acc += w_acc
        particle_results.append((context, w_acc))

    return {
        "result": weighted_acc,
        "particle_results": particle_results,
        "valid_count": valid_count,
    }


def eval_wrapper(datum):
    return posterior_weighted_eval(
        datum["contexts"],
        datum["log_weights"],
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--results_dir",
        type=str,
        required=True,
        help='Path to the directory containing the results. Each file in this directory should be named "i_result.pkl", where i is an integer representing the index of the result.',
    )
    parser.add_argument(
        "--n_workers",
        type=int,
        default=2,
        help="Number of workers to use for evaluation",
    )
    parser.add_argument(
        "--output_pkl",
        type=str,
        required=True,
        help="Path to save the evaluation results in pickled format.",
    )
    parser.add_argument(
        "--timeout", type=int, default=None, help="Timeout for evaluation"
    )
    args = parser.parse_args()

    data = []
    for file in os.listdir(args.results_dir):
        if file.endswith("_result.pkl"):
            with open(os.path.join(args.results_dir, file), "rb") as f:
                data.append(pickle.load(f))

    if args.n_workers > 1:
        with ProcessPoolExecutor(
            max_workers=args.n_workers,
        ) as executor:
            with tqdm(total=len(data)) as progress_bar:
                with HiddenPrints():
                    futures = [executor.submit(eval_wrapper, datum) for datum in data]
                    results = []
                    for future in as_completed(futures):
                        results.append(future.result())
                        progress_bar.update(1)
    else:
        with HiddenPrints():
            results = [eval_wrapper(datum) for datum in tqdm(data)]

    print(args.results_dir.split("/")[-1])
    mean, lower, upper = mean_ci([r["result"] for r in results])
    print(f"Mean accuracy: {round(mean, 4)} ({round(lower, 2)}, {round(upper, 2)})")
    mean, lower, upper = mean_ci(
        [datum["metadata"]["inference_time"] for datum in data]
    )
    print(
        f"Mean inference time: {round(mean, 4)} ({round(lower, 2)}, {round(upper, 2)})"
    )

    with open(args.output_pkl, "wb") as f:
        pickle.dump(results, f)


def mean_ci(values, ci=0.95, n_bootstrap=10000):
    mean = np.mean(values)
    bootstrap_means = []

    for _ in range(n_bootstrap):
        bootstrap_sample = np.random.choice(values, size=len(values), replace=True)
        bootstrap_means.append(np.mean(bootstrap_sample))

    lower_percentile = (1 - ci) / 2 * 100
    upper_percentile = (1 + ci) / 2 * 100
    lower, upper = np.percentile(bootstrap_means, [lower_percentile, upper_percentile])
    return mean, lower, upper


if __name__ == "__main__":
    main()
