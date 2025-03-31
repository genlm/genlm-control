import os
import sys
import pickle
import argparse
import numpy as np
from tqdm import tqdm
from pathlib import Path
from arsenal.maths import logsumexp
import functools
import planetarium
from concurrent.futures import ProcessPoolExecutor, as_completed
from lark.exceptions import UnexpectedInput, VisitError

from genlm_control.constant import EndOfSequence

project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))


@functools.cache
def equiv(x, y):
    try:
        return planetarium.evaluate(x, y)[2]

    except (UnexpectedInput, ValueError, VisitError, AttributeError):
        # except:
        # you can get unexpected input for ungrammatical inputs
        # ValueError for referencing an object not in the object list
        # VisitError for duplicate object names
        return False


def posterior_weighted_eval(contexts, log_weights, masked_pddl, full_pddl):
    weighted_acc = 0
    particle_results = []

    finished = [isinstance(context[-1], EndOfSequence) for context in contexts]
    log_weights = np.array([l if f else -np.inf for l, f in zip(log_weights, finished)])
    log_total = logsumexp(log_weights)
    log_normalized_weights = log_weights - log_total
    probs = np.exp(log_normalized_weights)

    assert len(contexts) == len(probs)

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
            full_output = masked_pddl.replace("[BLANK]", pred + ")")
            acc = equiv(full_pddl, full_output)
            w_acc = p * acc

        weighted_acc += w_acc
        particle_results.append((context, w_acc))

    return {
        "result": weighted_acc,
        "particle_results": particle_results,
    }


def eval_wrapper(datum):
    return posterior_weighted_eval(
        datum["contexts"],
        datum["log_weights"],
        datum["metadata"]["masked_pddl"],
        datum["metadata"]["problem_pddl"],
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
                futures = [executor.submit(eval_wrapper, datum) for datum in data]
                results = []
                for future in as_completed(futures):
                    results.append(future.result())
                    progress_bar.update(1)
    else:
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
