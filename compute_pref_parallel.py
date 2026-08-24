"""
Run prefactor computations in parallel across multiple target observables.

Discovery:
  Scans the saved-instanton directory for all `obs_*` values matching the
  configured (nu, c1, c2, branch) parameters and builds the work list.

Execution:
  Dispatches one prefactor computation per target observable to a pool of
  worker processes. Each worker runs single-threaded internally (BLAS,
  OpenMP, etc. all pinned to 1 thread) so the N workers don't oversubscribe
  the CPU.

  Each successful result is saved to its own scalar .npy file as soon as
  the worker finishes, into a dedicated output directory created at startup.

Run this from PyCharm — same interpreter, same environment.
"""

import os
import sys
import time
import glob
import re
import numpy as np

# ---- Pin BLAS / OpenMP threads BEFORE importing numpy / jax / scipy ----
for _var in ("MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS", "OMP_NUM_THREADS",
             "VECLIB_MAXIMUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

from concurrent.futures import ProcessPoolExecutor, as_completed
from importlib import import_module
# Import your instanton script as a module
pref_mod = import_module("compute_prefactor")

# ---- CONFIGURE HERE ----
N_WORKERS = 1
CHUNK_SIZE = 1
PREFACTOR_MODULE = "compute_prefactor"
STOP_ON_FAILURE = False
# These must match the values in the prefactor module so directory globbing matches.
NU = pref_mod.nu
C1 = pref_mod.c1
C2 = pref_mod.c2
BRANCH = pref_mod.branch

PARAMETER = pref_mod.PARAMETER
# -------------------------

# Instanton input directory (where solved-instanton data lives)
DIR_INST_TEMPLATE = (
    f"/Users/rawdata/Downloads/data/{PARAMETER}/"
    "seq_nu_{}_c1_{}_c2_{}_obs_*_date_*_*_*_*_*_*_branch_{}/"
)

INPUT_DIR_TEMPLATE = (f"/Users/rawdata/Downloads/data/{PARAMETER}/"
                      "seq_nu_{}_c1_{}_c2_{}_obs_*_date_*_*_*_*_*_*_branch_{}/")

# Output directory (where prefactor .npy files go).
OUTPUT_DIR_TEMPLATE = pref_mod.OUTPUT_DIR_TEMPLATE


################################################################
# Discovery: find all available target observables on disk

def discover_target_observables(nu=NU, c1=C1, c2=C2, branch=BRANCH):
    """
    Scan the instanton data directory and return a sorted list of unique
    target-observable values for which saved data exists.
    """
    pattern = INPUT_DIR_TEMPLATE.format(nu, c1, c2, branch)
    matches = glob.glob(pattern)

    obs_values = set()
    for path in matches:
        m = re.search(r'obs_([\d.eE+\-]+)', path)
        if m:
            try:
                obs_values.add(float(m.group(1)))
            except ValueError:
                continue

    return sorted(obs_values)


################################################################
# Per-result saving

def save_single_result(output_dir, target_obs, prefactor):
    """Save a single scalar prefactor to its own .npy file."""
    np.save(output_dir + f"/prefactor_obs_{target_obs}.npy", np.asarray(prefactor))
    return output_dir + f"/prefactor_obs_{target_obs}.npy"


################################################################
# Worker entry point — runs in each subprocess

def _worker_compute(target_obs):
    """
    Worker function invoked in a subprocess. Imports the prefactor module
    *inside* the worker so JAX initializes cleanly per process.
    """
    here = os.path.dirname(os.path.abspath(__file__))
    if here not in sys.path:
        sys.path.insert(0, here)

    t_start = time.time()
    try:
        prefactor = pref_mod.compute_prefactor(target_obs)
        elapsed = time.time() - t_start
        return target_obs, prefactor, None, elapsed
    except Exception as e:
        elapsed = time.time() - t_start
        import traceback
        return target_obs, None, traceback.format_exc(), elapsed


################################################################
# Chunked dispatcher

def chunks(seq, size):
    for i in range(0, len(seq), size):
        yield seq[i:i + size]


def run_parallel(target_obs_list, output_dir, n_workers=N_WORKERS, chunk_size=CHUNK_SIZE):
    """
    Run compute_prefactor for each target in target_obs_list in parallel.

    Each successful result is saved to its own .npy file in `output_dir`
    immediately after the worker returns it.
    """
    results = {}
    failures = []

    total = len(target_obs_list)
    completed = 0
    t0 = time.time()

    for batch_idx, batch in enumerate(chunks(target_obs_list, chunk_size), 1):
        print(f"\n--- Batch {batch_idx}: {len(batch)} tasks "
              f"(targets {batch[0]} ... {batch[-1]}) ---")

        with ProcessPoolExecutor(max_workers=n_workers) as ex:
            future_to_obs = {ex.submit(_worker_compute, z): z for z in batch}

            for fut in as_completed(future_to_obs):
                z = future_to_obs[fut]
                completed += 1
                try:
                    target_obs, prefactor, err, elapsed = fut.result()
                except Exception as e:
                    print(f"  [{completed}/{total}] z={z}: WORKER CRASHED: {e}")
                    failures.append((z, str(e)))
                    if STOP_ON_FAILURE:
                        print("Aborting (STOP_ON_FAILURE=True).")
                        return results, failures
                    continue

                if err is not None:
                    print(f"  [{completed}/{total}] z={target_obs}: FAILED "
                          f"({elapsed:.1f}s)")
                    print("    " + err.strip().splitlines()[-1])
                    failures.append((target_obs, err))
                    if STOP_ON_FAILURE:
                        print("Aborting (STOP_ON_FAILURE=True).")
                        return results, failures
                else:
                    results[target_obs] = prefactor
                    try:
                        saved_path = save_single_result(output_dir, target_obs, prefactor)
                        save_msg = f"  saved -> {os.path.basename(saved_path)}"
                    except Exception as save_err:
                        save_msg = f"  WARNING: save failed: {save_err}"
                    print(f"  [{completed}/{total}] z={target_obs}: "
                          f"prefactor={prefactor:.6e}  ({elapsed:.1f}s)")
                    print(save_msg)

    print(f"\nTotal wall time: {time.time() - t0:.1f}s")
    return results, failures


################################################################
# Main

if __name__ == "__main__":

    here = os.path.dirname(os.path.abspath(__file__))
    if here not in sys.path:
        sys.path.insert(0, here)

    print(f"Discovering target observables for "
          f"nu={NU}, c1={C1}, c2={C2}, branch={BRANCH}...")
    targets = discover_target_observables()
    # targets = [z for z in targets if -10.0 <= z <= 25.0] # ===== FOR EXTRA FILTERING =====

    if not targets:
        print("No saved instanton data found. Check parameters and path.")
        sys.exit(1)

    output_dir = OUTPUT_DIR_TEMPLATE.format(NU, C1, C2, BRANCH)
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output directory: {output_dir}")

    print(f"Found {len(targets)} target observables:")
    print(f"  {targets}")
    print(f"\nUsing {N_WORKERS} workers, chunk size {CHUNK_SIZE}")
    print(f"Python interpreter: {sys.executable}")

    results, failures = run_parallel(targets, output_dir)

    print("\n" + "=" * 60)
    print(f"Successful: {len(results)} / {len(targets)}")
    if failures:
        print(f"Failed:     {len(failures)}")
        for z, _ in failures:
            print(f"  - z={z}")

    if results:
        print("\nPrefactors:")
        for z in sorted(results):
            print(f"  z={z:>8.3f}  ->  c(z) = {results[z]:.6e}")

    print(f"\nIndividual scalar .npy files written to: {output_dir}")