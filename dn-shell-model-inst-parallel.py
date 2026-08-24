"""
Run instanton computations sequentially for multiple target observables.
Run this script from PyCharm — it uses the same Python interpreter and environment.

Usage: just run this file. Edit target_obs_list below to change the values.
"""

import subprocess
import sys
import os
import time

# ---- CONFIGURE HERE ----
target_obs_list = [
    0, 1, 2, 3, 4, 5, 6, 7, 8, 9,
    -1, -2, -3, -4, -5, -6, -7, -8, -9, -10,
]
script_name = "dn-shell-model-inst.py"  # your main instanton script
stop_on_failure = False   # set True to abort the whole sweep on the first failure
stream_output = False     # True = print child stdout live; False = only summary line
# -------------------------


def run_single(target_obs):
    """Run the instanton script for a single target observable."""
    env = os.environ.copy()
    # Keep each child single-threaded so libraries don't oversubscribe the CPU.
    env["MKL_NUM_THREADS"] = "1"
    env["NUMEXPR_NUM_THREADS"] = "1"
    env["OMP_NUM_THREADS"] = "1"
    env["VECLIB_MAXIMUM_THREADS"] = "1"
    env["OPENBLAS_NUM_THREADS"] = "1"

    script_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), script_name
    )

    if stream_output:
        # Inherit stdout/stderr so you see everything live in PyCharm.
        result = subprocess.run(
            [sys.executable, script_path, str(target_obs)],
            env=env,
        )
        tail = ""
    else:
        result = subprocess.run(
            [sys.executable, script_path, str(target_obs)],
            env=env,
            capture_output=True,
            text=True,
        )
        tail = (result.stdout.strip().split("\n")[-1]
                if result.stdout else "")

    if result.returncode != 0:
        print(f"[FAILED] targetObs={target_obs}")
        if not stream_output and result.stderr:
            print(result.stderr[-500:])
    else:
        print(f"[DONE] targetObs={target_obs}"
              + (f"  |  {tail}" if tail else ""))

    return target_obs, result.returncode


if __name__ == "__main__":
    print(f"Running {len(target_obs_list)} instanton computations sequentially")
    print(f"Using Python: {sys.executable}")
    print()

    t0 = time.time()
    results = []
    for i, obs in enumerate(target_obs_list, 1):
        print(f"--- [{i}/{len(target_obs_list)}] targetObs={obs} ---")
        t_start = time.time()
        obs_val, rc = run_single(obs)
        print(f"    elapsed: {time.time() - t_start:.1f}s")
        results.append((obs_val, rc))

        if rc != 0 and stop_on_failure:
            print("Aborting sweep (stop_on_failure=True).")
            break

    failed = [obs for obs, rc in results if rc != 0]
    total = time.time() - t0
    print()
    if failed:
        print(f"Failed runs: {failed}")
        print(f"Completed {len(results) - len(failed)}/{len(results)} "
              f"in {total:.1f}s")
    else:
        print(f"All {len(results)} runs completed successfully in {total:.1f}s.")