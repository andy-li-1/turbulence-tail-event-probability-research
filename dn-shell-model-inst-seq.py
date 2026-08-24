"""
Sequential instanton sweep with continuation (warm-starting).
Sweeps outward from z=0 in both directions, using each converged
solution as the initial guess for the next target observable.
"""

import sys
import os
import numpy as np
import time
import matplotlib.pyplot as plt
import jax
import jax.numpy as jnp

branch = "502"

# Add the script directory to path so we can import the instanton module
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, script_dir)

from importlib import import_module
# Import your instanton script as a module
inst_mod = import_module("dn-shell-model-inst-v2")

param = inst_mod.param

def run_sweep(z_values, label="", seed_eta=None, seed_lbda=None):
    if seed_eta is not None:
        prev_eta = seed_eta
        prev_lbda = seed_lbda
    else:
        prev_eta = np.random.randn(inst_mod.nt + 1, inst_mod.dim) / jnp.sqrt(inst_mod.dt)
        prev_lbda = 1.0
    results = {}
    inst_mod.plots=True
    t_np = np.array(inst_mod.t)
    k_np = np.array(inst_mod.k)

    for i, z in enumerate(z_values):
        print(f"\n[{label}] {i+1}/{len(z_values)}: z = {z}")
        start = time.time()

        from datetime import datetime
        now = datetime.now()
        dt_string = now.strftime('%Y_%m_%d_%H_%M_%S')
        data_dir = '/Users/rawdata/Downloads/data/{}/seq_nu_{}_c1_{}_c2_{}_obs_{}_date_{}_branch_{}'.format(
            param, inst_mod.nu, inst_mod.c1, inst_mod.c2, round(z, 2), dt_string, branch)
        if not os.path.isdir(data_dir):
            os.makedirs(data_dir)

        instanton = inst_mod.Instanton()
        obsValue, action, lbda, eta, u, ener_diss, dS = \
            instanton.searchInstantonViaAugmented(
                z, initialEta=prev_eta, initLbda=prev_lbda
            )

        u_snapshots = inst_mod.get_snapshots_per_second(u)

        # Compute the instanton observable F(u(t)) = sum_n k_n * u_n(t) at every time step,
        # then subsample to n_save points for storage and plotting.
        F_traj_full = np.array(u) @ k_np  # shape (nt+1,)

        n_save = 500
        save_indices = np.linspace(0, inst_mod.nt, n_save).astype(int)
        F_traj = F_traj_full[save_indices]
        t_save = t_np[save_indices]

        # Same saves as your original script
        np.save(data_dir + '/obs.npy', z)
        np.save(data_dir + '/inst_obs.npy', obsValue)
        np.save(data_dir + '/inst_act.npy', action)
        np.save(data_dir + '/inst_lbda.npy', lbda)
        np.save(data_dir + '/inst_eta.npy', eta)
        np.save(data_dir + '/inst_u.npy', u)
        np.save(data_dir + '/ener_diss.npy', ener_diss)
        np.save(data_dir + '/inst_ds.npy', dS)
        np.save(data_dir + '/inst_u_per_second.npy', u_snapshots)
        np.save(data_dir + '/inst_velo_grad.npy', F_traj)
        np.save(data_dir + '/inst_velo_grad_t.npy', t_save)

        elapsed = time.time() - start
        print(f"  -> obs={obsValue:.4f}, action={action:.4f}, time={elapsed:.1f}s")

        results[z] = {'obs': obsValue, 'action': action}
        prev_eta = np.array(eta)
        prev_lbda = float(lbda)


        plt.figure()
        for n in range(inst_mod.dim):
            plt.plot(t_np, np.array(u[:, n]), label=f'shell {n}')
        plt.xlabel(r'$t$')
        plt.ylabel(r'$u_n(t)$')
        plt.legend(fontsize=6, ncol=3)
        plt.savefig(data_dir + '/inst_u.pdf', bbox_inches='tight')
        plt.close()

        plt.figure()
        plt.plot(t_np, np.array(ener_diss))
        plt.xlabel(r'$t$')
        plt.ylabel(r'$\epsilon = \nu \sum k_n^2 u_n^2$')
        plt.savefig(data_dir + '/inst_ener_diss.pdf', bbox_inches='tight')
        # plt.show() # ================= SHOWING EDR PLOT =======================
        plt.close()

        plt.figure()
        plt.plot(np.arange(inst_mod.dim), np.array(inst_mod.jgetEnergy(u[-1])))
        plt.xlabel(r'shell $n$')
        plt.ylabel(r'$E_n = \frac{1}{2} u_n^2$')
        plt.savefig(data_dir + '/inst_energy.pdf', bbox_inches='tight')
        plt.close()

        plt.figure()
        shells = np.arange(inst_mod.dim)
        plot_times = list(np.linspace(0, inst_mod.T, 11).astype(int))

        for s in plot_times:
            if s == 0:
                u_at_t = np.array(inst_mod.init_u)
            else:
                u_at_t = u_snapshots[:, s - 1]  # u_snapshots is 0-indexed from t=1
            u_sq = np.array(u_at_t ** 2)
            u_sq = np.where(u_sq > 0, u_sq, np.nan)
            plt.plot(shells, u_sq, label=f't = {s}', marker='o')

        plt.xlabel(r'shell $n$')
        plt.ylabel(r'$\log(u_n^2)$')
        plt.yscale('log')
        plt.legend(fontsize=8, ncol=2)
        plt.savefig(data_dir + '/inst_log_energy_spectrum.pdf', bbox_inches='tight')
        plt.close()

        # Instanton observable trajectory: F(u(t)) = sum_n k_n * u_n(t)
        plt.figure()
        plt.plot(t_save, F_traj, color='black')
        plt.axhline(z, color='red', linestyle='--', linewidth=1,
                    label=f'target $z = {z}$')
        plt.xlabel(r'$t$')
        plt.ylabel(r'$F(u(t)) = \sum_n k_n u_n(t)$')
        plt.title('Instanton observable trajectory')
        plt.legend()
        plt.savefig(data_dir + '/inst_velo_grad.pdf', bbox_inches='tight')
        plt.close()

    return results


if __name__ == '__main__':
    z_start = 0.0  # your chosen starting observable

    # Solve once at the starting point
    instanton = inst_mod.Instanton()
    init_eta = np.random.randn(inst_mod.nt + 1, inst_mod.dim) / jnp.sqrt(inst_mod.dt)
    init_lbda = 1.0
    print("=== Sweeping initial z ===")
    obsValue, action, lbda, eta, u, ener_diss, dS = \
        instanton.searchInstantonViaAugmented(z_start, initialEta=init_eta, initLbda=init_lbda)

    # Use this converged solution as the seed for both directions
    seed_eta = np.array(eta)
    seed_lbda = float(lbda)

    # Positive sweep starts AFTER z_start going up
    z_pos = np.arange(z_start+0.25, 1.0, 0.25)
    # Negative sweep starts FROM z_start going down
    # z_neg = np.arange(z_start, -5.0, -0.25)

    print("=== Sweeping positive z ===")
    results_pos = run_sweep(z_pos, label="pos", seed_eta=seed_eta, seed_lbda=seed_lbda)

    print("\n=== Sweeping negative z ===")
    # results_neg = run_sweep(z_neg, label="neg", seed_eta=seed_eta, seed_lbda=seed_lbda)