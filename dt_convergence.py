"""
Systematic dt-convergence study at fixed (T, z, c1, nu).

Per the paper (Schorlepp et al, Fig 8): hold T fixed, vary dt over a sequence
of refinements, and plot eigenvalue spectra and eigenfunctions on the same
axes to check that they converge.

Assumes instanton data has been pre-computed at each dt value, and that
compute_prefactor.py has been run for each dt to generate the eigenvalue
arrays.
"""

import glob
import os
import numpy as np
import matplotlib.pyplot as plt

# Configuration: fix everything except dt
NU = 0.01
C1 = 0.01      # adjust per your study
C2 = 0.0
T = 80         # fixed
Z = 10.0       # the z value to study (try 10, 29 per the professor's suggestion)
BRANCH = 4
PARAMETER = "inst"

DT_VALUES = [4e-3, 2e-3, 1e-3, 5e-4, 2.5e-4]  # halving each time

OUT_DIR = f"/Users/rawdata/Downloads/data/{PARAMETER}/dt_convergence_z_{Z}_c1_{C1}"
os.makedirs(OUT_DIR, exist_ok=True)


def _load_evals_for_dt(dt):
    save_dir = (f"/Users/rawdata/Downloads/data/{PARAMETER}/"
                f"prefactors_nu_{NU}_c1_{C1}_c2_{C2}_branch_{BRANCH}_dt_{dt}")
    path = f"{save_dir}/evals_A_project_perp_True_z_{Z}.npy"
    if not os.path.exists(path):
        print(f"  [warn] missing evals at dt={dt}: {path}")
        return None
    return np.load(path)


def _load_eigfunc_for_dt(dt, j):
    save_dir = (f"/Users/rawdata/Downloads/data/{PARAMETER}/"
                f"prefactors_nu_{NU}_c1_{C1}_c2_{C2}_branch_{BRANCH}_dt_{dt}")
    path = f"{save_dir}/eigfunc_j_{j}_z_{Z}.npy"
    if not os.path.exists(path):
        print(f"  [warn] missing eigfunc j={j} at dt={dt}: {path}")
        return None
    return np.load(path)


def _load_instanton_traj_for_dt(dt):
    """Load the instanton's running observable trajectory."""
    inst_dir_pattern = (f"/Users/rawdata/Downloads/data/{PARAMETER}/"
                        f"seq_nu_{NU}_c1_{C1}_c2_{C2}_obs_{Z}_date_*_*_*_*_*_*_branch_{BRANCH}_dt_{dt}/")
    matches = glob.glob(inst_dir_pattern)
    if not matches:
        return None, None
    matches.sort(key=os.path.getmtime, reverse=True)
    F_traj = np.load(os.path.join(matches[0], "inst_F_traj.npy"))
    t_traj = np.load(os.path.join(matches[0], "inst_F_traj_t.npy"))
    return t_traj, F_traj


def _load_action_for_dt(dt):
    inst_dir_pattern = (f"/Users/rawdata/Downloads/data/{PARAMETER}/"
                        f"seq_nu_{NU}_c1_{C1}_c2_{C2}_obs_{Z}_date_*_*_*_*_*_*_branch_{BRANCH}_dt_{dt}/")
    matches = glob.glob(inst_dir_pattern)
    if not matches:
        return None
    matches.sort(key=os.path.getmtime, reverse=True)
    return float(np.load(os.path.join(matches[0], "inst_act.npy")))


def plot_eigenvalue_spectra():
    """Figure 8 reproduction: eigenvalue spectra at multiple dt overlaid."""
    fig, ax = plt.subplots(figsize=(10, 6))
    cmap = plt.cm.viridis
    colors = [cmap(i / max(1, len(DT_VALUES) - 1)) for i in range(len(DT_VALUES))]

    for dt, color in zip(DT_VALUES, colors):
        evals = _load_evals_for_dt(dt)
        if evals is None:
            continue
        nt_dt = int(T / dt)
        idx = np.arange(1, len(evals) + 1)
        ax.plot(idx, np.abs(evals), 'o', markersize=4, color=color, alpha=0.7,
                label=f'dt={dt} (n_t={nt_dt})')

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('Index $i$')
    ax.set_ylabel(r'$|\lambda^{(i)}_z|$')
    ax.set_title(f'Eigenvalue spectrum convergence at z={Z}, T={T}, c1={C1}')
    ax.legend()

    # Inset: running determinant convergence in dt
    ax_inset = fig.add_axes([0.55, 0.55, 0.3, 0.3])
    dt_arr, det_arr = [], []
    for dt in DT_VALUES:
        evals = _load_evals_for_dt(dt)
        if evals is None:
            continue
        # Use first 80 eigenvalues for the running det, like the paper
        n_keep = min(80, len(evals))
        det = np.prod(1.0 - evals[:n_keep])
        dt_arr.append(dt)
        det_arr.append(np.abs(det))
    if dt_arr:
        ax_inset.semilogx(dt_arr, det_arr, 'o-')
        ax_inset.invert_xaxis()  # so finer dt is on the right
        ax_inset.set_xlabel(r'$dt$', fontsize=8)
        ax_inset.set_ylabel(r'$|\prod_{i=1}^{80}(1-\mu_i)|$', fontsize=8)
        ax_inset.tick_params(labelsize=6)

    plt.savefig(f"{OUT_DIR}/eigenvalue_spectra_convergence.pdf", bbox_inches='tight')
    plt.close(fig)
    print(f"  saved {OUT_DIR}/eigenvalue_spectra_convergence.pdf")


def plot_eigenfunction_convergence(j):
    """Compare eigenfunction j across dt values, both full-domain and zoom."""
    cmap = plt.cm.viridis
    colors = [cmap(i / max(1, len(DT_VALUES) - 1)) for i in range(len(DT_VALUES))]

    # Pick a few representative shells to plot — too many will clutter
    shells_to_plot = [0, 4, 8, 12, 16]

    for view in ['full', 'zoom']:
        fig, axes = plt.subplots(len(shells_to_plot), 1,
                                 figsize=(10, 2 * len(shells_to_plot)),
                                 sharex=True)
        if len(shells_to_plot) == 1:
            axes = [axes]

        for ax, n_shell in zip(axes, shells_to_plot):
            for dt, color in zip(DT_VALUES, colors):
                eigfunc = _load_eigfunc_for_dt(dt, j)
                if eigfunc is None:
                    continue
                nt_dt = int(T / dt)
                t_np = np.linspace(0, T, nt_dt + 1)

                if view == 'zoom':
                    mask = t_np >= 0.95 * T
                    ax.plot(t_np[mask], eigfunc[mask, n_shell],
                            color=color, alpha=0.7, label=f'dt={dt}')
                else:
                    ax.plot(t_np, eigfunc[:, n_shell],
                            color=color, alpha=0.7, label=f'dt={dt}')

            ax.set_ylabel(f'shell {n_shell}', fontsize=9)
            ax.grid(alpha=0.3)

        axes[0].legend(fontsize=8, ncol=len(DT_VALUES))
        axes[0].set_title(f'Eigenfunction j={j} convergence at z={Z} ({view})')
        axes[-1].set_xlabel(r'$t$')
        plt.tight_layout()
        plt.savefig(f"{OUT_DIR}/eigfunc_j_{j}_convergence_{view}.pdf",
                    bbox_inches='tight')
        plt.close(fig)
        print(f"  saved {OUT_DIR}/eigfunc_j_{j}_convergence_{view}.pdf")


def plot_instanton_trajectory_convergence():
    """Compare F(u(t)) across dt values."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    cmap = plt.cm.viridis
    colors = [cmap(i / max(1, len(DT_VALUES) - 1)) for i in range(len(DT_VALUES))]

    actions = []

    for dt, color in zip(DT_VALUES, colors):
        t_traj, F_traj = _load_instanton_traj_for_dt(dt)
        action = _load_action_for_dt(dt)
        if t_traj is None:
            continue

        nt_dt = int(T / dt)
        # Full
        axes[0].plot(t_traj, F_traj, color=color, alpha=0.8,
                     label=f'dt={dt} (n_t={nt_dt}, I={action:.4f})')
        # Zoom
        mask = t_traj >= 0.95 * T
        axes[1].plot(t_traj[mask], F_traj[mask], color=color, alpha=0.8,
                     label=f'dt={dt}')

        if action is not None:
            actions.append((dt, action))

    for ax in axes:
        ax.axhline(Z, color='red', linestyle='--', linewidth=1,
                   label=f'target z={Z}')
        ax.set_xlabel(r'$t$')
        ax.set_ylabel(r'$F(u(t))$')
        ax.legend(fontsize=8)

    axes[0].set_title('Full domain')
    axes[1].set_title('Zoom: t >= 0.95 T')
    fig.suptitle(f'Instanton trajectory convergence at z={Z}, T={T}, c1={C1}')
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/instanton_trajectory_convergence.pdf",
                bbox_inches='tight')
    plt.close(fig)
    print(f"  saved {OUT_DIR}/instanton_trajectory_convergence.pdf")

    # Print action vs dt — most compact convergence indicator
    if actions:
        print(f"\n  Action I(z={Z}) vs dt:")
        print(f"  {'dt':>10}  {'I(z)':>14}  {'rel diff':>10}")
        I_finest = actions[-1][1]
        for dt, I in actions:
            rel = abs(I - I_finest) / abs(I_finest) if I_finest else float('nan')
            print(f"  {dt:>10}  {I:>14.6f}  {rel:>10.4e}")


if __name__ == "__main__":
    print(f"=== dt convergence study at z={Z}, T={T}, c1={C1} ===")
    plot_eigenvalue_spectra()
    for j in range(5):
        plot_eigenfunction_convergence(j)
    plot_instanton_trajectory_convergence()
    print("done.")