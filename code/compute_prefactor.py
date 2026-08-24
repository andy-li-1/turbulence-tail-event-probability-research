"""
Compute the instanton prefactor for a single target observable, using
already-saved instanton data on disk.

Entry point: compute_prefactor(target_observable) -> float

Each call is self-contained — no shared mutable state — so this module is
safe to call from multiple threads concurrently (subject to the JAX caveats
noted at the bottom of the file).
"""

import copy
import glob
import os
import re
import matplotlib.pyplot as plt

import numpy as np
from scipy.sparse.linalg import LinearOperator, eigs

import jax
import jax.numpy as jnp

jax.config.update("jax_traceback_filtering", "off")

################################################################
# Model parameters

T = 120
dt = 1e-2
nt = int(T / dt)
t = jnp.linspace(0., T, nt + 1)

# DN shell model parameters
dim = 17                                    # number of discretized concentric shells
c1 = 0.01                                   # constant parameter for getG
c2 = 0.0                                    # constant parameter for getG
nu = 0.01                                   # viscosity
k = 2 ** jnp.linspace(0, dim - 1, dim)      # wavenumbers
sigma = 1.                                  # noise strength for direct sampling
chi_sqrt = sigma * k ** (-3.)               # forcing cov sqrt
init_u = jnp.zeros(dim)
branch = 302
targetObs = 29.0

# Default eigenvalue count for the prefactor determinant
DEFAULT_NEVALS = 200

PARAMETER = "velo"

# parameterizing the times where eigenvalues and eigfuncs are saved for each run
SAVE_TIMES = {"velo":{
                0.0001:[-29.0, -20.0, -10.0, 0.0, 10.0, 20.0, 34.0],
                0.001:[-19.0, -10.0, 0.0, 10.0, 20.0, 30.0, 39.0],
                0.01:[-9.0, 0.0, 10.0, 20.0, 29.0],
                0.1:[-9.0, 0.0, 5.0, 10.0, 19.0],
                1:[-4.0, 0.0, 5.0, 9.0]
                },
              "edr":{
                0.0001:[0.2, 4.0, 8.0, 12.0, 16.0, 19.8],
                0.001:[0.2, 4.0, 8.0, 12.0, 14.8],
                0.01:[0.2, 4.0, 8.0, 12.0, 14.8],
                0.1:[0.2, 4.0, 8.0, 12.0, 14.8],
                1:[0.2, 4.0, 8.0, 12.0, 14.8]
              },
              "grad":{
                0.0001:[-29.0, -20.0, -10.0, 0.0, 10.0, 20.0, 30.0, 44.0],
                0.001:[-14.0, -10.0, 0.0, 10.0, 20.0, 30.0, 40.0, 49.0],
                0.01:[-9.0, 0.0, 10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 69.0],
                0.1:[-6.0, 0.0, 10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 89.0],
                1:[-2.0, 0.0, 10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 89.0]
              }}

################################################################
# Data directory templates
# Each call to compute_prefactor formats these with its own target_observable.
INPUT_DIR_TEMPLATE = (f"/Users/rawdata/Downloads/data/{PARAMETER}/"
                      "seq_nu_{}_c1_{}_c2_{}_obs_{}_date_*_*_*_*_*_*_branch_{}/")

# Directory where individual per-target prefactor .npy files will be written.
OUTPUT_DIR_TEMPLATE = (
    f"/Users/rawdata/Downloads/data/{PARAMETER}/"
    "prefactors_nu_{}_c1_{}_c2_{}_branch_{}/"
)

################################################################
# DN shell model functions (needed for the second-variation operator)

jgetIF_single = lambda u, dt: u * jnp.exp(-nu * k ** 2 * dt)


@jax.jit
def jgetG_single(u):
    Gu = jnp.zeros_like(u)
    Gu = Gu.at[1:].add(c1 * k[1:] * u[:-1] * u[:-1])
    Gu = Gu.at[:-1].add(-c1 * k[1:] * u[:-1] * u[1:])
    Gu = Gu.at[1:].add(c2 * k[1:] * u[:-1] * u[1:])
    Gu = Gu.at[:-1].add(-c2 * k[1:] * u[1:] * u[1:])
    return Gu


@jax.jit
def jgetChi_single(dW):
    return chi_sqrt * dW


@jax.jit
def jgetF(u):
    if PARAMETER == "grad":
        return jnp.sum(u * k)  # return velocity gradient across all shells
    elif PARAMETER == "edr":
        return nu * jnp.sum((u**2) * (k**2))  # return energy dissipation rate across all shells
    elif PARAMETER == "velo":
        return jnp.sum(u)
    else:
        print(f"No target observable with identifier \"{PARAMETER}\"")
        return None


@jax.jit
def jgetTimeIntegral(a, b):
    ret = jnp.sum(a * b, axis=1) * dt
    return jnp.sum(ret[:-1])


@jax.jit
def integrate_forward_jax(etaa):
    @jax.checkpoint  # rematerialize during backward pass
    def step(u, etaaa):
        ret_u = jgetIF_single(u + dt * jgetG_single(u) + dt * jgetChi_single(etaaa), dt)
        return ret_u, ret_u

    uT, u = jax.lax.scan(step, copy.copy(init_u), etaa[:-1])
    u = jnp.concatenate([init_u[None, :], u], axis=0)
    return u, jgetF(uT)


def integrate_forward_obs_jax(etaa):
    return integrate_forward_jax(etaa)[1]


################################################################
# Loading saved instanton data

def _find_data_dir(target_observable: float) -> str:
    """
    Locate the saved-instanton directory for a given target observable.
    Raises FileNotFoundError if no directory matches, and warns if multiple do.
    """
    pattern = INPUT_DIR_TEMPLATE.format(nu, c1, c2, target_observable, branch)
    matches = glob.glob(pattern)
    if not matches:
        raise FileNotFoundError(
            f"No instanton data directory found for target_observable={target_observable}.\n"
            f"  Pattern: {pattern}"
        )
    if len(matches) > 1:
        # Pick the most recent by directory mtime so repeated runs are deterministic-ish
        matches.sort(key=os.path.getmtime, reverse=True)
        print(f"  [warn] {len(matches)} directories match for z={target_observable}; "
              f"using most recent: {matches[0]}")
    return matches[0]


def _load_saved_instanton(target_observable: float):
    """
    Load (eta, lbda, action) from saved .npy files for the given target.
    Returns numpy arrays / Python floats — JAX-free at this stage.
    """
    data_dir = _find_data_dir(target_observable)
    eta = np.load(os.path.join(data_dir, "inst_eta.npy"))
    u = np.load(os.path.join(data_dir, "inst_u.npy"))

    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    for i in range(dim):
        ax[0].plot(u[:, i], label=f'shell {i}')
        ax[1].plot(eta[:, i], label=f'shell {i}')
    ax[0].set_title('u')
    ax[0].set_ylabel('u')
    ax[0].set_xlabel('i in range(nt+1)')
    ax[0].grid(True)

    ax[1].set_title('eta')
    ax[1].set_ylabel('eta')
    ax[1].set_xlabel('i in range(nt+1)')
    ax[1].grid(True)

    plt.tight_layout()
    plt.show()

    lbda = float(np.load(os.path.join(data_dir, "inst_lbda.npy")))
    action = float(np.load(os.path.join(data_dir, "inst_act.npy")))
    return eta, lbda, action, data_dir


################################################################
# Second variation eigenvalues + prefactor

def _find_second_variation_eigenvalues(eta: np.ndarray,
                                       lbda: float,
                                       z: float,
                                       n_evals: int = DEFAULT_NEVALS,
                                       project_eta_perp: bool = True):
    """
    Find the leading n_evals eigenvalues of the second variation operator
    A_lambda projected onto the subspace orthogonal to eta.
    """
    eta_jax = jnp.asarray(eta)

    @jax.jit
    def Adeta(deta):
        return (lbda / dt) * jax.jvp(
            jax.grad(integrate_forward_obs_jax),
            (eta_jax,),
            (jnp.asarray(deta),),
        )[1]

    eta_inner = float(jgetTimeIntegral(eta_jax, eta_jax))
    eta_np = np.asarray(eta_jax)

    class SecondVariationOperator(LinearOperator):
        def __init__(self):
            self.shape = (dim * (nt + 1), dim * (nt + 1))
            self.dtype = np.dtype('float64')
            self.counter = 0

        def _matvec(self, inp):
            self.counter += 1
            if self.counter % 10 == 0:
                print(f'  A_lambda application no. {self.counter}')
            inpp = np.reshape(inp, (nt + 1, dim)).astype(np.float64)
            if project_eta_perp:
                proj = float(jgetTimeIntegral(jnp.asarray(inpp), eta_jax)) / eta_inner
                inpp = inpp - proj * eta_np
            ret = np.asarray(Adeta(inpp))
            if project_eta_perp:
                proj = float(jgetTimeIntegral(jnp.asarray(ret), eta_jax)) / eta_inner
                ret = ret - proj * eta_np
            return ret.flatten()

    A = SecondVariationOperator()
    evals, evecs = eigs(A, n_evals, which='LM', tol=1e-6, ncv=2*n_evals+1, maxiter=5000)

    evals = evals.real
    idx = np.argsort(np.abs(evals))[::-1]
    evals = evals[idx]
    evecs = evecs[:, idx]  # apply same sort to eigenvectors

    indices = np.arange(1, len(evals) + 1)
    abs_evals = np.abs(evals)

    pos_mask = evals > 0
    neg_mask = evals < 0

    fig, ax = plt.subplots()

    # main plot: eigenvalue spectrum
    ax.plot(indices[pos_mask], abs_evals[pos_mask], 'o', markersize=6, color='blue', label='Positive')
    ax.plot(indices[neg_mask], abs_evals[neg_mask], 'x', markersize=4, color='red', label='Negative')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('Index i')
    ax.set_ylabel(r'$|\lambda_z^{(i)}|$')
    ax.set_title(r'Eigenvalue spectrum of $A_\lambda$')
    ax.legend()

    # inset: running determinant product
    running_det = np.cumprod(1. - evals)

    ax_inset = fig.add_axes([0.22, 0.2, 0.25, 0.25])  # [left, bottom, width, height]
    ax_inset.plot(indices, running_det, '-', color='black', linewidth=1)
    ax_inset.set_xlabel('$i$', fontsize=5)
    ax_inset.set_ylabel(r'$\prod_{i=1}^{n}(1 - \lambda_i)$', fontsize=5)
    ax_inset.tick_params(labelsize=5)
    ax_inset.set_title(r'Running $\det(I - A)$', fontsize=5)

    plt.figtext(0.5, 0.01,
                r"Parameters used were $T={}$, $\nu = {}$, $c_1 = {}$, $c_2 = {}$, and $z = {}$".format(T, nu, c1, c2,                                                                                  z),
                ha='center',
                fontsize=9)
    plt.tight_layout()
    plt.savefig(f"/Users/rawdata/Downloads/data/eig_spec_2.pdf")
    plt.show()

    # Save the leading few eigenvectors, reshaped to (nt+1, dim)
    if z in SAVE_TIMES[PARAMETER][c1]:
        save_dir = f"/Users/rawdata/Downloads/data/{PARAMETER}/prefactors_nu_{nu}_c1_{c1}_c2_{c2}_branch_{branch}"
        os.makedirs(save_dir, exist_ok=True)
        t_np = np.linspace(0, T, nt + 1)

        # Save and plot the first 5 eigenfunctions (the informative ones;
        # higher modes increasingly resemble Fourier modes, per Schorlepp et al.)
        for j in range(10):
            eigfunc = np.real(evecs[:, j]).reshape(nt + 1, dim)
            np.save(f"{save_dir}/eigfunc_j_{j}_z_{z}.npy", eigfunc)

            # Full-domain plot
            fig, ax = plt.subplots()
            for n in range(dim):
                ax.plot(t_np, eigfunc[:, n], label=f'shell {n}', alpha=0.7)
            ax.set_xlabel(r'$t$')
            ax.set_ylabel(rf'$v_{{{j}}}(t, n)$, $\lambda = {evals[j]:.4f}$')
            ax.legend(fontsize=6, ncol=3)
            ax.set_title(f'Eigenfunction j={j}, z={z}, dt={dt}')
            plt.savefig(f"{save_dir}/eigfunc_{j}_z_{z}_full.pdf", bbox_inches='tight')
            plt.close(fig)

            # Zoom plot: last 10% of the time domain (where the burst lives)
            t_cut = 0.9 * T
            mask = t_np >= t_cut
            fig, ax = plt.subplots()
            for n in range(dim):
                ax.plot(t_np[mask], eigfunc[mask, n], label=f'shell {n}', alpha=0.7)
            ax.set_xlabel(r'$t$')
            ax.set_ylabel(rf'$v_{{{j}}}(t, n)$, $\lambda = {evals[j]:.4f}$')
            ax.legend(fontsize=6, ncol=3)
            ax.set_title(f'Eigenfunction j={j}, z={z}, dt={dt}, zoom to t≥{t_cut:.1f}')
            plt.savefig(f"{save_dir}/eigfunc_{j}_z_{z}_zoom.pdf", bbox_inches='tight')
            plt.close(fig)

    return evals


def _prefactor_from_solution(eta: np.ndarray,
                             lbda: float,
                             action: float,
                             z: float,
                             n_evals: int = DEFAULT_NEVALS) -> float:
    """
    Given an instanton (eta, lbda, action), compute the prefactor:

        c(z) = |lambda| / sqrt(2 * I(z) * det(I - B))

    where det(I - B) is approximated by prod(1 - evals) over the leading
    n_evals eigenvalues of the second variation operator.
    """
    evals = _find_second_variation_eigenvalues(eta, lbda, z, n_evals=n_evals)
    if z in SAVE_TIMES[PARAMETER][c1]:
        save_dir = f"/Users/rawdata/Downloads/data/{PARAMETER}/prefactors_nu_{nu}_c1_{c1}_c2_{c2}_branch_{branch}"
        os.makedirs(save_dir, exist_ok=True)
        np.save(f"{save_dir}/evals_A_project_perp_True_z_{z}.npy", evals)
    det_I_minus_B = np.abs(float(np.prod(1.0 - evals)))
    prefactor = float(np.abs(lbda) / np.sqrt(2.0 * action * det_I_minus_B))
    print(f"  lambda={lbda:.4f}, action={action:.4f}, "
          f"det(I-B)={det_I_minus_B:.6e}, prefactor={prefactor:.6e}")
    return prefactor


################################################################
# Public entry point

def compute_prefactor(target_observable: float,
                      n_evals: int = DEFAULT_NEVALS) -> float:
    """
    Compute the instanton prefactor c(z) for a single target observable,
    using previously saved instanton data.

    Loads (eta, lambda, action) from disk, computes the second-variation
    eigenvalues, and returns the prefactor as a Python float.

    Parameters
    ----------
    target_observable : float
        The target value z of the observable F(u). Used to locate the
        matching saved-instanton directory.
    n_evals : int, optional
        Number of leading eigenvalues to use in the determinant approximation.

    Returns
    -------
    float
        The prefactor c(z).
    """
    z = float(target_observable)
    print(f"\n=== compute_prefactor(z={z}) ===")

    eta, lbda, action, data_dir = _load_saved_instanton(z)
    print(f"  loaded instanton from: {data_dir}")
    print(f"  lambda={lbda:.4f}, action={action:.4f}, eta.shape={eta.shape}")

    return _prefactor_from_solution(eta, lbda, action, z=z, n_evals=n_evals)


################################################################
# CLI

if __name__ == "__main__":
    import sys
    z = float(sys.argv[1]) if len(sys.argv) > 1 else targetObs
    pref = compute_prefactor(z)
    save_dir = f"/Users/rawdata/Downloads/data/{PARAMETER}/prefactors_nu_{nu}_c1_{c1}_c2_{c2}_branch_{branch}"
    os.makedirs(save_dir, exist_ok=True)
    np.save(f"{save_dir}/prefactor_obs_{targetObs}.npy", np.asarray(pref))
    print(f"\nFinal prefactor at z={z}: {pref}")
