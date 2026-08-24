import sys
import copy
import os
from datetime import datetime
import time
import matplotlib.pyplot as plt
import numpy as np
from scipy.sparse.linalg import LinearOperator, eigs
from scipy.stats import norm

import jax
import jax.numpy as jnp
import scipy.optimize as spo
import tqdm
jax.config.update("jax_traceback_filtering", "off")

######################################################################
# set up parameters for model

T = 10
dt = 1e-3
nt = int(T / dt)
t = jnp.linspace(0., T, nt + 1)

# DN shell model parameters
dim = 17    # number of discretized concentric shells n = 0 ... dim - 1
c1 = 1    # constant parameter for getG
c2 = 0.0    # constant parameter for getG
nu = 1e-2   # viscosity
k = 2 ** jnp.linspace(0, dim-1, dim) # wavenumbers
sigma = 1.  # noise strength for direct sampling
chi_sqrt = sigma * k**(-3.)  # forcing cov sqrt

init_u = jnp.zeros(dim)
targetObs = 0

print('Running Instanton computation for DN shell model')
print(f'Parameters: T={T}, dt={dt}, sigma^2={sigma ** 2.}, dim={dim}')


plots = True # option for producing plots of instanton and operator spectra
######################################################################
# define DN shell model functions

jgetIF = lambda u, dt: u * jnp.exp(-nu * k[:, None]**2 * dt)  # integrating factor for linear terms in b times x
jgetIF_single = lambda u, dt: u * jnp.exp(-nu * k**2 * dt)

jgetEnergy = lambda u: 0.5 * u**2
jgetEnergyDissipation = lambda u : jnp.sum(nu * k[:,None]**2 * u**2, axis = 0)


# to be used in MC computation
def jgetG(u):
    
    # :param u: (dim, nPaths)
    # :return: (dim, nPaths)
    
    Gu = jnp.zeros_like(u)

    Gu = Gu.at[1:].add(c1 * k[1:, None] * u[:-1] * u[:-1])
    Gu = Gu.at[:-1].add(-c1 * k[1:, None] * u[:-1] * u[1:])
    Gu = Gu.at[1:].add(c2 * k[1:, None] * u[:-1] * u[1:])
    Gu = Gu.at[:-1].add(-c2 * k[1:, None] * u[1:] * u[1:])

    return Gu


# to be used in MC computation
def jgetChi(dW):
    """Forcing correlation matrix multiplied by random motion
        :param dW: (dim, nPaths) random Brownian motion
        :return: (dim, nPaths) forcing applied to system
        """
    return chi_sqrt[:, None] * dW

# to be used in instanton computation
@jax.jit
def jgetG_single(u):
    """
    :param u: (dim,) single state vector
    :return: (dim,)
    """
    Gu = jnp.zeros_like(u)
    Gu = Gu.at[1:].add(c1 * k[1:] * u[:-1] * u[:-1])
    Gu = Gu.at[:-1].add(-c1 * k[1:] * u[:-1] * u[1:])
    Gu = Gu.at[1:].add(c2 * k[1:] * u[:-1] * u[1:])
    Gu = Gu.at[:-1].add(-c2 * k[1:] * u[1:] * u[1:])
    return Gu

# to be used in instanton computation
@jax.jit
def jgetChi_single(dW):
    """
    :param dW: (dim,)
    :return: (dim,)
    """
    return chi_sqrt * dW

@jax.jit
def jgetF(u):
    return jnp.sum(u) # return total velocity across all shells
######################################################################
# quadrature rule for time integrals

@jax.jit
def jgetTimeIntegral(a, b):
    ret = jnp.sum(a * b, axis = 1) * dt
    return jnp.sum(ret[:-1])

######################################################################
# JAX implementation of noise to observable map that can be automatically differentiated

@jax.jit
def integrate_forward_jax(etaa):
    def scan_fun(u, etaaa):
        ret_u = jgetIF_single(u + dt * jgetG_single(u) + dt * jgetChi_single(etaaa), dt) # * jnp.exp(-nu * k**2 * dt)
        return ret_u, ret_u
    uT, u = jax.lax.scan(scan_fun, copy.copy(init_u), etaa[:-1])
    u = jnp.concatenate([init_u[None, :], u], axis = 0)
    return u, jgetF(uT)

def integrate_forward_obs_jax(etaa):
    return integrate_forward_jax(etaa)[1]

@jax.jit
def integrate_forward(etaa):
    def scan_fun(u, eta_i):
        u_next = jgetIF_single(u + dt * jgetG_single(u) + dt * jgetChi_single(eta_i), dt)
        ener = jnp.sum(nu * k ** 2 * u_next ** 2)
        return u_next, (u_next, ener)

    uT, (u_all, ener_diss) = jax.lax.scan(scan_fun, init_u, etaa[:-1])
    u_all = jnp.concatenate([init_u[None, :], u_all], axis=0)
    ener_diss = jnp.concatenate([jnp.array([jnp.sum(nu * k ** 2 * init_u ** 2)]), ener_diss])
    return u_all, jgetF(uT), ener_diss

"""
# jax numpy implementation of forward map that returns the full state space instanton trajectory for diagnostics
def integrate_forward(etaa):
    u = jnp.zeros((nt + 1, dim))
    u = u.at[0, :].set(init_u)
    for i in range(nt):
        u = u.at[i+1, :].set(
            jgetIF_single(u[i] + dt * jgetG_single(u[i]) + dt * jgetChi_single(etaa[i]), dt) # * jnp.exp(-nu * k**2 * dt)
        )
    ener_diss = jnp.sum(nu * k**2 * u**2, axis=1)
    return u, jgetF(u[-1]), ener_diss
"""

######################################################################
# numpy implementation for direct Monte Carlo simulations of the DN shell model

def getSamplePaths(nPaths=1000):
    print('Performing direct sampling for DN shell model energy spectra')
    print('Obtaining {} simulations'.format(nPaths))

    u_current = np.zeros((dim, nPaths))

    # Store snapshots at specific time indices for plotting and statistics
    time_indices = np.linspace(0, nt, 1000).astype(int)
    u_snapshots = {}
    u_snapshots[0] = u_current.copy()

    # Euler-Maruyama steps
    for j in tqdm.tqdm(range(nt)):
        dW = np.random.randn(dim, nPaths) * np.sqrt(dt)
        u_current = jgetIF(u_current + dt * jgetG(u_current) + jgetChi(dW), dt)
        if (j + 1) in time_indices:
            u_snapshots[j + 1] = u_current.copy()

    # u_T = np.sum(u_current.copy(), axis=0)
    u_T = u_current.copy() # keeps the dimensions separate, so result is at final time T, the velocity per dimension per sample
    velo_grad = k @ u_current.copy()

    return u_snapshots, time_indices, u_T, velo_grad


######################################################################

class Instanton():
    
    def optimize(self, lbda, targetObservable = 1., mu = 0., initialEta = None):
        print('################################################')
        print("Computing Instanton for fixed penalty parameter mu = {}".format(mu))
        print("and lagrange multiplier lbda = {} and target observable = {}".format(lbda, targetObservable))
        print("Performing updates p^{k+1} = p^k - alpha (p^k - z^k)")
        print("where alpha is found via Armijo line search,")
        print("Terminates if gradient L^2 norm is below eps.")
        print("################################################")
        # initialize fields
        if initialEta is None:
            self.eta = np.random.randn(nt + 1, dim) / jnp.sqrt(dt) # initializes eta as a random white noise (Gaussian)
        else:
            self.eta = initialEta

        def target_func(eta):
            etaa = jnp.asarray(np.reshape(eta, (nt + 1, dim)))
            obs = integrate_forward_obs_jax(etaa)
            ret = 0.5 * jgetTimeIntegral(etaa, etaa) - lbda * (obs - targetObservable) + mu / 2. * (
                        targetObservable - obs) ** 2
            return ret  # keep as JAX array for grad tracing

        target_func_grad_jax = jax.grad(target_func)

        def target_func_grad(eta):
            return np.array(target_func_grad_jax(eta))

        def target_func_wrapper(eta):
            return float(target_func(eta))

        res = spo.minimize(target_func_wrapper, self.eta.flatten(), method='L-BFGS-B', jac=target_func_grad,
                           options={'ftol': 1e-8, 'gtol': 1e-8, 'disp': False})
        print(res)
        res = res.x

        self.eta = copy.copy(jnp.reshape(res, (nt + 1, dim)))
        
        u, obsValue, ener_diss = integrate_forward(self.eta)
        
        action = 0.5 * jgetTimeIntegral(self.eta, self.eta) # this is 1/2 * L^2 norm
        print('################################################')
        print('Parameters of the solution:')
        print('lambda =', lbda)
        print('mu =', mu)
        print('observable =', obsValue)
        print('Action =', action)
        ret = obsValue, action, lbda, copy.copy(self.eta), u, ener_diss
        dS = 0.5 * np.real(np.sum(self.eta**2., axis = 1) * dt)
        ret = ret + (dS,)
        return ret
        
    def searchInstantonViaAugmented(self, targetObservable, muMin = np.log10(5.), muMax = np.log10(100.), nMu = 8, initLbda = 1., initialEta = None):
        print("Find instanton for observable value", targetObservable, "via augmented Lagrangian method.")
        muList = np.logspace(muMin, muMax, nMu)
        obsValue, action, lbda, eta, u, ener_diss, dS = self.optimize(initLbda, targetObservable = targetObservable, mu = muList[0], initialEta = initialEta)
        print("Mu = {}, lambda = {} yields observable = {} and action = {}".format(muList[0], lbda, obsValue, action))
        lbda = muList[0] * (targetObservable - obsValue) + initLbda
        for j in range(1, nMu):
            obsValue, action, lbda, eta, u, ener_diss, dS = self.optimize(lbda, targetObservable = targetObservable, mu =  muList[j], initialEta = eta)
            print("Mu = {}, lambda = {} yields observable = {} and action = {}".format(muList[j], lbda, obsValue, action))
            lbda = muList[j] * (targetObservable - obsValue) + lbda
        return obsValue, action, lbda, eta, u, ener_diss, dS

######################################################################
# extracting u_n snapshots at each integer second

def get_snapshots_per_second(u_trajectory):
    """
    Extract u_n at t = 1, 2, ..., T from the full trajectory.
    :param u_trajectory: (nt+1, dim) array from integrate_forward
    :return: (dim, int(T)) array where column j is u(:, t=j+1)
    """
    n_seconds = int(T)
    snapshots = np.zeros((dim, n_seconds))
    for s in range(n_seconds):
        time_index = int((s + 1) / dt)  # index for t = s+1
        snapshots[:, s] = np.array(u_trajectory[time_index, :])
    return snapshots

######################################################################
######################################################################
######################################################################

if __name__ == '__main__':

    if len(sys.argv) > 1:
        targetObs = float(sys.argv[1])

    now = datetime.now()
    dt_string = now.strftime('%Y_%m_%d_%H_%M_%S')
    data_dir = '/Users/rawdata/Downloads/data/inst/nu_{}_c1_{}_c2_{}_obs_{}_date_{}'.format(nu, c1, c2, targetObs, dt_string)
    if not os.path.isdir(data_dir):
        os.makedirs(data_dir)
    np.save(data_dir + '/obs.npy', targetObs)

    ############################################################
    # instanton computation
    print("Running Instanton Computation for DN Shell Model")

    print("DN Shell Model Parameters: ")
    print("viscosity: ", nu)
    print("c1: ", c1)
    print("c2: ", c2)
    print("target observation: ", targetObs)

    start = time.time()

    instanton = Instanton()
    t_np = np.array(t)
    obsValue, action, lbda, eta, u, ener_diss, dS = instanton.searchInstantonViaAugmented(targetObs)
    u_snapshots = get_snapshots_per_second(u) # to track the evolution of u over time for each shell

    np.save(data_dir + '/inst_obs.npy', obsValue)
    np.save(data_dir + '/inst_act.npy', action)
    np.save(data_dir + '/inst_lbda.npy', lbda)
    np.save(data_dir + '/inst_eta.npy', eta)
    np.save(data_dir + '/inst_u.npy', u)
    np.save(data_dir + '/ener_diss.npy', ener_diss)
    np.save(data_dir + '/inst_ds.npy', dS)
    np.save(data_dir + '/inst_u_per_second.npy', u_snapshots)

    if plots:
        plt.figure()
        for n in range(dim):
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
        plt.close()

        plt.figure()
        plt.plot(np.arange(dim), np.array(jgetEnergy(u[-1])))
        plt.xlabel(r'shell $n$')
        plt.ylabel(r'$E_n = \frac{1}{2} u_n^2$')
        plt.savefig(data_dir + '/inst_energy.pdf', bbox_inches='tight')
        plt.close()

        plt.figure()
        shells = np.arange(dim)
        plot_times = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

        for s in plot_times:
            if s == 0:
                u_at_t = np.array(init_u)
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

    print('Needed', time.time() - start, 'seconds for the instanton computation...')
