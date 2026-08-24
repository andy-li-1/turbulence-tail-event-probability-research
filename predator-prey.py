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

######################################################################
# set up parameters for model

T = 10.
nt = 1000
dt = T / nt
t = jnp.linspace(0., T, nt + 1)
dim = 2

# rates of the model
alpha = 1.
beta  = 5.
gamma = 1.
delta = 0.1

# initial condition: start at fixed point
x0 = 1. / (5. * jnp.sqrt(2.))
y0 = x0 + 0.2
init_phi = jnp.array([x0, y0])

targetObs = 0.5

print('################################################')
print('################## Parameters ##################')
print('T =', T)
print('nt =', nt)
print('z =', targetObs)
print('################################################')


plots = True # option for producing plots of instanton and operator spectra
projectEtaPerp = True # set to false for calculating MGF prefactor and explicit transformation to tail probability; only works for convex rate function
######################################################################
# model functions

def jgetB(x):
    return jnp.array([-beta * x[0] * x[1] + alpha * x[0] + delta,
                      +beta * x[0] * x[1] - gamma * x[1] + delta])

def jgetSigma(x,dx):
    return jnp.array([jnp.sqrt(beta * x[0] * x[1] + alpha * x[0] + delta) * dx[0],
                      jnp.sqrt(beta * x[0] * x[1] + gamma * x[1] + delta) * dx[1]])

def jgetF(x):
    return x[0]

######################################################################
# quadrature rule for time integrals

def jgetTimeIntegral(a, b):
    ret = jnp.sum(a * b, axis = 1) * dt
    return jnp.sum(ret[:-1])
    
######################################################################
# for invertible diffusion matrix; use this to get theta easily from theta_z = a^{-1} sigma eta_z

def jgetAInverse(x, dx):
    sigma = jgetSigma(x, jnp.eye(dim))
    return jnp.linalg.inv(sigma @ sigma.T) @ dx

######################################################################
# JAX implementation of noise to observable map that can be automatically differentiated

def integrate_forward_jax(etaa):
    def scan_fun(phi, etaaa):
        ret_phi = phi + dt * (jgetB(phi) + jgetSigma(phi, etaaa))
        return ret_phi, ret_phi
    phiT, phi = jax.lax.scan(scan_fun, copy.copy(init_phi), etaa[:-1])
    phi = jnp.concatenate([init_phi[None, :], phi], axis = 0)
    return phi, jgetF(phiT)

def integrate_forward_obs_jax(etaa):
    return integrate_forward_jax(etaa)[1]

# jax numpy implementation of forward map that returns the full state space instanton trajectory for diagnostics
def integrate_forward(etaa):
    phi = jnp.zeros((nt + 1, dim))
    phi = phi.at[0,:].set(jnp.array([x0, y0]))
    for i in range(nt):
        phi = phi.at[i+1,:].set(phi[i] + dt * \
             (jgetB(phi[i]) + jgetSigma(phi[i], etaa[i])))
    return phi, jgetF(phi[-1])

# vectorized numpy implementation for direct Monte Carlo simulations of SDE
def getSamples(nParallel = int(1e4), maxSim = int(1e6), eps = 0.01, conf = 0.95):
    print('################################################')
    print('Performing direct sampling with eps = {}'.format(eps))
    print('Copmuting {} samples, with {} SDE simluations in parallel'.format(maxSim, nParallel))
    obss = np.zeros((maxSim,))
    nSamples = 0
    for run in tqdm.tqdm(range(maxSim//nParallel)):
        x = copy.copy(init_phi)[:, None]
        for j in range(nt):
            x = x + dt * jgetB(x) + np.sqrt(eps * dt) \
                        * jgetSigma(x, np.random.randn(dim, nParallel))
        obss[nSamples:(nSamples+nParallel)] = jgetF(x)
        nSamples += nParallel
    print('')
    print('Completed MC {} simulations with mean observable {}+-{}'.format(nSamples, np.mean(obss), np.std(obss)))

    prob = np.sum(np.where(obss >= targetObs, 1., 0.)) / nSamples
    std  = np.sqrt(prob * (1. - prob) /nSamples)
    c = norm.ppf((conf + 1) / 2.)
    print('')
    print('Estimated tail probability from this data: {}'.format(prob))
    print('Asymptotic {}% confidence interval: [{},{}]'.format( \
            100 * conf, round(prob - c * std, int(-np.log10(prob) + 3)), \
            round(prob + c * std, int(-np.log10(prob) + 3))))
    print('################################################')
    return obss

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
            self.eta = np.random.randn(nt + 1, dim)**2 * 0.1 + 0.2
        else:
            self.eta = initialEta
        
        def target_func(eta):
            etaa = jnp.asarray(np.reshape(eta, (nt + 1, dim)))
            obs = integrate_forward_obs_jax(etaa)
            ret = 0.5 * jgetTimeIntegral(etaa, etaa) - lbda * (obs - targetObservable) + mu / 2. * (targetObservable - obs)**2
            return ret

        target_func_grad = jax.jacrev(target_func)
        res = spo.minimize(target_func, self.eta.flatten(), method = 'L-BFGS-B', jac = target_func_grad, options = {'ftol': 1e-8, 'gtol': 1e-8, 'disp': False})
        print(res)
        res = res.x

        self.eta = copy.copy(jnp.reshape(res, (nt + 1, dim)))
        
        phi, obsValue = integrate_forward(self.eta)
        
        action = 0.5 * jgetTimeIntegral(self.eta, self.eta)
        print('################################################')
        print('Parameters of the solution:')
        print('lambda =', lbda)
        print('mu =', mu)
        print('observable =', obsValue)
        print('Action =', action)
        ret = obsValue, action, lbda, copy.copy(self.eta), phi
        dS = 0.5 * np.real(np.sum(self.eta**2., axis = 1) * dt)
        ret = ret + (dS,)
        return ret
        
    def searchInstantonViaAugmented(self, targetObservable, muMin = np.log10(5.), muMax = np.log10(100.), nMu = 8, initLbda = 1.):
        print("Find instanton for observable value", targetObservable, "via augmented Lagrangian method.")
        muList = np.logspace(muMin, muMax, nMu)
        obsValue, action, lbda, eta, phi, dS = self.optimize(initLbda, targetObservable = targetObservable, mu =  muList[0])
        print("Mu = {}, lambda = {} yields observable = {} and action = {}".format(muList[0], lbda, obsValue, action))
        lbda = muList[0] * (targetObservable - obsValue) + initLbda
        for j in range(1, nMu):
            obsValue, action, lbda, eta, phi, dS = self.optimize(lbda, targetObservable = targetObservable, mu =  muList[j], initialEta = eta)
            print("Mu = {}, lambda = {} yields observable = {} and action = {}".format(muList[j], lbda, obsValue, action))
            lbda = muList[j] * (targetObservable - obsValue) + lbda
        return obsValue, action, lbda, eta, phi, dS

    def findTraceAMinusAtilde(self, lbda, eta, theta, nEvals = 200, projectEtaPerp = True):

        def thetasigmadeta(detaa, etaa):
            phii, _ = integrate_forward_jax(etaa)
            return jgetTimeIntegral(theta, jgetSigma(phii.T, detaa.T).T)

        def thetasigmadetagamma(detaa):
            h = jax.tree_util.Partial(thetasigmadeta, detaa)
            return jax.jvp(h, (eta,), (jnp.array(detaa),))[1]

        tmp = jax.grad(thetasigmadetagamma)
        Atildedeta = lambda deta : tmp(deta) / dt

        Adeta = lambda deta: lbda / dt * jax.jvp(jax.grad(integrate_forward_obs_jax), \
                            (eta,),(jnp.array(deta),))[1]

        class regularizedSecondVariationOperator(LinearOperator):
            def __init__(self):
                self.shape  = (dim * (nt + 1), dim * (nt + 1))
                self.dtype = np.dtype('double')
                self.counter = 0
            def _matvec(self, inp):
                self.counter += 1
                print('A_λ - tilde A_λ Application no. {}'.format(self.counter))
                inpp = np.reshape(inp, (nt + 1, dim))
                if projectEtaPerp:
                    inpp = inpp - jgetTimeIntegral(inpp, eta) / jgetTimeIntegral(eta, eta) * eta
                ret = Adeta(inpp) - Atildedeta(inpp)
                if projectEtaPerp:
                    ret = ret - jgetTimeIntegral(ret, eta) /  jgetTimeIntegral(eta, eta) * eta
                return ret.flatten()

        AMinusAtilde = regularizedSecondVariationOperator()
        evals, evecs = eigs(AMinusAtilde, nEvals, which = 'LM', tol = 1E-8)

        evals = evals.real
        idx = np.argsort(np.abs(evals))[::-1]
        evals = evals[idx]
        evecs = evecs[:, idx]
        evecs = np.reshape(evecs, (nt + 1, dim, len(evals)))
        print('Evals of A - A tilde:', evals)

        ret = np.sum(evals), evals

        if projectEtaPerp:
            # also return < e_z, A tilde e_z>
            ret = ret + (jgetTimeIntegral(eta, Atildedeta(eta)) / jgetTimeIntegral(eta, eta),)

        return ret

    def findSecondVariationEigenvalues(self, eta, lbda, nEvals = 200, projectEtaPerp = True):

        Adeta = lambda deta: lbda / dt * jax.jvp(jax.grad(integrate_forward_obs_jax), \
                             (eta,),(jnp.array(deta),))[1]
        
        class SecondVariationOperator(LinearOperator):
            def __init__(self):
                self.shape  = (dim * (nt + 1), dim * (nt + 1))
                self.dtype = np.dtype('double')
                self.counter = 0
            def _matvec(self, inp):
                self.counter += 1
                print('A_λ Application no. {}'.format(self.counter))
                inpp = np.reshape(inp, (nt + 1, dim))
                if projectEtaPerp:
                    inpp = inpp - jgetTimeIntegral(inpp, eta) / jgetTimeIntegral(eta, eta) * eta
                ret = Adeta(inpp)
                if projectEtaPerp:
                    ret = ret - jgetTimeIntegral(ret, eta) /  jgetTimeIntegral(eta, eta) * eta
                return ret.flatten()
        
        A = SecondVariationOperator()
        evals, evecs = eigs(A, nEvals, which = 'LM', tol = 1E-8)
        
        evals = evals.real
        idx = np.argsort(np.abs(evals))[::-1]
        evals = evals[idx]
        evecs = evecs[:, idx]
        evecs = np.reshape(evecs, (nt + 1, dim, len(evals)))
        print('Evals of A:', evals)
        return evals, evecs

######################################################################
######################################################################
######################################################################

if __name__ == '__main__':

    now = datetime.now()
    dt_string = now.strftime('%Y_%m_%d_%H_%M_%S')
    data_dir = 'data/obs_{}_date_{}'.format(targetObs, dt_string)
    if not os.path.isdir(data_dir):
        os.makedirs(data_dir)
    np.save(data_dir + '/obs.npy', targetObs)

    # uncomment for output to log file
    #sys.stdout = open('{}/output_.log'.format(data_dir), 'w', buffering = 1)

    ############################################################
    # Monte Carlo of SDE
    eps = 0.01
    getSamples(nParallel = int(1e4), maxSim = int(5.2e5), eps = eps, conf = 0.95)

    ############################################################
    # instanton computation
    start = time.time()

    instanton = Instanton()

    obsValue, action, lbda, eta, phi, dS = instanton.searchInstantonViaAugmented(targetObs)

    np.save(data_dir + '/inst_obs.npy', obsValue)
    np.save(data_dir + '/inst_act.npy', action)
    np.save(data_dir + '/inst_lbda.npy', lbda)
    np.save(data_dir + '/inst_eta.npy', eta)
    np.save(data_dir + '/inst_phi.npy', phi)
    np.save(data_dir + '/inst_ds.npy', dS)

    print('Needed', time.time() - start, 'seconds for the instanton computation...')

    if plots:
        plt.figure()
        plt.plot(phi[:,0], phi[:,1])
        plt.axvline(x = targetObs, color = 'black', linestyle = 'dashed')
        plt.scatter([x0], [y0], color = 'orange')
        plt.xlabel(r'$x$')
        plt.ylabel(r'$y$')
        plt.savefig(data_dir + '/inst.pdf', bbox_inches = 'tight')
        plt.close()

    ############################################################
    # load previously computed instanton

    # data_dir = 'data/...'
    # obsValue = np.load(data_dir + '/inst_obs.npy')
    # action   = np.load(data_dir + '/inst_act.npy')
    # lbda     = np.load(data_dir + '/inst_lbda.npy')
    # eta      = np.load(data_dir + '/inst_eta.npy')
    # phi      = np.load(data_dir + '/inst_phi.npy')
    # dS       = np.load(data_dir + '/inst_ds.npy')
    # instanton = Instanton()

    ############################################################
    # calculate tr(A - Atilde) via dominant eigs:

    theta = jnp.array([jgetAInverse(phi[i], jgetSigma(phi[i], eta[i])) for i in range(nt + 1)])
    res = instanton.findTraceAMinusAtilde(lbda, eta, theta, nEvals = 200, projectEtaPerp = projectEtaPerp)
    if projectEtaPerp:
        trAAtilde_evals, evals, ezAtildeez = res
    else:
        trAAtilde_evals, evals = res

    np.save(data_dir + '/AAtilde_evals_project_perp_{}.npy'.format(projectEtaPerp), evals)
    np.save(data_dir + '/trAAtilde_evals_project_perp_{}.npy'.format(projectEtaPerp), trAAtilde_evals)
    if projectEtaPerp:
        print('ezAtildeez =', ezAtildeez)
        np.save(data_dir + '/ezAtildeez.npy', ezAtildeez)

    print('trAAtilde from eigenval:', trAAtilde_evals)

    if plots:

        m = np.linspace(1, len(evals), len(evals))

        plt.figure()
        plt.plot(m, np.cumsum(evals), '.-')
        plt.grid()
        plt.xlabel(r'$m$')
        plt.savefig(data_dir + '/AAtilde_cumsum_project_perp_{}.pdf'.format(projectEtaPerp), bbox_inches = 'tight')
        plt.close()

        plt.figure()
        plt.loglog(m,evals, '.')
        plt.loglog(m,-evals, 'x')
        plt.xlabel(r'$i$')
        plt.ylabel(r'$\left| \hat{\mu}_z^{(i)} \right|$')
        plt.grid()
        plt.savefig(data_dir + '/AAtilde_evals_project_perp_{}.pdf'.format(projectEtaPerp), bbox_inches = 'tight')
        plt.close()

    ############################################################
    # calculate det_2(Id - A) via dominant eigs:

    evals, evecs = instanton.findSecondVariationEigenvalues(eta, lbda, nEvals = 200, projectEtaPerp = projectEtaPerp)
    np.save(data_dir + '/evals_A_project_perp_{}.npy'.format(projectEtaPerp), evals)
    np.save(data_dir + '/evecs_A_project_perp_{}.npy'.format(projectEtaPerp), evecs)
    
    evals = np.load(data_dir + '/evals_A_project_perp_{}.npy'.format(projectEtaPerp))
    evecs = np.load(data_dir + '/evecs_A_project_perp_{}.npy'.format(projectEtaPerp))
    det2 = np.prod((1.- evals) * np.exp(evals))

    print('det2 =', det2)

    if plots:
        
        m = np.linspace(1, len(evals), len(evals))
        plt.figure()
        plt.loglog(m,evals, '.')
        plt.loglog(m,-evals, 'x')
        plt.xlabel(r'$i$')
        plt.ylabel(r'$\left| \mu_z^{(i)} \right|$')
        plt.grid()
        plt.savefig(data_dir + '/evals_A_project_perp_{}.pdf'.format(projectEtaPerp), bbox_inches = 'tight')
        plt.close()
    
        plt.figure()
        plt.plot(m,np.cumsum(np.abs(evals)))
        plt.xlabel('m')
        plt.ylabel(r'$\sum_{i=1}^m \left| \mu_z^{(i)}\right|$')
        plt.grid()
        plt.savefig(data_dir + '/evals-sum-abs_A_project_perp_{}.pdf'.format(projectEtaPerp), bbox_inches = 'tight')
        plt.close()
    
        plt.figure()
        plt.plot(m,np.cumsum(evals))
        plt.xlabel('m')
        plt.ylabel(r'$\sum_{i=1}^m \mu_z^{(i)}$')
        plt.grid()
        plt.savefig(data_dir + '/evals-sum_A_project_perp_{}.pdf'.format(projectEtaPerp), bbox_inches = 'tight')
        plt.close()

        plt.figure()
        plt.plot(m,np.cumsum(evals**2.))
        plt.xlabel('m')
        plt.ylabel(r'$\sum_{i=1}^m \left( \mu_z^{(i)} \right)^2$')
        plt.grid()
        plt.savefig(data_dir + '/evals-sum-squared_A_project_perp_{}.pdf'.format(projectEtaPerp), bbox_inches = 'tight')
        plt.close()
    
        plt.figure()
        plt.plot(m,np.cumprod(1.- evals))
        plt.xlabel('m')
        plt.ylabel(r'$\prod_{i=1}^m \left( 1 - \mu_z^{(i)} \right)$')
        plt.grid()
        plt.savefig(data_dir + '/det_A_project_perp_{}.pdf'.format(projectEtaPerp), bbox_inches = 'tight')
        plt.close()
    
        plt.figure()
        plt.plot(m,np.cumprod((1.- evals) * np.exp(evals)))
        plt.xlabel('m')
        plt.ylabel(r'$\prod_{i=1}^m \left( 1 - \mu_z^{(i)} \right) e^{\mu_z^{(i)}}$')
        plt.grid()
        plt.savefig(data_dir + '/det-2_A_project_perp_{}.pdf'.format(projectEtaPerp), bbox_inches = 'tight')
        plt.close()
    
    ############################################################
    if projectEtaPerp:
        prefProb = 1. / np.sqrt(2. * action * det2) * np.exp(.5 * trAAtilde_evals - .5 * ezAtildeez)
    else:
        # this calculates the tail probability prefactor from the MGF prefactor
        # and only works for locally convex rate functions
        prefJ =  1. / np.sqrt(det2) * np.exp(.5 * trAAtilde_evals)
        convertProb = np.sqrt(2 * action + np.sum(evals / (1. - evals) * \
                        np.sum(eta[:-1,:, None] * evecs[:-1,:,:] * dt, axis = (0,1))**2 \
                        / np.sum(evecs[:-1,:]**2 * dt, axis = (0,1))))
        prefProb = prefJ / convertProb

    print('Tail prob prefactor:', prefProb)
    np.save(data_dir + '/prefProb.npy', prefProb)
    print('Noise strength:', eps)
    print("Tail probability estimate:", np.sqrt(eps/(2. * np.pi)) * prefProb * np.exp(-action/eps))
    
