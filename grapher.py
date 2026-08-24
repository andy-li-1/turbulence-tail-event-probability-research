import numpy as np
import glob
import matplotlib.pyplot as plt
import re
import os

###############################################################
# inputs

nu = "0.01"
c1 = "1"
c2 = "0.0"
z = "1.0"

filename = "inst_act.npy"
filename_dns = "velo_grad.npy"
filename_pref = f"evals_A_project_perp_True_z_{z}.npy"

T="10"
branch = "502"
pref_branch = "500"

###############################################################
# constants

files = glob.glob(f"/Users/rawdata/Downloads/data/inst/seq_nu_{nu}_c1_{c1}_c2_{c2}_obs_*_date_*_*_*_*_*_*_branch_{branch}/{filename}")
files_branch = glob.glob(f"/Users/rawdata/Downloads/data/inst/seq_nu_{nu}_c1_{c1}_c2_{c2}_obs_*_date_*_*_*_*_*_*_branch_{branch}/{filename}")
files_dns = glob.glob(f"/Users/rawdata/Downloads/data/dns/nu_{nu}_c1_{c1}_c2_{c2}_sigma_*_T_{T}*_dt_*_seed_*/{filename_dns}")
# files_pref = glob.glob(f"/Users/rawdata/Downloads/data/inst/pref_nu_{nu}_c1_{c1}_c2_{c2}_obs_{z}_date_*_*_*_*_*_*/{filename_pref}")
files_pref = glob.glob(f"/Users/rawdata/Downloads/data/edr/prefactors_nu_{nu}_c1_{c1}_c2_{c2}_branch_*/{filename_pref}")

###############################################################
# constants for prefactor-computed pdf

LAYOUT = "B"  # "A" = flat arrays, "B" = one directory per z
DATA_DIR = f"/Users/rawdata/Downloads/data/inst/pref_nu_{nu}_c1_{c1}_c2_{c2}_obs_{z}_date_*_*_*_*_*_*/"  # where your .npy files live
EVALS_FILE = "evals_A_project_perp_True.npy"  # optional: precomputed eigenvalues per z
ACT_FILE = "inst_act.npy"
LBDA_FILE = "inst_lbda.npy"

# If eigenvalues are NOT precomputed, set this to True and fill in the
# `compute_evals_for_z` function below.
COMPUTE_EVALS_ON_THE_FLY = False


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_for_z(i, z):
    """
    Return (I_z, lambda_z, evals) for the i-th z value.
    `evals` is the 1-D array of eigenvalues of B_z used to form
    det(I - B_z) ~= prod(1 - evals).
    """
    if LAYOUT == "A":
        I_all = np.load(os.path.join(DATA_DIR, ACT_FILE))
        L_all = np.load(os.path.join(DATA_DIR, LBDA_FILE))
        I_z = float(I_all[i])
        lbda_z = float(L_all[i])

        if COMPUTE_EVALS_ON_THE_FLY:
            evals = compute_evals_for_z(z)
        else:
            evals_all = np.load(os.path.join(DATA_DIR, EVALS_FILE))
            evals = np.asarray(evals_all[i])
        return I_z, lbda_z, evals

    elif LAYOUT == "B":
        sub = os.path.join(DATA_DIR, f"z_{i:04d}")
        I_z = float(np.load(os.path.join(sub, ACT_FILE)))
        lbda_z = float(np.load(os.path.join(sub, LBDA_FILE)))
        if COMPUTE_EVALS_ON_THE_FLY:
            evals = compute_evals_for_z(z)
        else:
            evals = np.load(os.path.join(sub, EVALS_FILE))
        return I_z, lbda_z, evals

    else:
        raise ValueError(f"Unknown LAYOUT {LAYOUT!r}")


def compute_evals_for_z(z):
    """
    Fallback: compute the eigenvalues live. You must adapt this to your
    own instanton-solver workflow. The idea:

        instanton = build_instanton_for(z)
        eta, lbda = instanton.solve(...)
        evals, evecs = instanton.findSecondVariationEigenvalues(
            eta, lbda, nEvals=200, projectEtaPerp=True,
        )
        return evals

    Raising here by default so the user notices they need to fill it in.
    """
    raise NotImplementedError(
        "Set COMPUTE_EVALS_ON_THE_FLY=True and implement compute_evals_for_z "
        "to run the solver on the fly, or precompute inst_evals.npy."
    )


# ---------------------------------------------------------------------------
# c(z), rho(z)
# ---------------------------------------------------------------------------

def det_IminusB(evals):
    """det(I - B_z) ~= prod_k (1 - mu_k). The truncation to the top `nEvals`
    eigenvalues by magnitude is an approximation; remaining ones sit near 0
    so 1 - mu_k ~ 1 and contribute little to the product."""
    return float(np.prod(1.0 - np.asarray(evals, dtype=float)))


def prefactor_c(I_z, lbda_z, evals):
    """
    c(z) = |lambda_z| / sqrt( 2 * I(z) * det(I - B_z) ).

    Returns np.nan if the radicand is non-positive (happens past a caustic
    when det(I - B) flips sign or when I(z) == 0, i.e. at the mode).
    """
    det = det_IminusB(evals)
    radicand = 2.0 * I_z * det
    if radicand <= 0.0 or not np.isfinite(radicand):
        return np.nan, det
    return abs(lbda_z) / np.sqrt(radicand), det


def rho_z(I_z, c_z):
    """rho(z) = (1/sqrt(2*pi)) * c(z) * exp(-I(z))."""
    return c_z * np.exp(-I_z) / np.sqrt(2.0 * np.pi)

####################################################################################

# graphing inst files

def graph_inst_rate():
    files = glob.glob(f"/Users/rawdata/Downloads/data/velo/seq_nu_{nu}_c1_{c1}_c2_{c2}_obs_*_date_*_*_*_*_*_*_branch_{branch}/inst_act.npy")
    print(f"/Users/rawdata/Downloads/data/velo/seq_nu_{nu}_c1_{c1}_c2_{c2}_obs_*_date_*_*_*_*_*_*_branch_{branch}/inst_act.npy")


    obs_values = []
    data_values = []

    obs_values_dom = []
    data_values_dom = []

    for file in files:
        # Extract obs value from path
        match = re.search(r'obs_([\d.eE+\-]+)', file)
        if match:
            obs = float(match.group(1))
            data = np.load(file)

            # if data > 200:
            #     print(obs)

            obs_values_dom.append(obs)
            data_values_dom.append(data)

    sorted_pairs = sorted(zip(obs_values_dom, data_values_dom), key=lambda x: x[0])
    obs_values_dom, data_values_dom = zip(*sorted_pairs)

    sorted_pairs = sorted(zip(obs_values, data_values), key=lambda x: x[0])

    # plotting the rate function for each target observation
    plt.figure()
    # plt.plot(obs_values, data_values, marker='o', color='red', label = "subdominant branch")
    plt.plot(obs_values_dom, data_values_dom, marker='o', color='blue', label="dominant branch")
    plt.xlabel(r'Target Observation $\sum u_n$')
    plt.ylabel(r'Action $I(z) = \frac{1}{2} || \eta_z || _{L^2}^2$')
    plt.title('Rate Function Plot')
    plt.legend()
    plt.figtext(0.5, 0.01, r"Parameters used were $T={}$, $\nu = {}$, $c_1 = {}$, and $c_2 = {}$".format(T, nu, c1, c2), ha='center',
                fontsize=9)
    plt.tight_layout(rect=[0, 0.05, 1, 1])
    plt.savefig(f"/Users/rawdata/Downloads/data/velo/rate_function_c1_{c1}.pdf")
    plt.show()

# SHOULD BE KEPT NORMALIZED
def graph_inst_decay(normalized=True):
    print(f"number of files: {len(files_branch)}")

    obs_values = []
    data_values = []

    for file in files_branch:
        # Extract obs value from path
        match = re.search(r'obs_([\d.eE+\-]+)', file)
        if match:
            obs = float(match.group(1))
            data = np.load(file)

            # if data > 200:
            #     print(obs)

            obs_values.append(obs)
            data_values.append(data)

    sorted_pairs = sorted(zip(obs_values, data_values), key=lambda x: x[0])
    obs_values, data_values = zip(*sorted_pairs)

    if normalized:
        y = np.exp(-np.array(data_values))
        normalization = np.trapz(y, obs_values)  # ∫ e^{-I(z)} dz
        y_normalized = y / normalization

        plt.figure()
        plt.plot(obs_values, y_normalized, marker='o')
        plt.xlabel(r'Target Observation $\sum u_n$')
        plt.ylabel(r'PDF $\rho(z) \propto e^{-I(z)}$')
        plt.title('Instanton PDF')
        plt.figtext(0.5, 0.01, r"Parameters used were $T = {}$, $\nu = {}$, $c_1 = {}$, and $c_2 = {}$".format(T, nu, c1, c2),
                    ha='center',
                    fontsize=9)
        plt.tight_layout(rect=[0, 0.05, 1, 1])
        # plt.savefig("/Users/rawdata/Downloads/data/inst/pdf_agg_velo_inst_c1_{}.pdf".format(c1))
        plt.show()
    else:
        # plotting the exponential decay rate, e^{-I(z)}
        plt.figure()
        plt.plot(obs_values, np.exp(-np.array(data_values)), marker='o')
        plt.xlabel(r'Target Observation $\sum u_n$')
        plt.ylabel(r'Exponential Decay Rate $e^{-I(z)}$')
        plt.title('Exponential Decay Rate Plot')
        plt.figtext(0.5, 0.01, r"Parameters used were $T={}$, $\nu = {}$, $c_1 = {}$, and $c_2 = {}$".format(T, nu, c1, c2), ha='center',
                    fontsize=9)
        plt.tight_layout(rect=[0, 0.05, 1, 1])
        # plt.savefig("/Users/rawdata/Downloads/data/inst/pdf_agg_velo_inst_c1_{}.pdf".format(c1))
        plt.show()

def graph_dns(parameter="velo", save=False):
    FILENAME_MAP = {"velo":"u.npy", "grad":"velo_grad.npy", "edr":"ener_diss.npy"}
    TITLE_MAP = {"velo": "PDF of Agg Velo", "grad": "PDF of Velocity Gradient", "edr": "PDF of Energy Dissipation Rate"}
    XLABEL_MAP = {"velo": r'$\sum u_n$', "grad": r'$\sum u_n k_n$', "edr": r'$\nu \sum u_n^2 k_n^2$'}

    filename_dns = FILENAME_MAP[parameter]
    files_dns = glob.glob(
        f"/Users/rawdata/Downloads/data/dns/nu_{nu}_c1_{c1}_c2_{c2}_sigma_*_T_{T}*_dt_*_seed_*/{filename_dns}")

    print(f"number of dns files: {len(files_dns)}")

    all_velo = []

    for file in files_dns:
        data = np.load(
            file)  # data := one list of length 10,000 for each file in files, each value represents the total EDR (sum over shells) at time T for a given sample
        all_velo.append(data)  # appending energy dissipation rate at time t=T=10 for all paths

    all_velo = np.concatenate(all_velo)

    print(f"number of samples: {len(all_velo)}")
    print(f"average velocity: {np.mean(all_velo)}")

    plt.style.use('default')
    fig, ax = plt.subplots()

    hist, bins = np.histogram(all_velo, bins=200, density=True)
    centers = 0.5 * (bins[:-1] + bins[1:])
    mask = hist > 0
    ax.plot(centers[mask], hist[mask], "o", markersize=3, label=r'$\nu = {}$'.format(nu))

    # ax.set_yscale('log')
    ax.set_xlabel(XLABEL_MAP[parameter])
    ax.set_ylabel(r'PDF $\rho$')
    ax.set_title(TITLE_MAP[parameter])
    ax.legend()
    ax.set_yscale('log')
    plt.figtext(0.5, 0.01, r"Parameters used were $T = {}$, $\nu = {}$, $c_1 = {}$, and $c_2 = {}$".format(T, nu, c1, c2), ha='center',
                fontsize=9)
    plt.tight_layout(rect=[0, 0.05, 1, 1])
    if save:
        plt.savefig("/Users/rawdata/Downloads/data/pdf_edr_dns_c1_{}.pdf".format(c1))
    plt.show()

def graph_overlay(parameter="grad", semilog=False, save=False):
    # preparing DNS data
    FILENAME_MAP = {"inst": "u.npy", "grad": "velo_grad.npy", "edr": "ener_diss.npy"}
    TITLE_MAP = {"inst": "PDF of Agg Velo", "grad": "PDF of Velocity Gradient", "edr": "PDF of Energy Dissipation Rate"}
    XLABEL_MAP = {"inst": r'$\sum u_n$', "grad": r'$\sum u_n k_n$', "edr": r'$\nu \sum u_n^2 k_n^2$'}

    filename_dns = FILENAME_MAP[parameter]
    files_dns = glob.glob(
        f"/Users/rawdata/Downloads/data/dns/nu_{nu}_c1_{c1}_c2_{c2}_sigma_*_T_{T}*_dt_*_seed_*/{filename_dns}")

    print(f"number of dns files: {len(files_dns)}")

    all_velo = []

    for file in files_dns:
        data = np.load(
            file)  # data := one list of length 10,000 for each file in files, each value represents the total EDR (sum over shells) at time T for a given sample
        all_velo.append(data)  # appending energy dissipation rate at time t=T=10 for all paths

    all_velo = np.concatenate(all_velo)

    print(f"number of samples: {len(all_velo)}")

    plt.style.use('default')
    fig, ax = plt.subplots()

    # preparing instanton data
    filename = "inst_act.npy"
    files = glob.glob(
        f"/Users/rawdata/Downloads/data/{parameter}/seq_nu_{nu}_c1_{c1}_c2_{c2}_obs_*_date_*_*_*_*_*_*_branch_{branch}/{filename}")

    print(f"number of files: {len(files)}")

    obs_values = []
    data_values = []

    for file in files:
        # Extract obs value from path
        match = re.search(r'obs_([\d.eE+\-]+)', file)
        if match:
            obs = float(match.group(1))
            data = np.load(file)

            # if data > 200:
            #     print(obs)

            obs_values.append(obs)
            data_values.append(data)

    sorted_pairs = sorted(zip(obs_values, data_values), key=lambda x: x[0])
    obs_values, data_values = zip(*sorted_pairs)

    ################### GRAPHING ##############################

    # before plotting, print all your (obs, action) pairs
    for obs, action in sorted(zip(obs_values, data_values)):
        print(f"z = {obs:.2f},  I(z) = {action:.4f}")

    # DNS histogram: use all_velo directly, not normalized
    hist, bins = np.histogram(all_velo, bins=200, density=True)
    centers = 0.5 * (bins[:-1] + bins[1:])
    mask = hist > 0
    ax.plot(centers[mask], hist[mask], "o", markersize=5, label=r'$\nu = 10^{-2}$')

    # Instanton plot normalized to have area = 1 using the trapezoid rule
    y = np.exp(-np.array(data_values))
    normalization = np.trapz(y, obs_values)
    y_normalized = y / normalization  # no * std

    ax.plot(obs_values, y_normalized, marker="o", markersize=3, label='Instanton', color='black')

    obs_arr = np.array(obs_values)
    y_arr = np.array(y_normalized)
    inset_mask = (obs_arr >= centers[mask][0]) & (obs_arr <= centers[mask][-1])

    ax.set_xlabel(XLABEL_MAP[parameter])
    ax.set_title(TITLE_MAP[parameter])
    ax.legend()

    if(semilog):
        ax.set_ylabel(r'PDF $\rho$ (log scale)')
        ax.set_yscale('log')
        plt.figtext(0.5, 0.01,
                    r"Parameters used were $T={}$, $\nu = {}$, $c_1 = {}$, and $c_2 = {}$".format(T, nu, c1, c2),
                    ha='center',
                    fontsize=9)
        plt.tight_layout(rect=[0, 0.05, 1, 1])

        ax_inset = fig.add_axes([0.7, 0.45, 0.25, 0.25])  # [left, bottom, width, height]
        ax_inset.plot(obs_arr[inset_mask], y_arr[inset_mask], marker="o", markersize = 2, color='black')
        ax_inset.plot(centers[mask], hist[mask], marker="o", markersize=2, color='blue')
        ax_inset.set_xlabel(XLABEL_MAP[parameter], fontsize=5)
        ax_inset.set_ylabel(r'PDF $\rho$ (log scale)', fontsize=5)
        ax_inset.set_yscale('log')
        ax_inset.tick_params(labelsize=5)
        if save:
            plt.savefig(f"/Users/rawdata/Downloads/data/{parameter}/pdf_double_overlay_semilog_c1_{c1}.pdf")
    else:
        ax.set_ylabel(r'PDF $\rho$')
        plt.figtext(0.5, 0.01,
                    r"Parameters used were $T={}$, $\nu = {}$, $c_1 = {}$, and $c_2 = {}$".format(T, nu, c1, c2),
                    ha='center',
                    fontsize=9)
        plt.tight_layout(rect=[0, 0.05, 1, 1])
        if save:
            plt.savefig(f"/Users/rawdata/Downloads/data/{parameter}/pdf_double_overlay_c1_{c1}.pdf")
    plt.show()

###############################################################
# graphing pref files

def graph_evals(save=False):
    print(f"number of files in files_pref: {len(files_pref)}")
    print(f"files_pref filepath: {files_pref}")

    for file in files_pref:
        evals = np.load(file)
        print(f"Shape: {evals.shape}")

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
                    r"Parameters used were $T={}$, $\nu = {}$, $c_1 = {}$, $c_2 = {}$, and $z = {}$".format(T, nu, c1, c2, z),
                    ha='center',
                    fontsize=9)
        plt.tight_layout()
        if save:
            plt.savefig(f"/Users/rawdata/Downloads/data/inst/eval_spec_c1_{c1}_obs_{z}.pdf")
        plt.show()

###############################################################
# graphing plots from a specific individual run

# this is to recreate Figure 6 (RHS) from Goedert and Biferale.
# parameters: z = 82.34, nu = 0.01, c1 = 0.001, c2 = 0, T = 300

def graph_energy_spec():
    TT = "300"
    nuu = "0.01"
    cc1 = "0.001"
    cc2 = "0.0"
    zz = "82.34"
    print(r"parameters: $T={}$, $\nu = {}$, $c_1 = {}$, $c_2 = {}$, $z={}$".format(TT, nuu, cc1, cc2, zz))
    filess = glob.glob(f"/Users/rawdata/Downloads/data/inst/nu_{nuu}_c1_{cc1}_c2_{cc2}_obs_{zz}_date_*_*_*_*_*_*/inst_u.npy")
    print(f"Number of files for energy spec: {len(filess)}")
    for file in filess:
        data = np.load(file)
        plt.figure()
        shells = np.arange(data.shape[1])
        plot_times = [0, 5, 25, 50, 100, 200, 300]

        for s in plot_times:
            u_at_t = np.array(data[s*1000,:])
            u_sq = np.array(u_at_t ** 2)
            u_sq = np.where(u_sq > 0, u_sq, np.nan)
            plt.plot(shells, u_sq, label=f't = {s}', marker='*')

        plt.xlim([-0.5,6.5])
        plt.ylim([0.000000001,10000])
        plt.xlabel(r'shell $n$')
        plt.ylabel(r'$\log(u_n^2)$')
        plt.yscale('log')
        plt.legend(fontsize=8, ncol=2)
        # plt.savefig('/Users/rawdata/Downloads/data/inst/adjusted_inst_log_energy_spectrum.pdf', bbox_inches='tight')
        plt.show()

def graph_ener_spec(save=False):
    c_1 = 1.0
    c_2 = 0.0
    seed = 11
    nu_list = ["0.01", "0.001", "0.0001", "1e-05", "1e-06"]

    print(r"parameters: $c_1 = {}$, $c_2 = {}$, $seed={}$".format(c_1, c_2, seed))

    plt.figure()

    for nu in nu_list:
        filenames = f"/Users/rawdata/Downloads/data/dns/nu_{nu}_c1_{c_1}_c2_{c_2}_sigma_1.0_T_*_dt_*_seed_{seed}/mean_energy.npy"
        files = sorted(glob.glob(filenames))

        print(filenames)
        print(files)

        for file in files:
            data = np.load(file)
            print(data.shape)
            shells = np.arange(data.shape[0])

            plt.plot(shells, data, label=r'$\nu = {}$'.format(nu), marker='*')

    plt.ylim([2 ** -12, 2 ** 0])
    plt.yticks([2 ** -12, 2 ** -10, 2** -8, 2** -6, 2**-4, 2**-2, 2 ** 0])
    plt.xlabel(r'shell $n$')
    plt.ylabel(r'$E(k_n) = \log(u_n^2)$')
    plt.yscale('log', base=2)
    plt.legend(fontsize=8, ncol=2)
    if save:
        plt.savefig('/Users/rawdata/Downloads/data/dns/log_ener_spec_viscosity.pdf', bbox_inches='tight')
    plt.show()



def graph_adjoint():
    TT = 300
    nuu = "0.01"
    cc1 = "0.001"
    cc2 = "0.0"
    zz = "82.34"
    dt = 1e-3
    nt = int(TT / dt)

    dim = 17
    k = 2 ** np.linspace(0, dim - 1, dim)
    sigma = 1.
    chi_sqrt = sigma * k ** (-3.)

    filess = glob.glob(f"/Users/rawdata/Downloads/data/inst/nu_{nuu}_c1_{cc1}_c2_{cc2}_obs_{zz}_date_*_*_*_*_*_*/inst_eta.npy")
    print(f"Number of files: {len(filess)}")

    for file in filess:
        eta = np.load(file)
        p = eta / chi_sqrt[None, :]

        t_shifted = np.linspace(0, TT, nt + 1)

        step = int(1 / dt)  # 1000 indices per second
        indices = np.arange(0, nt + 1, step)

        plt.figure()
        for n in range(4):
            plt.plot(t_shifted[indices], p[indices, n], label=f'n = {n}')

        plt.xlabel(r'$t$')
        plt.ylabel(r'$p_n = \eta_z / \sqrt{\chi}$')
        plt.ylim(-1.5, 3.5)
        plt.legend(fontsize=6, ncol=3)
        plt.tight_layout()
        # plt.savefig(f"/Users/rawdata/Downloads/data/inst/adjoint_plot_c1_{cc1}_obs_{zz}.pdf")
        plt.show()

def graph_triple_overlay(semilog = False, save = False):
    # preparing DNS data
    filename_dns = "u.npy"
    files_dns = glob.glob(
        f"/Users/rawdata/Downloads/data/dns/nu_{nu}_c1_{c1}_c2_{c2}_sigma_*_T_{T}*_dt_*_seed_*/{filename_dns}")

    print(f"number of dns files: {len(files_dns)}")

    all_velo = []

    for file in files_dns:
        data = np.load(
            file)  # data := one list of length 10,000 for each file in files, each value represents the total EDR (sum over shells) at time T for a given sample
        all_velo.append(data)  # appending energy dissipation rate at time t=T=10 for all paths

    all_velo = np.concatenate(all_velo)

    print(f"number of samples: {len(all_velo)}")

    plt.style.use('default')
    fig, ax = plt.subplots()

    # preparing instanton data
    filename_inst = "inst_act.npy"
    files_inst = glob.glob(
        f"/Users/rawdata/Downloads/data/velo/seq_nu_{nu}_c1_{c1}_c2_{c2}_obs_*_date_*_*_*_*_*_*_branch_{branch}/{filename_inst}")

    print(f"number of instanton files: {len(files_inst)}")

    obs_values = []
    data_values = []

    for file in files_inst:
        # Extract obs value from path
        match = re.search(r'obs_([\d.eE+\-]+)', file)
        if match:
            obs = float(match.group(1))
            data = np.load(file)

            # if data > 200:
            #     print(obs)

            obs_values.append(obs)
            data_values.append(data)

    sorted_pairs = sorted(zip(obs_values, data_values), key=lambda x: x[0])
    obs_values, data_values = zip(*sorted_pairs)

    # preparing pref data
    dir_pref = "/Users/rawdata/Downloads/data/velo/prefactors_nu_{}_c1_{}_c2_{}_branch_{}/".format(nu, c1, c2, branch)
    files_pref = glob.glob(dir_pref + "prefactor_obs_*.npy")

    obs_values_pref = []
    data_values_pref = []

    for file in files_pref:
        # Extract obs value from path
        match = re.search(r'obs_([\d.eE+\-]+)\.npy', file)
        if match:
            obs = float(match.group(1))
            data = np.load(file)

            # if data > 200:
            #     print(obs)
            if obs != 0.0:
                obs_values_pref.append(obs)
                data_values_pref.append(data)
    sorted_pairs = sorted(zip(obs_values_pref, data_values_pref), key=lambda x: x[0])
    obs_values_pref, data_values_pref = zip(*sorted_pairs)

    # Match prefactor obs values with instanton obs values
    # Build a dict for fast lookup, then keep only obs that exist in both
    pref_dict = dict(zip(obs_values_pref, data_values_pref))

    obs_values_pref_matched = []
    pref_y_values = []
    exp_neg_I_values = []
    c_z_values = []
    for obs, action in zip(obs_values, data_values):
        if obs in pref_dict:
            c_z = float(pref_dict[obs])
            I_z = float(action)
            exp_neg_I = np.exp(-I_z)
            rho_z = c_z * exp_neg_I / np.sqrt(2 * np.pi)
            obs_values_pref_matched.append(obs)
            pref_y_values.append(rho_z)
            exp_neg_I_values.append(exp_neg_I)
            c_z_values.append(c_z)

    pref_y_values = np.array(pref_y_values)
    obs_values_pref_matched = np.array(obs_values_pref_matched)

    # Print e^{-I(z)}, c(z), and rho(z) for matched obs values
    print(f"\n{'z':>8}  {'e^{-I(z)}':>14}  {'c(z)':>14}  {'rho(z)':>14}")
    print("-" * 56)
    for obs, exp_neg_I, c_z, rho_z in zip(obs_values_pref_matched,
                                          exp_neg_I_values,
                                          c_z_values,
                                          pref_y_values):
        print(f"{obs:>8.2f}  {exp_neg_I:>14.6e}  {c_z:>14.6e}  {rho_z:>14.6e}")
    print()


    ################### GRAPHING ##############################

    # before plotting, print all your (obs, action) pairs
    for obs, action in sorted(zip(obs_values, data_values)):
        print(f"z = {obs:.2f},  I(z) = {action:.4f}")

    # DNS histogram: use all_velo directly, not normalized
    hist, bins = np.histogram(all_velo, bins=200, density=True)
    centers = 0.5 * (bins[:-1] + bins[1:])
    mask = hist > 0
    ax.plot(centers[mask], hist[mask], "o", markersize=5, label=r'DNS', color='black')

    # Instanton plot normalized to have area = 1 using the trapezoid rule
    y = np.exp(-np.array(data_values)) # e^{-I(z)}
    normalization = np.trapz(y, obs_values)
    y_normalized = y / normalization  # no * std

    ax.plot(obs_values, y_normalized, marker="o", markersize=3, label='Instanton', color='blue')

    # Plotting prefactor with instanton
    ax.plot(obs_values_pref_matched, pref_y_values, marker="s", markersize=3,
            label=r'Instanton $\times$ prefactor', color='red')

    obs_arr = np.array(obs_values)
    y_arr = np.array(y_normalized)
    inset_mask = (obs_arr >= centers[mask][0]) & (obs_arr <= centers[mask][-1])

    pref_inset_mask = ((obs_values_pref_matched >= centers[mask][0]) &
                       (obs_values_pref_matched <= centers[mask][-1]))

    ax.set_xlabel(r'$\sum u_n$')
    ax.set_ylim([1e-6, 1e2])
    ax.set_title('PDF of Agg Velo')
    ax.legend()

    if (semilog):
        ax.set_ylabel(r'PDF $\rho$ (log scale)')
        ax.set_yscale('log')
        plt.figtext(0.5, 0.01,
                    r"Parameters used were $T={}$, $\nu = {}$, $c_1 = {}$, and $c_2 = {}$".format(T, nu, c1, c2),
                    ha='center',
                    fontsize=9)
        plt.tight_layout(rect=[0, 0.05, 1, 1])

        # ax_inset = fig.add_axes([0.4, 0.26, 0.25, 0.25])  # [left, bottom, width, height]
        # ax_inset.plot(obs_arr[inset_mask], y_arr[inset_mask], marker="o", markersize=2, color='blue')
        # ax_inset.plot(centers[mask], hist[mask], marker="o", markersize=2, color='black')
        # ax_inset.plot(obs_values_pref_matched[pref_inset_mask],
        #               pref_y_values[pref_inset_mask],
        #               marker="s", markersize=2, color='red')
        # ax_inset.set_xlabel(r'$\sum u_n$', fontsize=5)
        # ax_inset.set_ylabel(r'PDF $\rho$ (log scale)', fontsize=5)
        # ax_inset.set_yscale('log')
        # ax_inset.tick_params(labelsize=5)
        if save:
            plt.savefig(f"/Users/rawdata/Downloads/data/velo/pdf_velo_overlay_semilog_c1_{c1}.pdf")
    else:
        ax.set_ylabel(r'PDF $\rho$')
        plt.figtext(0.5, 0.01,
                    r"Parameters used were $T={}$, $\nu = {}$, $c_1 = {}$, and $c_2 = {}$".format(T, nu, c1, c2),
                    ha='center',
                    fontsize=9)
        plt.tight_layout(rect=[0, 0.05, 1, 1])
        if save:
            plt.savefig(f"/Users/rawdata/Downloads/data/velo/pdf_velo_overlay_c1_{c1}.pdf")
    plt.show()

def graph_triple_edr_overlay(semilog = False, save = False):
    # preparing DNS data
    filename_dns = "ener_diss.npy"
    files_dns = glob.glob(
        f"/Users/rawdata/Downloads/data/dns/nu_{nu}_c1_{c1}_c2_{c2}_sigma_*_T_{T}*_dt_*_seed_*/{filename_dns}")

    print(f"number of dns files: {len(files_dns)}")

    all_velo = []

    for file in files_dns:
        data = np.load(
            file)  # data := one list of length 10,000 for each file in files, each value represents the total EDR (sum over shells) at time T for a given sample
        all_velo.append(data)  # appending energy dissipation rate at time t=T=10 for all paths

    all_velo = np.concatenate(all_velo)

    print(f"number of samples: {len(all_velo)}")

    plt.style.use('default')
    fig, ax = plt.subplots()

    # preparing instanton data
    filename_inst = "inst_act.npy"
    files_inst = glob.glob(
        f"/Users/rawdata/Downloads/data/edr/seq_nu_{nu}_c1_{c1}_c2_{c2}_obs_*_date_*_*_*_*_*_*_branch_{branch}/{filename_inst}")

    print(f"number of instanton files: {len(files_inst)}")

    obs_values = []
    data_values = []

    for file in files_inst:
        # Extract obs value from path
        match = re.search(r'obs_([\d.eE+\-]+)', file)
        if match:
            obs = float(match.group(1))
            data = np.load(file)

            # if data > 200:
            #     print(obs)

            obs_values.append(obs)
            data_values.append(data)

    sorted_pairs = sorted(zip(obs_values, data_values), key=lambda x: x[0])
    obs_values, data_values = zip(*sorted_pairs)

    # preparing pref data
    dir_pref = "/Users/rawdata/Downloads/data/edr/prefactors_nu_{}_c1_{}_c2_{}_branch_{}/".format(nu, c1, c2, branch)
    files_pref = glob.glob(dir_pref + "prefactor_obs_*.npy")

    obs_values_pref = []
    data_values_pref = []

    for file in files_pref:
        # Extract obs value from path
        match = re.search(r'obs_([\d.eE+\-]+)\.npy', file)
        if match:
            obs = float(match.group(1))
            data = np.load(file)

            # if data > 200:
            #     print(obs)
            if obs != 0.0:
                obs_values_pref.append(obs)
                data_values_pref.append(data)
    sorted_pairs = sorted(zip(obs_values_pref, data_values_pref), key=lambda x: x[0])
    obs_values_pref, data_values_pref = zip(*sorted_pairs)

    # Match prefactor obs values with instanton obs values
    # Build a dict for fast lookup, then keep only obs that exist in both
    pref_dict = dict(zip(obs_values_pref, data_values_pref))

    obs_values_pref_matched = []
    pref_y_values = []
    exp_neg_I_values = []
    c_z_values = []
    for obs, action in zip(obs_values, data_values):
        if obs in pref_dict:
            c_z = float(pref_dict[obs])
            I_z = float(action)
            exp_neg_I = np.exp(-I_z)
            rho_z = c_z * exp_neg_I / np.sqrt(2 * np.pi)
            obs_values_pref_matched.append(obs)
            pref_y_values.append(rho_z)
            exp_neg_I_values.append(exp_neg_I)
            c_z_values.append(c_z)

    pref_y_values = np.array(pref_y_values)
    obs_values_pref_matched = np.array(obs_values_pref_matched)

    # Print e^{-I(z)}, c(z), and rho(z) for matched obs values
    print(f"\n{'z':>8}  {'e^{-I(z)}':>14}  {'c(z)':>14}  {'rho(z)':>14}")
    print("-" * 56)
    for obs, exp_neg_I, c_z, rho_z in zip(obs_values_pref_matched,
                                          exp_neg_I_values,
                                          c_z_values,
                                          pref_y_values):
        print(f"{obs:>8.2f}  {exp_neg_I:>14.6e}  {c_z:>14.6e}  {rho_z:>14.6e}")
    print()


    ################### GRAPHING ##############################

    # before plotting, print all your (obs, action) pairs
    for obs, action in sorted(zip(obs_values, data_values)):
        print(f"z = {obs:.2f},  I(z) = {action:.4f}")

    # DNS histogram: use all_velo directly, not normalized
    hist, bins = np.histogram(all_velo, bins=200, density=True)
    centers = 0.5 * (bins[:-1] + bins[1:])
    mask = hist > 0
    ax.plot(centers[mask], hist[mask], "o", markersize=5, label=r'DNS', color='black')

    # Instanton plot normalized to have area = 1 using the trapezoid rule
    y = np.exp(-np.array(data_values)) # e^{-I(z)}
    normalization = np.trapz(y, obs_values)
    y_normalized = y / normalization  # no * std

    ax.plot(obs_values, y_normalized, marker="o", markersize=3, label='Instanton', color='blue')

    # Plotting prefactor with instanton
    ax.plot(obs_values_pref_matched, pref_y_values, marker="s", markersize=3,
            label=r'Instanton $\times$ prefactor', color='red')

    obs_arr = np.array(obs_values)
    y_arr = np.array(y_normalized)
    inset_mask = (obs_arr >= centers[mask][0]) & (obs_arr <= centers[mask][-1])

    pref_inset_mask = ((obs_values_pref_matched >= centers[mask][0]) &
                       (obs_values_pref_matched <= centers[mask][-1]))

    ax.set_xlabel(r'$\epsilon (T) = \nu \sum u_n^2 k_n^2$')
    ax.set_title('PDF of Energy Dissipation Rate')
    ax.legend()

    if (semilog):
        ax.set_ylabel(r'PDF $\rho$ (log scale)')
        ax.set_yscale('log')
        plt.figtext(0.5, 0.01,
                    r"Parameters used were $T={}$, $\nu = {}$, $c_1 = {}$, and $c_2 = {}$".format(T, nu, c1, c2),
                    ha='center',
                    fontsize=9)
        plt.tight_layout(rect=[0, 0.05, 1, 1])

        # ax_inset = fig.add_axes([0.72, 0.35, 0.25, 0.25])  # [left, bottom, width, height]
        # ax_inset.plot(obs_arr[inset_mask], y_arr[inset_mask], marker="o", markersize=2, color='blue')
        # ax_inset.plot(centers[mask], hist[mask], marker="o", markersize=2, color='black')
        # ax_inset.plot(obs_values_pref_matched[pref_inset_mask],
        #               pref_y_values[pref_inset_mask],
        #               marker="s", markersize=2, color='red')
        # ax_inset.set_xlabel(r'$\sum u_n$', fontsize=5)
        # ax_inset.set_ylabel(r'PDF $\rho$ (log scale)', fontsize=5)
        # ax_inset.set_yscale('log')
        # ax_inset.tick_params(labelsize=5)
        if save:
            plt.savefig(f"/Users/rawdata/Downloads/data/edr/pdf_edr_overlay_semilog_c1_{c1}.pdf")
    else:
        ax.set_ylabel(r'PDF $\rho$')
        plt.figtext(0.5, 0.01,
                    r"Parameters used were $T={}$, $\nu = {}$, $c_1 = {}$, and $c_2 = {}$".format(T, nu, c1, c2),
                    ha='center',
                    fontsize=9)
        plt.tight_layout(rect=[0, 0.05, 1, 1])
        if save:
            plt.savefig(f"/Users/rawdata/Downloads/data/edr/pdf_edr_overlay_c1_{c1}.pdf")
    plt.show()

def graph_triple_grad_overlay(semilog = False, save = False):
    # preparing DNS data
    filename_dns = "velo_grad.npy"
    files_dns = glob.glob(
        f"/Users/rawdata/Downloads/data/dns/nu_{nu}_c1_{c1}_c2_{c2}_sigma_*_T_{T}*_dt_*_seed_*/{filename_dns}")

    print(f"number of dns files: {len(files_dns)}")

    all_velo = []

    for file in files_dns:
        data = np.load(
            file)  # data := one list of length 10,000 for each file in files, each value represents the total EDR (sum over shells) at time T for a given sample
        all_velo.append(data)  # appending energy dissipation rate at time t=T=10 for all paths

    all_velo = np.concatenate(all_velo)

    print(f"number of samples: {len(all_velo)}")

    plt.style.use('default')
    fig, ax = plt.subplots()

    # preparing instanton data
    filename_inst = "inst_act.npy"
    files_inst = glob.glob(
        f"/Users/rawdata/Downloads/data/grad/seq_nu_{nu}_c1_{c1}_c2_{c2}_obs_*_date_*_*_*_*_*_*_branch_{branch}/{filename_inst}")

    print(f"number of instanton files: {len(files_inst)}")

    obs_values = []
    data_values = []

    for file in files_inst:
        # Extract obs value from path
        match = re.search(r'obs_([\d.eE+\-]+)', file)
        if match:
            obs = float(match.group(1))
            data = np.load(file)

            # if data > 200:
            #     print(obs)

            obs_values.append(obs)
            data_values.append(data)

    sorted_pairs = sorted(zip(obs_values, data_values), key=lambda x: x[0])
    obs_values, data_values = zip(*sorted_pairs)

    # preparing pref data
    dir_pref = "/Users/rawdata/Downloads/data/grad/prefactors_nu_{}_c1_{}_c2_{}_branch_{}/".format(nu, c1, c2, branch)
    files_pref = glob.glob(dir_pref + "prefactor_obs_*.npy")

    obs_values_pref = []
    data_values_pref = []

    for file in files_pref:
        # Extract obs value from path
        match = re.search(r'obs_([\d.eE+\-]+)\.npy', file)
        if match:
            obs = float(match.group(1))
            data = np.load(file)

            # if data > 200:
            #     print(obs)

            if obs != 0.0:
                obs_values_pref.append(obs)
                data_values_pref.append(data)
    sorted_pairs = sorted(zip(obs_values_pref, data_values_pref), key=lambda x: x[0])
    obs_values_pref, data_values_pref = zip(*sorted_pairs)

    # Match prefactor obs values with instanton obs values
    # Build a dict for fast lookup, then keep only obs that exist in both
    pref_dict = dict(zip(obs_values_pref, data_values_pref))

    obs_values_pref_matched = []
    pref_y_values = []
    exp_neg_I_values = []
    c_z_values = []
    for obs, action in zip(obs_values, data_values):
        if obs in pref_dict:
            c_z = float(pref_dict[obs])
            I_z = float(action)
            exp_neg_I = np.exp(-I_z)
            rho_z = c_z * exp_neg_I / np.sqrt(2 * np.pi)
            obs_values_pref_matched.append(obs)
            pref_y_values.append(rho_z)
            exp_neg_I_values.append(exp_neg_I)
            c_z_values.append(c_z)

    pref_y_values = np.array(pref_y_values)
    obs_values_pref_matched = np.array(obs_values_pref_matched)

    # Print e^{-I(z)}, c(z), and rho(z) for matched obs values
    print(f"\n{'z':>8}  {'e^{-I(z)}':>14}  {'c(z)':>14}  {'rho(z)':>14}")
    print("-" * 56)
    for obs, exp_neg_I, c_z, rho_z in zip(obs_values_pref_matched,
                                          exp_neg_I_values,
                                          c_z_values,
                                          pref_y_values):
        print(f"{obs:>8.2f}  {exp_neg_I:>14.6e}  {c_z:>14.6e}  {rho_z:>14.6e}")
    print()


    ################### GRAPHING ##############################

    # before plotting, print all your (obs, action) pairs
    for obs, action in sorted(zip(obs_values, data_values)):
        print(f"z = {obs:.2f},  I(z) = {action:.4f}")

    # DNS histogram: use all_velo directly, not normalized
    hist, bins = np.histogram(all_velo, bins=200, density=True)
    centers = 0.5 * (bins[:-1] + bins[1:])
    mask = hist > 0
    ax.plot(centers[mask], hist[mask], "o", markersize=5, label=r'DNS', color='black')

    # Instanton plot normalized to have area = 1 using the trapezoid rule
    y = np.exp(-np.array(data_values)) # e^{-I(z)}
    normalization = np.trapz(y, obs_values)
    y_normalized = y / normalization  # no * std

    ax.plot(obs_values, y_normalized, marker="o", markersize=3, label='Instanton', color='blue')

    # Plotting prefactor with instanton
    ax.plot(obs_values_pref_matched, pref_y_values, marker="s", markersize=3,
            label=r'Instanton $\times$ prefactor', color='red')

    obs_arr = np.array(obs_values)
    y_arr = np.array(y_normalized)
    inset_mask = (obs_arr >= centers[mask][0]) & (obs_arr <= centers[mask][-1])

    pref_inset_mask = ((obs_values_pref_matched >= centers[mask][0]) &
                       (obs_values_pref_matched <= centers[mask][-1]))

    ax.set_xlabel(r'$\epsilon (T) = \sum u_n k_n$')
    ax.set_title('PDF of Velocity Gradient')
    ax.legend(loc='lower right')

    if (semilog):
        ax.set_ylabel(r'PDF $\rho$ (log scale)')
        ax.set_yscale('log')
        plt.figtext(0.5, 0.01,
                    r"Parameters used were $T={}$, $\nu = {}$, $c_1 = {}$, and $c_2 = {}$".format(T, nu, c1, c2),
                    ha='center',
                    fontsize=9)
        plt.tight_layout(rect=[0, 0.05, 1, 1])

        ax_inset = fig.add_axes([0.35, 0.26, 0.25, 0.25])  # [left, bottom, width, height]
        ax_inset.plot(obs_arr[inset_mask], y_arr[inset_mask], marker="o", markersize=2, color='blue')
        ax_inset.plot(centers[mask], hist[mask], marker="o", markersize=2, color='black')
        ax_inset.plot(obs_values_pref_matched[pref_inset_mask],
                      pref_y_values[pref_inset_mask],
                      marker="s", markersize=2, color='red')
        ax_inset.set_xlabel(r'$\sum u_n$', fontsize=5)
        ax_inset.set_ylabel(r'PDF $\rho$ (log scale)', fontsize=5)
        ax_inset.set_yscale('log')
        ax_inset.tick_params(labelsize=5)
        if save:
            plt.savefig(f"/Users/rawdata/Downloads/data/grad/pdf_grad_overlay_semilog_c1_{c1}.pdf")
    else:
        ax.set_ylabel(r'PDF $\rho$')
        plt.figtext(0.5, 0.01,
                    r"Parameters used were $T={}$, $\nu = {}$, $c_1 = {}$, and $c_2 = {}$".format(T, nu, c1, c2),
                    ha='center',
                    fontsize=9)
        plt.tight_layout(rect=[0, 0.05, 1, 1])
        if save:
            plt.savefig(f"/Users/rawdata/Downloads/data/grad/pdf_grad_overlay_c1_{c1}.pdf")
    plt.show()

###############################################################
# graphing excess kurtosis

###############################################################
# graphing excess kurtosis

# c1 values to scan along the x-axis. Adjust to whatever you have data for.
KURT_C1_VALUES = [0.0001, 0.001, 0.01, 0.1, 1]


def compute_kurt_dns(c1_val, observable="velo"):
    """
    Compute excess kurtosis from DNS samples for a given c1.

    observable: "velo" -> uses u.npy (sum u_n at t=T)
                "edr"  -> uses ener_diss.npy (energy dissipation rate at t=T)

    Returns (F, n_samples) or (np.nan, 0) if no data found.
    """
    fname_map = {"velo":"u.npy", "edr":"ener_diss.npy", "grad":"velo_grad.npy"}

    fname = fname_map[observable]
    pattern = f"/Users/rawdata/Downloads/data/dns/nu_{nu}_c1_{c1_val}_c2_{c2}_sigma_*_T_*_dt_*_seed_*/{fname}"
    files_dns_local = glob.glob(pattern)

    print(f"  [DNS c1={c1_val}] found {len(files_dns_local)} file(s)")

    if not files_dns_local:
        print(f"  [DNS c1={c1_val}] pattern: {pattern}")
        return np.nan, 0

    samples = np.concatenate([np.load(f) for f in files_dns_local])

    mean = np.mean(samples)
    centered = samples - mean
    var = np.mean(centered ** 2)
    fourth = np.mean(centered ** 4)

    print(f"  [DNS c1={c1_val}] {len(samples)} samples, "
          f"mean={mean:.4e}, var={var:.4e}, fourth={fourth:.4e}, "
          f"sample range=[{samples.min():.3f}, {samples.max():.3f}]")

    if var <= 0:
        return np.nan, len(samples)

    F = fourth / (var ** 2) - 3.0
    return F, len(samples)

def compute_kurt_instanton(c1_val, observable="velo", with_prefactor=False):
    """
    Compute excess kurtosis from the instanton-predicted PDF for a given c1.

    observable: "velo" -> reads from data/inst/...
                "edr"  -> reads from data/edr/...
    with_prefactor: False -> use rho(z) ~ exp(-I(z)) / Z (instanton only)
                    True  -> use rho(z) = c(z) * exp(-I(z)) / sqrt(2 pi)

    Returns F or np.nan if not enough data found.
    """
    tag = f"INST+pref c1={c1_val}" if with_prefactor else f"INST c1={c1_val}"

    inst_pattern = (f"/Users/rawdata/Downloads/data/{observable}/seq_nu_{nu}_c1_{c1_val}_c2_{c2}"
            f"_obs_*_date_*_*_*_*_*_*_branch_*/inst_act.npy")
    print(f"inst_pattern: {inst_pattern}")
    inst_files = glob.glob(inst_pattern)

    pref_dir = (
        f"/Users/rawdata/Downloads/data/{observable}/"
        f"prefactors_nu_{nu}_c1_{c1_val}_c2_{c2}_branch_*/"
    )
    print(f"pref_dir: {pref_dir}")

    if not inst_files:
        print(f"  [{tag}] no instanton files; returning NaN")
        return np.nan

    obs_values = []
    data_values = []
    for f in inst_files:
        match = re.search(r'obs_([\d.eE+\-]+)', f)
        if match:
            # if float(match.group(1)) < -0.01 or float(match.group(1)) > 0.01:
            obs_values.append(float(match.group(1)))
            data_values.append(float(np.load(f)))

    if len(obs_values) < 4:
        print(f"  [{tag}] only {len(obs_values)} parsed obs values; returning NaN")
        return np.nan

    # If there are multiple branches per z value, pick the one with the
    # smallest action (the dominant branch)
    from collections import defaultdict
    by_obs = defaultdict(list)
    for obs, action in zip(obs_values, data_values):
        by_obs[obs].append(action)
    n_dupes = sum(1 for v in by_obs.values() if len(v) > 1)
    obs_values = sorted(by_obs.keys())
    data_values = [min(by_obs[o]) for o in obs_values]

    print(f"  [{tag}] {len(obs_values)} unique z values "
          f"({n_dupes} had duplicates, kept dominant branch), "
          f"z range [{min(obs_values):.2f}, {max(obs_values):.2f}], "
          f"action range [{min(data_values):.4f}, {max(data_values):.4f}]")

    obs_arr = np.array(obs_values)
    action_arr = np.array(data_values)

    import matplotlib.pyplot as plt

    plt.plot(obs_arr, action_arr, 'o-')
    plt.xlabel('z');
    plt.ylabel('I(z)')
    plt.title(f'Instanton action, c1={c1_val}')

    if with_prefactor:
        files_pref_local = glob.glob(pref_dir + "prefactor_obs_*.npy")
        obs_pref, data_pref = [], []
        for f in files_pref_local:
            match = re.search(r'obs_([\d.eE+\-]+)\.npy', f)
            if match:
                # if float(match.group(1)) < -0.01 or float(match.group(1)) > 0.01:
                obs_pref.append(float(match.group(1)))
                data_pref.append(float(np.load(f)))
        pref_dict = dict(zip(obs_pref, data_pref))

        print(f"  [{tag}] prefactor files: {len(files_pref_local)} "
              f"-> {len(pref_dict)} unique z values")

        z_list, rho_list = [], []
        for obs, action in zip(obs_arr, action_arr):
            if obs in pref_dict:
                c_z = pref_dict[obs]
                z_list.append(obs)
                rho_list.append(c_z * np.exp(-action) / np.sqrt(2 * np.pi))

        print(f"  [{tag}] {len(z_list)} z values matched between inst and pref")

        if len(z_list) < 4:
            if len(z_list) > 0:
                print(f"  [{tag}] matched z range [{min(z_list)}, {max(z_list)}]")
            sample_pref = sorted(pref_dict.keys())[:5]
            sample_inst = sorted(obs_arr)[:5]
            print(f"  [{tag}] first prefactor obs: {sample_pref}")
            print(f"  [{tag}] first instanton obs: {sample_inst}")
            return np.nan
        z_arr = np.array(z_list)
        rho_arr = np.array(rho_list)
    else:
        z_arr = obs_arr
        rho_arr = np.exp(-action_arr)

    # After building z_arr and rho_arr but before computing moments:

    # 1) Check tail decay
    plt.figure()
    plt.semilogy(z_arr, rho_arr, 'o-')
    plt.axhline(rho_arr.max() * 1e-4, color='r', linestyle='--', label='1e-4 of peak')
    plt.legend();
    plt.show()

    # 2) Check the integrand for the 4th moment
    mean_check = np.trapz(z_arr * rho_arr, z_arr) / np.trapz(rho_arr, z_arr)
    integrand_4 = (z_arr - mean_check) ** 4 * rho_arr
    plt.figure()
    plt.plot(z_arr, integrand_4, 'o-')
    plt.title('4th moment integrand — should go to 0 at both ends')
    plt.show()

    norm = np.trapz(rho_arr, z_arr)
    if norm <= 0:
        print(f"  [{tag}] norm={norm:.4e} non-positive; returning NaN")
        return np.nan

    mean = np.trapz(z_arr * rho_arr, z_arr) / norm
    centered = z_arr - mean
    var = np.trapz(centered ** 2 * rho_arr, z_arr) / norm
    fourth = np.trapz(centered ** 4 * rho_arr, z_arr) / norm

    print(f"  [{tag}] norm={norm:.4e}, mean={mean:.4e}, "
          f"var={var:.4e}, fourth={fourth:.4e}")

    if var <= 0:
        return np.nan

    F = np.abs(fourth / (var ** 2) - 3.0)
    return F

def graph_kurt(observable="velo", semilog_y=True, save=False):
    """
    Recreate the excess-kurtosis-vs-c1 plot.

    observable: "velo" or "edr" -- which observable to compute kurtosis for.
    """
    c1_arr = []
    F_dns_arr = []
    F_inst_arr = []
    F_pref_arr = []

    for c1_val in KURT_C1_VALUES:
        print(f"\n=== c1={c1_val} ===")
        F_dns, n_samples = compute_kurt_dns(c1_val, observable=observable)
        F_inst = compute_kurt_instanton(c1_val, observable=observable, with_prefactor=False)
        F_pref = compute_kurt_instanton(c1_val, observable=observable, with_prefactor=True)

        print(f"  -> F_dns={F_dns:.4f} (n={n_samples}), "
              f"F_inst={F_inst:.4f}, F_pref={F_pref:.4f}")

        c1_arr.append(float(c1_val))
        F_dns_arr.append(F_dns)
        F_inst_arr.append(F_inst)
        F_pref_arr.append(F_pref)

    c1_arr = np.array(c1_arr)
    F_dns_arr = np.array(F_dns_arr)
    F_inst_arr = np.array(F_inst_arr)
    F_pref_arr = np.array(F_pref_arr)

    fig, ax = plt.subplots()

    def _plot(target_ax, arr, **kw):
        m = np.isfinite(arr)
        target_ax.plot(c1_arr[m], arr[m], **kw)

    _plot(ax, F_dns_arr, marker='o', color='blue', label='DNS')
    _plot(ax, F_inst_arr, marker='*', color='red', label='Instanton')
    _plot(ax, F_pref_arr, marker='s', color='green',
          label=r'Instanton $\times$ prefactor')

    OBS_LABEL = {'velo': r'$\sum u_n$', 'edr':r'$\nu \sum k_n^2 u_n^2$', 'grad':r'$\sum k_n u_n$'}

    ax.set_xscale('log')
    if semilog_y:
        ax.set_yscale('log')
    ax.set_xlabel(r'$c_1$')
    ax.set_ylabel(r'$\log(F)$' if semilog_y else r'$F$')
    obs_label = OBS_LABEL[observable]
    ax.set_title(f'Excess kurtosis of {obs_label}')
    ax.legend()

    # Linear-y inset (matching the original figure's design)
    ax_inset = fig.add_axes([0.67, 0.23, 0.28, 0.28])  # [left, bottom, width, height]
    _plot(ax_inset, F_dns_arr, marker='o', markersize=4, color='blue')
    _plot(ax_inset, F_inst_arr, marker='*', markersize=4, color='red')
    _plot(ax_inset, F_pref_arr, marker='s', markersize=4, color='green')
    ax_inset.set_xscale('log')
    ax_inset.set_yscale('linear')
    ax_inset.set_xlim(8e-5, 2e-2)
    ax_inset.set_title('Linear plot', fontsize=8)
    ax_inset.tick_params(labelsize=6)

    plt.figtext(0.5, 0.01,
                r"Parameters used were $T={}$, $\nu = {}$, $c_2 = {}$".format(T, nu, c2),
                ha='center', fontsize=9)
    plt.tight_layout(rect=[0, 0.05, 1, 1])
    if save:
        plt.savefig(f"/Users/rawdata/Downloads/data/{observable}/kurt_vs_c1_{observable}.pdf")
    plt.show()



if __name__ == "__main__":
    # graph_inst_rate()
    # graph_dns(parameter="edr", save=False)
    # graph_inst_decay(True)
    # graph_overlay(parameter="edr", semilog=False, save=False)
    # graph_evals(False)
    # graph_ener_spec(save=True)
    # graph_adjoint()
    # graph_triple_overlay(True, True)
    # graph_triple_edr_overlay(True, False)
    # graph_triple_grad_overlay(True, False)
    graph_kurt(observable="grad", save=True, semilog_y=True)