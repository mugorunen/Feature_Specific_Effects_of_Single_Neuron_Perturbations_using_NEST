import numpy as np
from scipy.ndimage import shift
import matplotlib.pyplot as plt

def Gabor_fields(sigma, gamma, psi, wavelength, theta, loc, ppd, vf_size):
    sz = vf_size * ppd  # size in pixels

    sigma = sigma * ppd
    wavelength = wavelength * ppd

    loc = loc * ppd

    sigma_x = sigma
    sigma_y = sigma / gamma

    x, y = np.meshgrid(np.arange(-sz//2, sz//2 + 1), np.arange(sz//2, -sz//2 - 1, -1))

    # Rotation
    x_theta = x * np.cos(theta) + y * np.sin(theta)
    y_theta = -x * np.sin(theta) + y * np.cos(theta)

    gb0 = np.exp(-0.5 * (x_theta**2 / sigma_x**2 + y_theta**2 / sigma_y**2)) * np.cos(2 * np.pi / wavelength * x_theta + psi)

    gb = shift(gb0, loc)

    return gb

    

def generate_RFs(N, NE, NI, sz, ppd, vf_size):
    locs = (np.random.rand(N, 2) - 0.5) * (sz / 20)
    po_exc = np.linspace(-np.pi/2, np.pi/2, NE)
    po_inh = np.linspace(-np.pi/2, np.pi/2, NI)
    po_all = np.concatenate((po_exc, po_inh))

    sigmas = 2.5 * np.ones(N)
    sigmas[NE:] = 2.5 * np.ones(NI)
    gammas = 0.5 * np.ones(N)

    #psis = np.random.rand(N) * 2 * np.pi
    psis = np.ones(N)*200

    sfs_exc = np.random.gamma(2, 1, NE) * 0.04
    sfs_inh = np.random.gamma(2, 1, NI) * 0.02
    sfs = np.concatenate((sfs_exc, sfs_inh))

    RFs = np.zeros((N, sz, sz))

    for i in range(NE):
        rf = Gabor_fields(sigmas[i], gammas[i], psis[i], 1 / sfs[i], po_all[i], locs[i], ppd, vf_size)
        RFs[i, :, :] = rf[:sz, :sz]

    for i in range(NE, N):
        rf = Gabor_fields(sigmas[i], gammas[i], psis[i], 1 / sfs[i], po_all[i], locs[i], ppd, vf_size)
        RFs[i, :, :] = rf[:sz, :sz]

    # Initialize an empty correlation matrix
    cc_rfs = np.zeros((N, N))

    # Compute the pairwise correlations
    for i in range(N):
        for j in range(i, N):
            corr = np.corrcoef(RFs[i].flatten(), RFs[j].flatten())[0, 1]
            cc_rfs[i, j] = corr
            cc_rfs[j, i] = corr  # Since correlation is symmetric

    #cc_rfs = np.corrcoef(RFs.reshape(N, -1), rowvar=False)

    
    dpo_all = np.zeros((N, N))
    for i in range(N):
        dpo_all[i, :] = np.mod(po_all[i] - po_all, np.pi)

    return RFs, cc_rfs, dpo_all

# Network parameters
ppd = 4  # Resolution (pixel per degree)
vf_size = 50  # Visual field size (in degrees)
sz = vf_size * ppd  # Visual size (in pixels)

N_stim_all = 1000
N_stim = 1

# Network and connectivity parameters
NE = 400
NI = 400
N = NE + NI

J0 = 1 / NE
JEE = J0
JEI = 2 * J0
JIE = -2 * J0
JII = -2 * J0

exp_pwr = 2

# Simulation parameters
dt = 1
tau = 10

# Neuron IDs to perturb
N_pert = 400

# Random sample to perturb
generate_random_set = True
if generate_random_set:
    nids = np.arange(1, NE + 1)
    np.random.shuffle(nids)
    nids = nids[:N_pert]

# A sample reference RF
nids_smpl = nids[0]

# Example usage:
RFs, cc_rfs, dpo_all = generate_RFs(N, NE, NI, sz, ppd, vf_size)

plt.figure()
plt.imshow(cc_rfs, cmap='viridis')
plt.colorbar()  # Add a colorbar to the plot
plt.title("Correlation Matrix (Subset)")
plt.xlabel("Columns")
plt.ylabel("Rows")


wEE = JEE * np.exp(exp_pwr * cc_rfs[:NE, :NE])
wEI = JEI * np.exp(exp_pwr * cc_rfs[:NE, NE:])
wIE = JIE * np.exp(exp_pwr * cc_rfs[NE:, :NE])
wII = JII * np.exp(exp_pwr * cc_rfs[NE:, NE:])

w = np.vstack((np.hstack((wEE, wEI)), np.hstack((wIE, wII))))

w = w + 0.1/20 * (np.random.rand(N, N) - 0.5)
np.fill_diagonal(w, 0)

zz = w[:NE, :]
zz[zz < 0] = 0
w[:NE, :] = zz

zz = w[NE:, :]
zz[zz > 0] = 0
w[NE:, :] = zz


plt.figure()
plt.imshow(w, cmap='viridis')
plt.colorbar()  # Add a colorbar to the plot
plt.title("Correlation Matrix (Subset)")
plt.xlabel("Columns")
plt.ylabel("Rows")






plt.show()

print('Done')