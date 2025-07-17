import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LogNorm
import matplotlib as mpl

import astropy.units as u

mpl.style.use(
    "/home/btcook/Desktop/github_repositories/chc_python/matplotlib_style_file"
)

DATA_DIR = "/home/btcook/Desktop/krios_ii_paper/validation_tests/circular_orbit_subdir/krios_stream/circular_1_orbit_test_exec_and_results/"
df = pd.read_csv(
    DATA_DIR + "snapshot_job_id_8370693_time_2477.49787215.csv",
    sep=", ",
    engine="python",
)
E_wrt_cluster = (df["K_wrt_whole_barycenter"] + df["U_nbody"])
df = df[E_wrt_cluster < 0.0]

x, y, z, vx, vy, vz = df["x"], df["y"], df["z"], df["vx"], df["vy"], df["vz"]

x -= df["x"].mean()
y -= df["y"].mean()
z -= df["z"].mean()
vx -= df["vx"].mean()
vy -= df["vy"].mean()
vz -= df["vz"].mean()

# Particle positions
pos = np.vstack((x, y, z)).T
r = np.linalg.norm(pos, axis=1)  # Euclidean distance from center

v = np.sqrt(vx**2 + vy**2 + vz**2)

# Choose bins
r_bins = np.logspace(np.log10(r.min()), np.log10(r.max()), num=50)
r_bin_centers = 0.5 * (r_bins[:-1] + r_bins[1:])

# Digitize to assign each particle to a bin
bin_indices = np.digitize(r, r_bins)

# For each particle, assign sigma^2 from its radial bin
sigma_squared_per_particle = np.empty_like(r)
for i in range(1, len(r_bins)):
    in_bin = bin_indices == i
    if np.count_nonzero(in_bin) > 1:
        sigma_sq = np.var(v[in_bin])  # Or use component-based version
        sigma_squared_per_particle[in_bin] = sigma_sq
    else:
        sigma_squared_per_particle[in_bin] = np.nan

pos = np.vstack((x, y, z)).T
v_all = np.vstack((vx, vy, vz)).T
nn = NearestNeighbors(n_neighbors=30).fit(pos)
distances, neighbors = nn.kneighbors(pos)

sigma_squared_per_particle = np.zeros(len(pos))
for i, idxs in enumerate(neighbors):
    local_velocities = v_all[idxs]
    sigma_sq = np.mean(np.var(local_velocities, axis=0))  # average over vx, vy, vz
    sigma_squared_per_particle[i] = sigma_sq

G_pc_km2_s2_Msun = 4.301e-3  # [pc (km/s)^2 / Msun]
m_particle = 10.0  # MSun
numerator = G_pc_km2_s2_Msun * (m_particle)
denominator = sigma_squared_per_particle
quantity = numerator / denominator

plt.scatter(
    r,
    quantity,
    s=0.5,
    alpha=1.0,
    edgecolors=None,
    linewidths=0,
)
plt.xlabel(r"$r$ [pc]", fontsize=16)
plt.ylabel(r"$\frac{Gm}{\sigma^{2}}$ [pc]", fontsize=16)
plt.yscale("log")
plt.xscale("log")
plt.savefig("b90_plot.png", dpi=500)
