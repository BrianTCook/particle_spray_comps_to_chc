import pandas as pd
import numpy as np
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

density_unit = u.Msun / (u.pc) ** 3.0
energy_unit = u.Msun * (u.km / u.s) ** 2.0

rho_local = df["rho_local"] * density_unit
E_wrt_cluster = (df["K_wrt_whole_barycenter"] + df["U_nbody"]) * energy_unit

varepsilon = 0.001 * (24.5621 * u.pc)
n_local = rho_local / (10.0 * u.Msun)

print((n_local ** (-1 / 3.0)) / varepsilon)

plt.figure()
plt.scatter(
    (n_local ** (-1 / 3.0)) / varepsilon,
    E_wrt_cluster,
    s=0.5,
    alpha=1.0,
    edgecolors=None,
    linewidths=0,
)
plt.xlabel(r"$n_{\rm local}^{-1/3} / \varepsilon$", fontsize=8)
plt.ylabel(r"$K_{\rm wrt \, whole \, center} + U_{\rm nbody}$ [$M_{\odot}$ (km/s)$^{2}$]", fontsize=8)
plt.gca().set_xscale("log")
plt.gca().set_yscale("symlog", linthresh=10)
plt.savefig("foo.png", dpi=500)
