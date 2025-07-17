import glob
import os

from pdb import set_trace

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LogNorm
import matplotlib as mpl

from scipy.spatial.distance import cdist

mpl.style.use(
    "/home/btcook/Desktop/github_repositories/chc_python/matplotlib_style_file"
)

import gala.coordinates as gc
import gala.dynamics as gd
import gala.potential as gp
from gala.units import galactic

import astropy.coordinates as coord
import astropy.units as u

_ = coord.galactocentric_frame_defaults.set("v4.0")

from scipy.stats import gaussian_kde, ks_2samp

import h5py

gala_pot = gp.MilkyWayPotential2022(units=galactic)

orbit_type = "circular_1"
pairing_type = "radial"

if "circular" in orbit_type:
    DATA_DIR = (
        f"/home/btcook/Desktop/krios_ii_paper/validation_tests/circular_orbit_subdir/"
    )
else:
    DATA_DIR = f"/home/btcook/Desktop/krios_ii_paper/validation_tests/eccentric_misaligned_orbit_subdir/"

KPC_TO_PC = 1000.0
KMS_TO_KPC_PER_MYR = 1 / 978.5


def get_key(filename):
    return float(filename.split("_")[-1].replace(".csv", ""))


def load_data_krios():
    if "circular" in orbit_type:
        subdir = f"krios_stream/{orbit_type}_orbit_test_exec_and_results"
    else:
        subdir = "krios_stream/orbit_0_test_exec_and_results"

    snapshot_filenames = f"{subdir}/snapshot_job_id_*.csv"
    scf_frame_filename = f"{subdir}/scf_frame_*.csv"

    all_files = sorted(
        glob.glob(DATA_DIR + snapshot_filenames),
        key=get_key,
    )
    df_scf_frame_info = pd.read_csv(
        glob.glob(DATA_DIR + scf_frame_filename)[0],
        header=0,
        sep=", ",
        engine="python",
    )

    file = all_files[-1]
    df_krios = pd.read_csv(file, sep=", ", engine="python")

    import pdb
    pdb.set_trace()

    time = float(file.split("_")[-1].replace(".csv", ""))

    print(time)

    idx_closest = np.argmin(np.abs(df_scf_frame_info["t"] - time))
    # frame = df_scf_frame_info.iloc[idx_closest].to_dict()

    frame = {
        "x": df_krios["x"].mean(),
        "y": df_krios["y"].mean(),
        "z": df_krios["z"].mean(),
        "vx": df_krios["vx"].mean(),
        "vy": df_krios["vy"].mean(),
        "vz": df_krios["vz"].mean(),
    }

    # kinetic energy
    df_krios["K"] = (
        0.5
        * df_krios["mass"]
        * (df_krios["vx"] ** 2 + df_krios["vy"] ** 2 + df_krios["vz"] ** 2)
    )

    pos = [
        df_krios["x"].values,
        df_krios["y"].values,
        df_krios["z"].values,
    ] * u.kpc

    # host potential energy
    psi_host = gala_pot.energy(pos)
    energy_unit = (u.km / u.s) ** (2.0)
    df_krios["U_c"] = np.multiply(df_krios["mass"], psi_host.to_value(energy_unit))

    # internal energy due to all the other particles
    newton_G = 4.301e-6  # kpc * (km/s)^2 / Msun

    # Positions in kpc
    positions = df_krios[["x", "y", "z"]].values  # shape (N, 3)
    masses = df_krios["mass"].values  # shape (N,)

    # Compute pairwise distances
    distances = cdist(positions, positions)  # shape (N, N)

    # Avoid divide-by-zero for i == j
    np.fill_diagonal(distances, np.inf)

    # Outer product of masses to get m_i * m_j for all pairs
    mass_matrix = np.outer(masses, masses)  # shape (N, N)

    # Compute potential energy matrix: G * m_i * m_j / |r_i - r_j|
    potential_matrix = -newton_G * mass_matrix / distances  # shape (N, N)

    # Internal energy for each particle: sum over j ≠ i, assign to DataFrame
    df_krios["U_int"] = np.sum(potential_matrix, axis=1)  # in (km/s)^2 units

    # specific total energy
    df_krios["E_total_N_body"] = df_krios["K"] + df_krios["U_int"] + df_krios["U_c"]

    # Coordinates of the cluster via the most bound particles
    vx_cluster, vy_cluster, vz_cluster = (
        frame["vx"],
        frame["vy"],
        frame["vz"],
    )

    # remove bound particles
    df_krios["E_wrt_cluster"] = df_krios["U_int"] + (
        0.5
        * df_krios["mass"]
        * (
            (df_krios["vx"] - vx_cluster) ** 2.0
            + (df_krios["vy"] - vy_cluster) ** 2.0
            + (df_krios["vz"] - vz_cluster) ** 2.0
        )
    )

    df_krios = df_krios[df_krios["E_wrt_cluster"] <= 0.0]

    info_dict = {
        "df": df_krios,
        "frame": frame,
    }

    return info_dict


def load_data_n_body(txt_filename, hf5_filename):
    """
    Loads the N-body simulation data, converts units based on length_unit and mass_unit,
    and computes additional quantities.
    all of the columns are in HU:
    cols: x, y, z, vx, vy, vz, U_int, U_c
    U_int is the N-body interaction energy of a star with all other N-1 stars
    U_c is the host potential energy
    E_tot = sum_i K_i + 0.5*sum_i U_{int,i} + sum_i U_{c,i} (edite

    <KeysViewHDF5 ['G_in_kpc_MSun_Myr', 'Msun_per_HU', 'Myr_per_HU', 'Npart', 'Trh_HU', 'data_E_wrt_cluster_HU',
    'data_Etot', 'data_Lx', 'data_Lx_wrt_cluster_HU', 'data_Ly', 'data_Ly_wrt_cluster_HU', 'data_Lz', 'data_Lz_wrt_cluster_HU',
    'data_lagrange_rad_01_pc', 'data_lagrange_rad_10_pc', 'data_lagrange_rad_20_pc', 'data_lagrange_rad_50_pc', 'data_lagrange_rad_90_pc',
    'data_time_HU', 'data_time_Myr', 'data_unbound_frac', 'kpc_per_HU']>

    Returns:
        pd.DataFrame: Dataframe with converted physical units and computed values.
    """
    df = pd.read_csv(
        txt_filename,
        sep="\t",
        names=["x", "y", "z", "vx", "vy", "vz", "m", "U_int", "U_c"],
        index_col=False,
    )

    # get unit conversion info
    with h5py.File(hf5_filename, "r") as f:
        mass_unit = f["Msun_per_HU"][()]
        length_unit = f["kpc_per_HU"][()]
        time_unit = f["Myr_per_HU"][()]

    # length unit: kpc, time_unit: Myr, need to convert velocity_unit to km/s
    velocity_unit = (length_unit / time_unit) / KMS_TO_KPC_PER_MYR
    energy_unit = mass_unit * velocity_unit**2  # Conversion factor for potential energy

    # Msun
    df["m"] *= mass_unit

    # kpc
    df["x"] *= length_unit
    df["y"] *= length_unit
    df["z"] *= length_unit

    # km/s
    df["vx"] *= velocity_unit
    df["vy"] *= velocity_unit
    df["vz"] *= velocity_unit

    # MSun * (km/s)^2
    df["U_int"] *= energy_unit
    df["U_c"] *= energy_unit

    # specific angular momentum
    df["L_z"] = df["m"] * (df["x"] * df["vy"] - df["y"] * df["vx"])

    # kinetic energy
    df["K"] = 0.5 * df["m"] * (df["vx"] ** 2 + df["vy"] ** 2 + df["vz"] ** 2)

    # specific total energy
    df["E_total"] = df["K"] + df["U_int"] + df["U_c"]

    # Coordinates of the cluster via the most bound particles
    cluster_particles = df[df["U_int"] < np.percentile(df["U_int"], 20.0)]
    vx_cluster, vy_cluster, vz_cluster = (
        cluster_particles["vx"].mean(),
        cluster_particles["vy"].mean(),
        cluster_particles["vz"].mean(),
    )

    # remove bound particles
    df["E_wrt_cluster"] = df["U_int"] + (
        0.5
        * df["m"]
        * (
            (df["vx"] - vx_cluster) ** 2.0
            + (df["vy"] - vy_cluster) ** 2.0
            + (df["vz"] - vz_cluster) ** 2.0
        )
    )

    df = df[df["E_wrt_cluster"] <= 0.0]

    frame = {
        "x": df["x"].mean(),
        "y": df["y"].mean(),
        "z": df["z"].mean(),
        "vx": df["vx"].mean(),
        "vy": df["vy"].mean(),
        "vz": df["vz"].mean(),
    }

    info_dict = {
        "df": df,
        "frame": frame,
    }

    return info_dict


if __name__ in "__main__":

    info_dict_krios = load_data_krios()

    # KRIOS is already in the right units
    df_krios = info_dict_krios["df"]

    nbody_file_str = orbit_type
    first_seed_txt_filename = (
        DATA_DIR + f"/n_body_stream/{nbody_file_str}_seed_0_snapshot.txt"
    )
    first_seed_hf5_filename = DATA_DIR + f"/n_body_stream/{nbody_file_str}_seed_0.hf5"
    n_body_info_dict = load_data_n_body(
        first_seed_txt_filename, first_seed_hf5_filename
    )
    df_n_body_one = n_body_info_dict["df"]

    second_seed_txt_filename = (
        DATA_DIR + f"/n_body_stream/{nbody_file_str}_seed_1_snapshot.txt"
    )
    second_seed_hf5_filename = DATA_DIR + f"/n_body_stream/{nbody_file_str}_seed_1.hf5"
    n_body_info_dict_second_seed = load_data_n_body(
        second_seed_txt_filename, second_seed_hf5_filename
    )
    df_n_body_two = n_body_info_dict_second_seed["df"]

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(6, 4))

    # Parameters for symlog binning
    linthresh = 10.0  # Linear threshold around zero
    num_log_bins = 20  # Number of log bins on each side of zero
    num_linear_bins = 50  # Number of bins in the linear region

    # Log-spaced bins on the negative and positive side
    log_bins_negative = -np.logspace(np.log10(linthresh), 2.5, num=num_log_bins)
    log_bins_positive = np.logspace(
        np.log10(linthresh), 6.0, num=int(num_log_bins * (6.0 / 2.5))
    )
    linear_bins = np.linspace(-linthresh, linthresh, num_linear_bins)

    # Combine into a single bin array
    bins = np.sort(np.concatenate([log_bins_negative, linear_bins, log_bins_positive]))

    # Plot histogram
    ax.hist(
        df_krios["E_wrt_cluster"],
        bins=bins,
        histtype="step",
        lw=0.5,
        alpha=0.8,
        label=r"KRIOS data, $N$-body energy calcluation",
    )
    ax.hist(
        df_krios["E_wrt_SCF"],
        bins=bins,
        histtype="step",
        lw=0.5,
        alpha=0.8,
        label="KRIOS data, KRIOS energy calculation",
    )
    ax.hist(
        df_n_body_one["E_wrt_cluster"],
        bins=bins,
        histtype="step",
        lw=0.5,
        alpha=0.8,
        label=r"$N$-body, first seed",
    )
    ax.hist(
        df_n_body_two["E_wrt_cluster"],
        bins=bins,
        histtype="step",
        lw=0.5,
        alpha=0.8,
        label=r"$N$-body, second seed",
    )

    # Axes labels and scales
    ax.set_xlabel("Energy wrt cluster [Msun * (km/s)**2]")
    ax.set_ylabel("Count")
    ax.set_xscale(
        "symlog", linthresh=10.0
    )  # linthresh sets the half-width of the linear region
    ax.legend(loc="upper right")

    # Save figure
    ax.set_xlim(-10**(2.5), 0.0)
    plt.savefig(f"energy_wrt_cluster_comp.png", dpi=500)
    plt.close()

    plt.figure()
    plt.scatter(
        df_krios["x"],
        df_krios["y"],
        s=0.1,
        edgecolors=None,
        linewidths=0,
        label="KRIOS",
    )
    plt.scatter(
        df_n_body_one["x"],
        df_n_body_one["y"],
        s=0.1,
        edgecolors=None,
        linewidths=0,
        label=r"$N$-body, first seed",
    )
    plt.scatter(
        df_n_body_two["x"],
        df_n_body_two["y"],
        s=0.1,
        edgecolors=None,
        linewidths=0,
        label=r"$N$-body, second seed",
    )
    plt.xlabel("x [kpc]")
    plt.ylabel("y [kpc]")
    plt.gca().set_aspect("equal")
    plt.legend(loc="best")

    plt.savefig("foo.png", dpi=500)
    plt.close()
