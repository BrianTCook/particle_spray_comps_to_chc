import glob
import os

from pdb import set_trace

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LogNorm

import gala.coordinates as gc
import gala.dynamics as gd
import gala.potential as gp

import astropy.coordinates as coord
import astropy.units as u

_ = coord.galactocentric_frame_defaults.set("v4.0")

from scipy.stats import gaussian_kde, ks_2samp

import h5py

plt.rcParams["font.family"] = "serif"
plt.rcParams["mathtext.fontset"] = "dejavuserif"

plt.rcParams["font.family"] = "serif"
plt.rcParams["mathtext.fontset"] = "dejavuserif"

orbit_type = "eccentric_misaligned"
pairing_type = "radial"

DATA_DIR = f"/home/btcook/Desktop/krios_ii_paper/{orbit_type}_orbit_subdir/"
KPC_TO_PC = 1000.0
KMS_TO_KPC_PER_MYR = 1 / 978.5

import agama  # to calculate action

agama.setUnits(length=1, velocity=1, mass=1)  # working units: 1 Msun, 1 kpc, 1 km/s

actFinder = agama.ActionFinder(
    agama.Potential(
        "/home/btcook/Desktop/github_repositories/CHC/agama/data/MWPotential2014.ini"
    )
)


def get_action_diffs(df_bound, df_unbound):
    # Assumes df_bound and df_unbound are DataFrames with columns ['x','y','z','vx','vy','vz'] in AGAMA units
    posvel_bound = df_bound[["x", "y", "z", "vx", "vy", "vz"]].to_numpy()
    posvel_unbound = df_unbound[["x", "y", "z", "vx", "vy", "vz"]].to_numpy()

    actions_bound = actFinder(posvel_bound)
    actions_unbound = actFinder(posvel_unbound)

    mean_bound_action = np.mean(actions_bound, axis=0)

    # shape (N_unbound, 3)
    return actions_unbound - mean_bound_action


def get_key(filename):
    return float(filename.split("_")[-1].replace(".csv", ""))


def load_data_krios():
    all_files = sorted(
        glob.glob(
            DATA_DIR
            + f"krios_stream/{orbit_type}_orbit_{pairing_type}_pairing_exec_and_results/snapshot_job_id_*.csv"
        ),
        key=get_key,
    )
    last_snapshot = all_files[-1]
    df_krios = pd.read_csv(last_snapshot, sep=", ")
    df_krios_bound = df_krios[df_krios["E_wrt_cluster"] < 0.0]
    df_krios_unbound = df_krios[df_krios["E_wrt_cluster"] >= 0.0]

    df_conserved_quantities_info = pd.read_csv(
        glob.glob(
            DATA_DIR
            + f"krios_stream/{orbit_type}_orbit_{pairing_type}_pairing_exec_and_results/conserved_quantities_*.csv"
        )[0],
        header=0,
        sep=", ",
        engine="python",
    )

    t_scf = df_conserved_quantities_info["t"]
    m_init_scf = df_conserved_quantities_info["M_tot_SCF"].values[0]
    unbound_scf = 1.0 - df_conserved_quantities_info["M_tot_SCF"] / m_init_scf

    info_dict = {
        "df_bound": df_krios_bound,
        "df_unbound": df_krios_unbound,
        "times": t_scf,
        "unbound_frac": unbound_scf,
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
        # print(f.keys())
        mass_unit = f["Msun_per_HU"][()]
        length_unit = f["kpc_per_HU"][()]
        time_unit = f["Myr_per_HU"][()]

        data_time = f["data_time_Myr"][:]
        data_unbound_frac = f["data_unbound_frac"][:]

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
    df["L_z"] = df["x"] * df["vy"] - df["y"] * df["vx"]

    # kinetic energy
    df["K"] = 0.5 * df["m"] * (df["vx"] ** 2 + df["vy"] ** 2 + df["vz"] ** 2)

    # specific total energy
    df["E_total"] = (df["K"] + df["U_int"] + df["U_c"]) / df["m"]

    # Coordinates of the cluster via the most bound particles
    cluster_particles = df[df["U_int"] < np.percentile(df["U_int"], 10.0)]
    vx_cluster, vy_cluster, vz_cluster = (
        cluster_particles["vx"].mean(),
        cluster_particles["vy"].mean(),
        cluster_particles["vz"].mean(),
    )

    # remove bound particles
    df["E_wrt_cluster"] = df["U_int"] + 0.5 * (
        (df["vx"] - vx_cluster) ** 2.0
        + (df["vy"] - vy_cluster) ** 2.0
        + (df["vz"] - vz_cluster) ** 2.0
    )

    df_bound = df[df["E_wrt_cluster"] <= 0.0]
    df_unbound = df[df["E_wrt_cluster"] > 0.0]

    print(len(df_bound), len(df_unbound))

    info_dict = {
        "df_bound": df_bound,
        "df_unbound": df_unbound,
        "times": data_time,
        "unbound_frac": data_unbound_frac,
    }

    return info_dict


def get_orbital_poles(df):
    """
    compute the unit vector that points in the direction of r x v for each particle in df
    this is orthogonal to the instantaneous orbital plane and defines the pole of the orbit
    needed for constructing a great circle coordinate system
    """

    x, y, z = df["x"].values, df["y"].values, df["z"].values
    vx, vy, vz = df["vx"].values, df["vy"].values, df["vz"].values

    pos = np.concatenate(
        (
            np.reshape(x, (x.size, 1)),
            np.reshape(y, (y.size, 1)),
            np.reshape(z, (z.size, 1)),
        ),
        axis=1,
    )
    vel = np.concatenate(
        (
            np.reshape(vx, (vx.size, 1)),
            np.reshape(vy, (vy.size, 1)),
            np.reshape(vz, (vz.size, 1)),
        ),
        axis=1,
    )

    # |r x v|
    r_cross_v = np.cross(pos, vel, axis=1)
    r_cross_v_norm = np.linalg.norm(r_cross_v, axis=1)

    # return normalized orbital pole of all particles
    return r_cross_v.T / r_cross_v_norm


def get_phi_info(df, ref_point_vec):
    """
    Compute phi_1 and phi_2 for a given dataframe containing MW phase space (x, y, z, vx, vy, vz),
    dynamically determining the best great circle fit from the dataset

    Args:
        df (pd.DataFrame): DataFrame with columns ["x", "y", "z", "vx", "vy", "vz"]
        ref_point_vec (tuple): (x, y, z) coordinates of the progenitor in kpc.
    """
    orbital_poles = get_orbital_poles(df)
    pole_vec = np.mean(orbital_poles, axis=1)
    pole_vec /= np.linalg.norm(pole_vec)  # normalize to unit vector

    # x direction: vector from origin to progenitor (ref point in position space)
    x_vec = np.array(ref_point_vec[:3])
    x_vec /= np.linalg.norm(x_vec)

    # z direction: use the orbital pole
    z_vec = pole_vec

    # y direction: complete right-handed system
    y_vec = np.cross(z_vec, x_vec)
    y_vec /= np.linalg.norm(y_vec)

    # Recompute x_vec to be orthogonal to z and y
    x_vec = np.cross(y_vec, z_vec)
    x_vec /= np.linalg.norm(x_vec)

    xnew = coord.representation.CartesianRepresentation(*(x_vec * u.kpc))
    znew = coord.representation.CartesianRepresentation(*(z_vec * u.kpc))

    frame = gc.GreatCircleICRSFrame.from_xyz(xnew=xnew, znew=znew)

    # Positions (in kpc)
    positions = coord.representation.CartesianRepresentation(
        df["x"].values * u.kpc, df["y"].values * u.kpc, df["z"].values * u.kpc
    )

    # Velocities (in km/s)
    velocities = coord.representation.CartesianDifferential(
        df["vx"].values * u.km / u.s,
        df["vy"].values * u.km / u.s,
        df["vz"].values * u.km / u.s,
    )

    # Combine position and velocity
    positions_with_velocities = positions.with_differentials(velocities)

    # Construct SkyCoord from this
    skycoord = coord.SkyCoord(positions_with_velocities)
    coord_new = skycoord.transform_to(frame)

    mu_phi1 = coord_new.pm_phi1_cosphi2 / np.cos(coord_new.phi2)
    mu_phi1 = mu_phi1.to(u.mas / u.yr).value
    mu_phi2 = coord_new.pm_phi2.to(u.mas / u.yr).value

    phi_info_dict = {
        "phi1": coord_new.phi1.to(u.deg).value,
        "phi2": coord_new.phi2.to(u.deg).value,
        "mu_phi1": mu_phi1,
        "mu_phi2": mu_phi2,
    }

    return phi_info_dict


def plot_streams(stream_info_dict):
    """
    Create a 2-row, 2-column plot:
    - Scatter plot for (Lz, Etot)
    - Contour plot for (vz, z)
    - Histogram for phi_1
    - Mass-loss rate for KRIOS, N-body
    """
    fig, axs = plt.subplots(2, 2, figsize=(12, 12))
    colors = {
        "KRIOS": "C0",
        "Particle Spray (Chen et al. (2025))": "C1",
        r"Direct $N$-body (First Seed)": "C2",
        r"Direct $N$-body (Second Seed)": "C3",
    }

    # --- Panel 1: Scatter Plot (Lz vs. Etot) ---
    ax1 = axs[0, 0]

    pot = gp.BovyMWPotential2014()

    for label, data in stream_info_dict.items():
        # Lz, Etot = data["L_z"], data["E_wrt_host"]
        df = data["df"]

        # Extract positions and velocities with units
        pos = df[["x", "y", "z"]].values * u.kpc
        vel = df[["vx", "vy", "vz"]].values * u.km / u.s

        # Create PhaseSpacePosition object for all rows
        phase_space = gd.PhaseSpacePosition(pos=pos.T, vel=vel.T)

        # Compute energies and Lz for all particles
        E_wrt_host = pot.total_energy(phase_space.pos.xyz, phase_space.vel.d_xyz).to(
            (u.km / u.s) ** 2.0
        )  # shape (N,)
        Lz = phase_space.angular_momentum()[2]  # shape (N,)

        ax1.scatter(
            Lz / 1e3,
            E_wrt_host / 1e5,
            c=colors[label],
            s=0.5,
            alpha=1.0,
            label=label,
            edgecolors=None,
            linewidths=0,
        )

    ax1.set_xlabel(r"$L_{z'} \, [10^{3} \, {\rm kpc} \,  {\rm km/s}]$", fontsize=18)
    ax1.set_ylabel(
        r"$E_{\mathrm{wrt \, host}} [10^{5} \, ({\rm km/s})^{2}]$", fontsize=18
    )
    ax1.legend(loc="upper left", markerscale=10, fontsize=12)
    ax1.grid(linewidth=0.5, linestyle="--", c="k", alpha=0.5)

    # --- Panel 2: Contour Plot (vz vs. z) ---
    ax2 = axs[0, 1]

    for label, data in stream_info_dict.items():
        Jr, Jz = data["Jr"], data["Jz"]

        # Compute mean and standard deviation
        Jr_mean, Jr_std = np.mean(Jr), np.std(Jr)
        Jz_mean, Jz_std = np.mean(Jz), np.std(Jz)

        # Plot mean with 1-sigma error bars
        ax2.errorbar(
            Jr_mean,
            Jz_mean,
            xerr=Jr_std,
            yerr=Jz_std,
            fmt="o",
            color=colors[label],
            label=label,
            markersize=4,
            capsize=2,
            alpha=1.0,
        )

    ax2.set_xlabel(r"$\Delta J_{r'}$ [kpc km/s]", fontsize=18)
    ax2.set_ylabel(r"$\Delta J_{z'}$ [kpc km/s]", fontsize=18)
    ax2.grid(linewidth=0.5, linestyle="--", c="k", alpha=0.5)

    # --- Panel 3: f(phi_1) = \int d \phi_2 rho(\phi_1, \phi_2) ---
    # Global bounds for consistency across panels
    all_phi1 = np.concatenate(
        [
            stream_info_dict[label]["phi_info"]["phi1"]
            for label in stream_info_dict.keys()
        ]
    )
    all_phi2 = np.concatenate(
        [
            stream_info_dict[label]["phi_info"]["phi2"]
            for label in stream_info_dict.keys()
        ]
    )

    lower, upper = 2.275, 97.725
    min_phi1, max_phi1 = np.percentile(all_phi1, [lower, upper])
    min_phi2, max_phi2 = np.percentile(all_phi2, [lower, upper])

    n_third_panel_gridpoints = 200
    phi1_values = np.linspace(min_phi1, max_phi1, n_third_panel_gridpoints)
    phi2_values = np.linspace(min_phi2, max_phi2, n_third_panel_gridpoints)

    xi, yi = np.meshgrid(phi1_values, phi2_values)

    label_phi_dict = {}

    ax3 = axs[1, 0]
    for label, data in stream_info_dict.items():
        data = stream_info_dict[label]
        phi1, phi2 = data["phi_info"]["phi1"], data["phi_info"]["phi2"]

        kde = gaussian_kde(np.vstack([phi1, phi2]))
        zi = kde(np.vstack([xi.ravel(), yi.ravel()])).reshape(xi.shape)
        zi /= np.sum(zi)

        phi1_condensed = np.sum(zi, axis=0)
        label_phi_dict[label] = phi1_condensed
        ax3.plot(phi1_values, phi1_condensed, label=label)

    # Compute KS test for each pair of phi1 distributions
    labels = list(label_phi_dict.keys())
    for i in range(len(labels)):
        for j in range(i + 1, len(labels)):
            label1, label2 = labels[i], labels[j]
            data1, data2 = label_phi_dict[label1], label_phi_dict[label2]

            ks_stat, p_value = ks_2samp(data1, data2)

            print(f"KS test between {label1} and {label2}:")
            print(f"    KS statistic = {ks_stat:.4f}, p-value = {p_value:.4e}")

    ax3.set_xlabel(r"$\phi_1$ [deg]", fontsize=18)
    ax3.set_ylabel(
        r"$f(\phi_{1}) = \int {\rm d}\phi_{2} \, \rho(\phi_{1},\phi_{2})$",
        fontsize=18,
    )
    ax3.grid(linewidth=0.5, linestyle="--", c="k", alpha=0.5)

    ax4 = axs[1, 1]
    for label, data in stream_info_dict.items():
        if label != "Particle Spray (Chen et al. (2025))":
            ax4.plot(
                data["info_dict"]["times"],
                data["info_dict"]["unbound_frac"],
                label=label,
                linewidth=1,
                color=colors[label],
            )

    ax4.set_xlabel(r"Time [Myr]", fontsize=18)
    ax4.set_ylabel(r"$M_{\rm unbound}/M_{\rm total}$", fontsize=18)
    ax4.set_aspect("auto")
    ax4.set_xlim(0.0, 3000.0)
    ax4.set_ylim(0.0, 0.5)
    ax4.grid(linewidth=0.5, linestyle="--", alpha=0.5)

    plt.tight_layout()
    plt.savefig(f"{orbit_type}_orbit_comps.png", bbox_inches="tight", dpi=400)


def main():
    krios_info_dict = load_data_krios()
    df_krios_bound, df_krios_unbound = (
        krios_info_dict["df_bound"],
        krios_info_dict["df_unbound"],
    )
    ref_point_vec_krios = np.array(
        [
            df_krios_bound["x"].mean(),
            df_krios_bound["y"].mean(),
            df_krios_bound["z"].mean(),
        ]
    )
    action_diffs = get_action_diffs(df_krios_bound, df_krios_unbound)
    df_krios_unbound["Jr"], df_krios_unbound["Jz"] = (
        action_diffs[:, 0],
        action_diffs[:, 1],
    )

    df_chen = pd.read_csv(DATA_DIR + f"chen_stream/chen_{orbit_type}_orbit_stream.csv")
    df_chen_bound, df_chen_unbound = (
        df_chen[df_chen["source"] == "prog"],
        df_chen[df_chen["source"] == "stream"],
    )
    action_diffs = get_action_diffs(df_chen_bound, df_chen_unbound)
    df_chen_unbound["Jr"], df_chen_unbound["Jz"] = (
        action_diffs[:, 0],
        action_diffs[:, 1],
    )

    ref_point_vec_chen = np.array(
        [
            df_chen_bound["x"].mean(),
            df_chen_bound["y"].mean(),
            df_chen_bound["z"].mean(),
        ]
    )

    first_seed_txt_filename = DATA_DIR + "/n_body_stream/last_snapshot_first_seed.txt"
    first_seed_hf5_filename = DATA_DIR + "/n_body_stream/first_seed.hf5"
    n_body_info_dict = load_data_n_body(
        first_seed_txt_filename, first_seed_hf5_filename
    )
    df_n_body_bound, df_n_body_unbound = (
        n_body_info_dict["df_bound"],
        n_body_info_dict["df_unbound"],
    )
    action_diffs = get_action_diffs(df_n_body_bound, df_n_body_unbound)
    df_n_body_unbound["Jr"], df_n_body_unbound["Jz"] = (
        action_diffs[:, 0],
        action_diffs[:, 1],
    )

    ref_point_vec_n_body = np.array(
        [
            df_n_body_bound["x"].mean(),
            df_n_body_bound["y"].mean(),
            df_n_body_bound["z"].mean(),
        ]
    )

    second_seed_txt_filename = DATA_DIR + "/n_body_stream/last_snapshot_second_seed.txt"
    second_seed_hf5_filename = DATA_DIR + "/n_body_stream/second_seed.hf5"
    n_body_info_dict_second_seed = load_data_n_body(
        second_seed_txt_filename, second_seed_hf5_filename
    )
    df_n_body_bound_second_seed = n_body_info_dict_second_seed["df_bound"]
    df_n_body_unbound_second_seed = n_body_info_dict_second_seed["df_unbound"]
    action_diffs = get_action_diffs(
        df_n_body_bound_second_seed, df_n_body_unbound_second_seed
    )
    df_n_body_unbound_second_seed["Jr"], df_n_body_unbound_second_seed["Jz"] = (
        action_diffs[:, 0],
        action_diffs[:, 1],
    )
    ref_point_vec_n_body_second_seed = np.array(
        [
            df_n_body_bound_second_seed["x"].mean(),
            df_n_body_bound_second_seed["y"].mean(),
            df_n_body_bound_second_seed["z"].mean(),
        ]
    )

    m_star = 10.0  # MSun

    # KRIOS IOMs have to be convert to specific IOMs
    stream_info_dict = {
        "KRIOS": {
            "df": df_krios_unbound,
            "L_z": df_krios_unbound["L_z"] / (1e3 * m_star),
            "E_wrt_host": df_krios_unbound["E_wrt_host"] / (1e5 * m_star),
            "phi_info": get_phi_info(df_krios_unbound, ref_point_vec_krios),
            "Jr": df_krios_unbound["Jr"],
            "Jz": df_krios_unbound["Jz"],
            "info_dict": krios_info_dict,
        },
        "Particle Spray (Chen et al. (2025))": {
            "df": df_chen_unbound,
            "L_z": df_chen_unbound["L_z"] / 1e3,
            "E_wrt_host": df_chen_unbound["E_wrt_host"] / 1e5,
            "phi_info": get_phi_info(df_chen_unbound, ref_point_vec_chen),
            "Jr": df_chen_unbound["Jr"],
            "Jz": df_chen_unbound["Jz"],
        },
        r"Direct $N$-body (First Seed)": {
            "df": df_n_body_unbound,
            "L_z": df_n_body_unbound["L_z"] / 1e3,
            "E_wrt_host": df_n_body_unbound["E_total"] / (1e5 * m_star),
            "phi_info": get_phi_info(df_n_body_unbound, ref_point_vec_n_body),
            "Jr": df_n_body_unbound["Jr"],
            "Jz": df_n_body_unbound["Jz"],
            "info_dict": n_body_info_dict,
        },
        r"Direct $N$-body (Second Seed)": {
            "df": df_n_body_unbound_second_seed,
            "L_z": df_n_body_unbound_second_seed["L_z"] / 1e3,
            "E_wrt_host": df_n_body_unbound_second_seed["E_total"] / (1e5 * m_star),
            "phi_info": get_phi_info(
                df_n_body_unbound_second_seed, ref_point_vec_n_body_second_seed
            ),
            "Jr": df_n_body_unbound_second_seed["Jr"],
            "Jz": df_n_body_unbound_second_seed["Jz"],
            "info_dict": n_body_info_dict_second_seed,
        },
    }

    plot_streams(stream_info_dict)


if __name__ == "__main__":
    main()
