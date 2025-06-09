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
import astropy.coordinates as coord
import astropy.units as u

_ = coord.galactocentric_frame_defaults.set("v4.0")

from scipy.stats import gaussian_kde

import h5py

plt.rcParams["font.family"] = "serif"
plt.rcParams["mathtext.fontset"] = "dejavuserif"

orbit_type = "eccentric_misaligned"
pairing_type = "radial"

DATA_DIR = f"/home/btcook/Desktop/krios_ii_paper/{orbit_type}_orbit_subdir/"
KMS_TO_KPC_PER_MYR = 1 / 978.5

import agama  # to calculate action

agama.setUnits(length=1, velocity=1, mass=1)  # working units: 1 Msun, 1 kpc, 1 km/s

actFinder = agama.ActionFinder(
    agama.Potential(
        "/home/btcook/Desktop/github_repositories/CHC/agama/data/MWPotential2014.ini"
    )
)


def get_action_info(df_unbound):
    # Assumes df_bound and df_unbound are DataFrames with columns ['x','y','z','vx','vy','vz'] in AGAMA units
    posvel_unbound = df_unbound[["x", "y", "z", "vx", "vy", "vz"]].to_numpy()
    actions_unbound = actFinder(posvel_unbound)

    # shape (N_unbound, 3)
    return {"Jr": actions_unbound[:, 0], "Lz": actions_unbound[:, 2]}


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
    df_krios = pd.read_csv(last_snapshot, sep=", ", engine="python")
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


def compute_kld_angles(stream_info_dict):
    """
    Compute Kullback-Leibler divergence for different stream models.
    """
    codenames_to_compare = [
        "Particle Spray (Chen et al. (2025))",
        "KRIOS",
        "N-body (second seed)",
    ]
    nbody_references = ["N-body (first seed)"]

    # Combine all phi1 and phi2 data for consistent binning
    all_phi1 = np.concatenate(
        [
            stream_info_dict[codename]["phi_info"]["phi1"]
            for codename in codenames_to_compare + nbody_references
        ]
    )
    all_phi2 = np.concatenate(
        [
            stream_info_dict[codename]["phi_info"]["phi2"]
            for codename in codenames_to_compare + nbody_references
        ]
    )

    lower, upper = 0.0, 100.0 #2.275, 97.725

    n_phi1 = 100
    phi1_bins = np.linspace(*np.percentile(all_phi1, [lower, upper]), n_phi1)

    n_phi2 = 100
    phi2_bins = np.linspace(*np.percentile(all_phi2, [lower, upper]), n_phi2)

    phi1_min, phi1_max = np.amin(all_phi1), np.amax(all_phi1)
    phi2_min, phi2_max = np.amin(all_phi2), np.amax(all_phi2)

    # Combine all codenames
    all_codenames = codenames_to_compare + nbody_references
    n_codenames = len(all_codenames)

    # Prepare figure
    fig, axes = plt.subplots(
        nrows=n_codenames,
        ncols=1,
        figsize=(18, 5),
        sharex=True,
        sharey=True,
        gridspec_kw={"hspace": -0.25},  # <<< This removes vertical space between rows
    )

    # Handle single-axis case
    if n_codenames == 1:
        axes = [axes]

    # Dictionary to store KDE results
    zi_dict = {}

    for ax, codename in zip(axes, all_codenames):
        phi1 = stream_info_dict[codename]["phi_info"]["phi1"]
        phi2 = stream_info_dict[codename]["phi_info"]["phi2"]

        # Plot scatter
        ax.scatter(phi1, phi2, c="k", s=0.5, alpha=1.0, edgecolors="none", linewidths=0)
        ax.set_xlim(phi1_min, phi1_max)
        ax.set_ylim(phi2_min, phi2_max)
        #ax.set_aspect("equal")
        ax.annotate(
            codename,
            xy=(0.7, 0.8),
            xycoords="axes fraction",
            color="black",
            fontsize=10,
            verticalalignment="top",
            horizontalalignment="left",
        )

        # KDE
        kde = gaussian_kde(np.vstack([phi1, phi2]))
        xi, yi = np.meshgrid(phi1_bins, phi2_bins)
        zi = kde(np.vstack([xi.ravel(), yi.ravel()])).reshape(xi.shape)
        zi /= np.sum(zi)  # Normalize to make it a probability distribution
        zi_dict[codename] = zi

    # Shared axis labels
    fig.text(0.5, 0.03, r"$\phi_1$ [deg]", ha="center", fontsize=24)
    fig.text(
        0.07, 0.5, r"$\phi_2$ [deg]", va="center", fontsize=24, rotation="vertical"
    )

    plt.savefig("phi1_phi2_dda_scatter.png", dpi=800, bbox_inches="tight")
    plt.close()

    # Compute KLD
    for nbody_key in nbody_references:
        zi_nbody = zi_dict[nbody_key]
        print(f"\nUsing {nbody_key} as reference for sky-plane angle KLDs:")
        for codename in codenames_to_compare:
            zi = zi_dict[codename]
            kld = np.sum(np.multiply(zi_nbody, np.log(np.divide(zi_nbody, zi))))
            print(f"  KLD({nbody_key} || {codename}) = {kld:.3f}")


def compute_kld_actions(stream_info_dict):
    """
    Compute Kullback-Leibler divergence for different stream models.
    """
    codenames_to_compare = [
        "KRIOS",
        "Particle Spray (Chen et al. (2025))",
        "N-body (second seed)",
    ]
    nbody_references = ["N-body (first seed)"]

    # Combine all Lz and Jr data for consistent binning
    all_Lz = np.concatenate(
        [
            stream_info_dict[codename]["action_info"]["Lz"]
            for codename in codenames_to_compare + nbody_references
        ]
    )
    all_Jr = np.concatenate(
        [
            stream_info_dict[codename]["action_info"]["Jr"]
            for codename in codenames_to_compare + nbody_references
        ]
    )

    lower, upper = 0.0, 100.0 #2.275, 97.725

    n_Lz = 100
    Lz_bins = np.linspace(*np.percentile(all_Lz, [lower, upper]), n_Lz)

    n_Jr = 100
    Jr_bins = np.linspace(*np.percentile(all_Jr, [lower, upper]), n_Jr)

    # Estimate density for each codename
    zi_dict = {}
    for codename in codenames_to_compare + nbody_references:
        Lz = stream_info_dict[codename]["action_info"]["Lz"]
        Jr = stream_info_dict[codename]["action_info"]["Jr"]

        kde = gaussian_kde(np.vstack([Lz, Jr]))
        xi, yi = np.meshgrid(Lz_bins, Jr_bins)
        zi = kde(np.vstack([xi.ravel(), yi.ravel()])).reshape(xi.shape)
        zi /= np.sum(zi)  # Normalize to make it a probability distribution

        zi_dict[codename] = zi

    # Compute KLD
    for nbody_key in nbody_references:
        zi_nbody = zi_dict[nbody_key]
        print(f"\nUsing {nbody_key} as reference for action KLDs:")
        for codename in codenames_to_compare:
            zi = zi_dict[codename]
            kld = np.sum(np.multiply(zi_nbody, np.log(np.divide(zi_nbody, zi))))
            print(f"  KLD({codename} || {nbody_key}) = {kld:.3f}")


def main():
    """
    KRIOS
    """
    print(orbit_type, pairing_type)
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
            df_krios_bound["vx"].mean(),
            df_krios_bound["vy"].mean(),
            df_krios_bound["vz"].mean(),
        ]
    )

    """
    particle-spray
    """
    df_chen = pd.read_csv(DATA_DIR + f"chen_stream/chen_{orbit_type}_orbit_stream.csv")
    df_chen_bound, df_chen_unbound = (
        df_chen[df_chen["source"] == "prog"],
        df_chen[df_chen["source"] == "stream"],
    )
    ref_point_vec_chen = np.array(
        [
            df_chen_bound["x"].mean(),
            df_chen_bound["y"].mean(),
            df_chen_bound["z"].mean(),
            df_chen_bound["vx"].mean(),
            df_chen_bound["vy"].mean(),
            df_chen_bound["vz"].mean(),
        ]
    )

    """
    N-body, first seed
    """
    first_seed_txt_filename = DATA_DIR + "/n_body_stream/last_snapshot_first_seed.txt"
    first_seed_hf5_filename = DATA_DIR + "/n_body_stream/first_seed.hf5"
    n_body_info_dict_first_seed = load_data_n_body(
        first_seed_txt_filename, first_seed_hf5_filename
    )
    df_n_body_bound_first_seed, df_n_body_unbound_first_seed = (
        n_body_info_dict_first_seed["df_bound"],
        n_body_info_dict_first_seed["df_unbound"],
    )
    ref_point_vec_n_body_first_seed = np.array(
        [
            df_n_body_bound_first_seed["x"].mean(),
            df_n_body_bound_first_seed["y"].mean(),
            df_n_body_bound_first_seed["z"].mean(),
            df_n_body_bound_first_seed["vx"].mean(),
            df_n_body_bound_first_seed["vy"].mean(),
            df_n_body_bound_first_seed["vz"].mean(),
        ]
    )

    """
    N-body, second seed
    """
    second_seed_txt_filename = DATA_DIR + "/n_body_stream/last_snapshot_second_seed.txt"
    second_seed_hf5_filename = DATA_DIR + "/n_body_stream/second_seed.hf5"
    n_body_info_dict_second_seed = load_data_n_body(
        second_seed_txt_filename, second_seed_hf5_filename
    )
    df_n_body_bound_second_seed, df_n_body_unbound_second_seed = (
        n_body_info_dict_second_seed["df_bound"],
        n_body_info_dict_second_seed["df_unbound"],
    )
    ref_point_vec_n_body_second_seed = np.array(
        [
            df_n_body_bound_second_seed["x"].mean(),
            df_n_body_bound_second_seed["y"].mean(),
            df_n_body_bound_second_seed["z"].mean(),
            df_n_body_bound_second_seed["vx"].mean(),
            df_n_body_bound_second_seed["vy"].mean(),
            df_n_body_bound_second_seed["vz"].mean(),
        ]
    )

    # KRIOS IOMs have to be convert to specific IOMs
    stream_info_dict = {
        "KRIOS": {
            "df": df_krios_unbound,
            "phi_info": get_phi_info(df_krios_unbound, ref_point_vec_krios),
            "action_info": get_action_info(df_krios_unbound),
        },
        "Particle Spray (Chen et al. (2025))": {
            "df": df_chen_unbound,
            "phi_info": get_phi_info(df_chen_unbound, ref_point_vec_chen),
            "action_info": get_action_info(df_chen_unbound),
        },
        "N-body (first seed)": {
            "df": df_n_body_unbound_first_seed,
            "phi_info": get_phi_info(
                df_n_body_unbound_first_seed, ref_point_vec_n_body_first_seed
            ),
            "action_info": get_action_info(df_n_body_unbound_first_seed),
        },
        "N-body (second seed)": {
            "df": df_n_body_unbound_second_seed,
            "phi_info": get_phi_info(
                df_n_body_unbound_second_seed, ref_point_vec_n_body_second_seed
            ),
            "action_info": get_action_info(df_n_body_unbound_second_seed),
        },
    }

    compute_kld_angles(stream_info_dict)
    compute_kld_actions(stream_info_dict)


if __name__ == "__main__":
    main()
