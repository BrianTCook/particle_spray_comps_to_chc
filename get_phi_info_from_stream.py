import glob
import os

from pdb import set_trace

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LogNorm

import gala.coordinates as gc
import astropy.coordinates as coord
import astropy.units as u

_ = coord.galactocentric_frame_defaults.set("v4.0")

from scipy.stats import gaussian_kde, ks_2samp

import h5py

plt.rcParams["font.family"] = "serif"
plt.rcParams["mathtext.fontset"] = "dejavuserif"

ORBIT_TYPE = "eccentric_misaligned"
DATA_DIR = f"/home/btcook/Desktop/krios_ii_paper/{ORBIT_TYPE}_orbit_subdir/"
KPC_TO_PC = 1000.0
KMS_TO_KPC_PER_MYR = 1 / 978.5


def get_key(filename):
    return float(filename.split("_")[-1].replace(".csv", ""))

def load_data_krios():
    krios_dir = DATA_DIR + f"krios_stream/{ORBIT_TYPE}_orbit_radial_pairing_exec_and_results/"

    all_files = sorted(
        glob.glob(krios_dir + "snapshot_job_id_*.csv"),
        key=get_key,
    )
    last_snapshot = all_files[-1]
    df_krios = pd.read_csv(last_snapshot, sep=", ")
    df_krios_bound = df_krios[df_krios["E_wrt_cluster"] < 0.0]
    df_krios_unbound = df_krios[df_krios["E_wrt_cluster"] >= 0.0]

    df_conserved_quantities_info = pd.read_csv(
        glob.glob(krios_dir + "conserved_quantities_*.csv")[
            0
        ],
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


def load_data_n_body():
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
        DATA_DIR + "n_body_stream/",
        sep="\t",
        names=["x", "y", "z", "vx", "vy", "vz", "m", "U_int", "U_c"],
        index_col=False,
    )

    # get unit conversion info
    n_body_h5_filename = DATA_DIR + "stream_frog/iom_cluster_63879115934970.hf5"
    with h5py.File(n_body_h5_filename, "r") as f:
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

    """
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

    set_trace()

    plt.hist(df["E_wrt_cluster"])
    plt.show()

    df = df[df["E_wrt_cluster"] > 0.0]
    """

    # Coordinates of the well particle
    well_particle = df[df["U_int"] == df["U_int"].min()]
    x_well, y_well, z_well = (
        well_particle["x"].values[0],
        well_particle["y"].values[0],
        well_particle["z"].values[0],
    )

    # Compute squared distances for efficiency
    dist_sq = (
        (df["x"] - x_well) ** 2 + (df["y"] - y_well) ** 2 + (df["z"] - z_well) ** 2
    )

    # Filter out particles within r_t
    r_t = 0.08781883404823516  # tidal radius, kpc
    df_bound = df[dist_sq <= r_t**2.0]
    df_unbound = df[dist_sq > r_t**2.0]

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


def plot_phi1_phi2_kde_panels(stream_info_dict):
    """
    Create a 3-row, 1-column plot:
    - Row 1: KDE for KRIOS (phi1 vs phi2)
    - Row 2: KDE for Particle Spray
    - Row 3: KDE for Direct N-body
    """
    fig, axs = plt.subplots(3, 1, figsize=(8, 14), sharex=True, sharey=True)

    codenames_ordered = [
        "KRIOS",
        "Chen",
        "N-body",
    ]

    label_dict = {
        "KRIOS": "KRIOS",
        "Chen": "Particle Spray (Chen et al. (2025))",
        "N-body": r"Direct $N$-body",
    }

    # Global bounds for consistency across panels
    all_phi1 = np.concatenate(
        [
            stream_info_dict[codename]["phi_info"]["phi1"]
            for codename in codenames_ordered
        ]
    )
    all_phi2 = np.concatenate(
        [
            stream_info_dict[codename]["phi_info"]["phi2"]
            for codename in codenames_ordered
        ]
    )

    n_phi1 = 100
    min_phi1, max_phi1 = np.percentile(all_phi1, [0.5, 99.5])
    phi1_bins = np.linspace(min_phi1, max_phi1, n_phi1)

    n_phi2 = 100
    min_phi2, max_phi2 = np.percentile(all_phi2, [0.5, 99.5])
    phi2_bins = np.linspace(min_phi2, max_phi2, n_phi2)

    n_bins = n_phi1 * n_phi2

    zi_dict = {}
    for ax, codename in zip(axs, codenames_ordered):
        data = stream_info_dict[codename]
        phi1, phi2 = data["phi_info"]["phi1"], data["phi_info"]["phi2"]

        kde = gaussian_kde(np.vstack([phi1, phi2]))
        xi, yi = np.meshgrid(phi1_bins, phi2_bins)
        zi = kde(np.vstack([xi.ravel(), yi.ravel()])).reshape(xi.shape)
        zi /= np.sum(zi)
        
        """
        # Compute 2D histogram
        hist, _, _= np.histogram2d(
            phi1,
            phi2,
            bins=[phi1_bins, phi2_bins],
            range=[[min_phi1, max_phi1], [min_phi2, max_phi2]],
            density=True,  # Normalize the histogram
        )

        # hist is indexed as [x, y] but imshow expects [y, x], so transpose
        zi = hist.T
        """

        zi_dict[codename] = zi

        im = ax.imshow(
            zi,
            extent=(min_phi1, max_phi1, min_phi2, max_phi2),
            origin="lower",
            aspect="auto",
            cmap="jet",  # You can change to 'plasma', 'inferno', etc.
            #norm=LogNorm(vmin=0.01 / n_bins, vmax=10.0 / n_bins),
            alpha=0.8,
        )

        ax.set_title(label_dict[codename], fontsize=14)
        ax.grid(linewidth=0.5, linestyle="--", alpha=0.5)

    zi_krios = zi_dict["KRIOS"]
    zi_chen = zi_dict["Chen"]
    zi_n_body = zi_dict[r"N-body"]

    print(f"KLD for KRIOS: {np.sum(zi_krios * np.log(zi_krios/zi_n_body))}")
    print(f"KLD for particle spray: {np.sum(zi_chen * np.log(zi_chen/zi_n_body))}")

    axs[-1].set_xlabel(r"$\phi_1$ [deg]", fontsize=16)
    for ax in axs:
        ax.set_ylabel(r"$\phi_2$ [deg]", fontsize=16)

    fig.subplots_adjust(right=0.85)
    cbar_ax = fig.add_axes([1.03, 0.05, 0.05, 0.9])  # adjust position as needed
    fig.colorbar(im, cax=cbar_ax).set_label("PDF", fontsize=12)

    plt.tight_layout()
    plt.savefig("phi1_phi2_kde_panels.png", bbox_inches="tight", dpi=400)


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
            df_krios_bound["vx"].mean(),
            df_krios_bound["vy"].mean(),
            df_krios_bound["vz"].mean(),
        ]
    )
    krios_info_dict["phi_info"] = get_phi_info(df_krios_unbound, ref_point_vec_krios)

    df_chen = pd.read_csv(DATA_DIR + f"chen_stream/chen_{ORBIT_TYPE}_orbit_stream.csv")
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
    chen_phi_info = {"phi_info": get_phi_info(df_chen, ref_point_vec_chen)}

    n_body_info_dict = load_data_n_body()
    df_n_body_bound, df_n_body_unbound = (
        n_body_info_dict["df_bound"],
        n_body_info_dict["df_unbound"],
    )
    ref_point_vec_n_body = np.array(
        [
            df_n_body_bound["x"].mean(),
            df_n_body_bound["y"].mean(),
            df_n_body_bound["z"].mean(),
            df_n_body_bound["vx"].mean(),
            df_n_body_bound["vy"].mean(),
            df_n_body_bound["vz"].mean(),
        ]
    )
    n_body_info_dict["phi_info"] = get_phi_info(df_n_body_unbound, ref_point_vec_n_body)

    # KRIOS IOMs have to be convert to specific IOMs
    stream_info_dict = {
        "KRIOS": krios_info_dict,
        "Chen": chen_phi_info,
        "N-body": n_body_info_dict,
    }

    plot_phi1_phi2_kde_panels(stream_info_dict)


if __name__ == "__main__":
    main()
