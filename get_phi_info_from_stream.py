import glob
import os

from pdb import set_trace

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LogNorm

from astropy.coordinates import SkyCoord, Galactocentric, ICRS
import astropy.units as u

from gala.coordinates import GreatCircleICRSFrame

from scipy.stats import gaussian_kde, ks_2samp

import h5py

plt.rcParams["font.family"] = "serif"
plt.rcParams["mathtext.fontset"] = "dejavuserif"

DATA_DIR = "/home/btcook/Desktop/chc_ii_paper/circular_orbit_subdir/"
KPC_TO_PC = 1000.0
KMS_TO_KPC_PER_MYR = 1 / 978.5


def get_key(filename):
    return float(filename.split("_")[-1].replace(".csv", ""))


def load_data_chc():
    all_files = sorted(
        glob.glob(DATA_DIR + "chc_stream_no_stodolkiewicz/snapshot_job_id_*.csv"),
        key=get_key,
    )
    last_snapshot = all_files[-1]
    df_chc = pd.read_csv(last_snapshot, sep=", ")
    df_chc_bound = df_chc[df_chc["E_wrt_cluster"] < 0.0]
    df_chc_unbound = df_chc[df_chc["E_wrt_cluster"] >= 0.0]

    df_conserved_quantities_info = pd.read_csv(
        glob.glob(DATA_DIR + "chc_stream_no_stodolkiewicz/conserved_quantities_*.csv")[
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
        "df_bound": df_chc_bound,
        "df_unbound": df_chc_unbound,
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
        DATA_DIR + "stream_frog/time_567.0.txt",
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


def get_phi_one_two(df, ref_point_vec):
    """
    Compute phi_1 and phi_2 for a given dataframe containing MW phase space (x, y, z, vx, vy, vz),
    dynamically determining the best rotation plane from the dataset.

    Args:
        df (pd.DataFrame): DataFrame with columns ["x", "y", "z", "vx", "vy", "vz"]
        ref_point_vec (tuple): (x, y, z) coordinates of the progenitor in kpc.

    Returns:
        tuple: (phi_1, phi_2) in degrees as numpy arrays
    """
    # Convert position and velocity to arrays
    x, y, z = df["x"].values, df["y"].values, df["z"].values
    vx, vy, vz = df["vx"].values, df["vy"].values, df["vz"].values

    # Compute angular momentum vector L = r × v (cross product)
    Lx = y * vz - z * vy
    Ly = z * vx - x * vz
    Lz = x * vy - y * vx

    # Normalize the mean angular momentum vector
    L_vec = np.array([Lx.mean(), Ly.mean(), Lz.mean()])
    L_vec /= np.linalg.norm(L_vec)

    # Convert mean angular momentum to SkyCoord in Galactocentric frame
    """
    pole = SkyCoord(
        x=L_vec[0] * u.kpc,
        y=L_vec[1] * u.kpc,
        z=L_vec[2] * u.kpc,
        representation_type="cartesian",
        frame=Galactocentric(),
    ).transform_to(ICRS())
    """
    pole = SkyCoord(l=0.0 * u.deg, b=90.0 * u.deg, frame="galactic")
    pole = pole.transform_to(ICRS())

    # Set reference point for the progenitor
    ref_x, ref_y, ref_z = ref_point_vec
    ref = SkyCoord(
        x=ref_x * u.kpc,
        y=ref_y * u.kpc,
        z=ref_z * u.kpc,
        representation_type="cartesian",
        frame=Galactocentric(),
    ).transform_to(ICRS())

    # Define the great circle frame using the calculated pole and reference RA
    stream_frame = GreatCircleICRSFrame.from_pole_ra0(pole, ref.ra)

    # Transform the coordinates into the stream-aligned frame
    coords = SkyCoord(
        x=x * u.kpc,
        y=y * u.kpc,
        z=z * u.kpc,
        representation_type="cartesian",
        frame=Galactocentric(),
    ).transform_to(stream_frame)

    # Convert phi1 and phi2 to degrees, and wrap phi1 to [-180, 180)
    phi1_zero_to_360 = coords.phi1.deg + 180.0

    phi1 = np.where(
        phi1_zero_to_360 > 180.0, phi1_zero_to_360 - 360.0, phi1_zero_to_360
    )
    phi2 = coords.phi2.deg

    return phi1, phi2


def plot_phi1_phi2_kde_panels(stream_info_dict):
    """
    Create a 3-row, 1-column plot:
    - Row 1: KDE for CHC (phi1 vs phi2)
    - Row 2: KDE for Particle Spray
    - Row 3: KDE for Direct N-body
    """
    fig, axs = plt.subplots(3, 1, figsize=(8, 14), sharex=True, sharey=True)

    labels_ordered = [
        "CHC",
        "Particle Spray (Chen et al. (2024))",
        r"Direct $N$-body",
    ]

    colors = {
        "CHC": "C0",
        "Particle Spray (Chen et al. (2024))": "C1",
        r"Direct $N$-body": "C2",
    }

    # Global bounds for consistency across panels
    all_phi1 = np.concatenate(
        [stream_info_dict[label]["phi_one"] for label in labels_ordered]
    )
    all_phi2 = np.concatenate(
        [stream_info_dict[label]["phi_two"] for label in labels_ordered]
    )

    min_phi1, max_phi1 = np.percentile(all_phi1, [1, 99])
    min_phi2, max_phi2 = np.percentile(all_phi2, [1, 99])

    xi, yi = np.meshgrid(
        np.linspace(min_phi1, max_phi1, 200),
        np.linspace(min_phi2, max_phi2, 200),
    )

    zi_dict = {}
    for ax, label in zip(axs, labels_ordered):
        data = stream_info_dict[label]
        phi1, phi2 = data["phi_one"], data["phi_two"]

        kde = gaussian_kde(np.vstack([phi1, phi2]))
        zi = kde(np.vstack([xi.ravel(), yi.ravel()])).reshape(xi.shape)
        zi /= np.sum(zi)

        zi_dict[label] = zi

        im = ax.imshow(
            zi,
            extent=(min_phi1, max_phi1, min_phi2, max_phi2),
            origin="lower",
            aspect="auto",
            cmap="jet",  # You can change to 'plasma', 'inferno', etc.
            alpha=0.9,
            vmin=1e-6,
            vmax=7e-5,
        )

        ax.set_title(label, fontsize=14)
        ax.grid(linewidth=0.5, linestyle="--", alpha=0.5)

    import pdb

    pdb.set_trace()

    zi_chc = zi_dict["CHC"]
    zi_chen = zi_dict["Particle Spray (Chen et al. (2024))"]
    zi_n_body = zi_dict[r"Direct $N$-body"]

    print(f"KLD for CHC: {np.sum(zi_chc * np.log(zi_chc/zi_n_body))}")
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
    chc_info_dict = load_data_chc()
    df_chc_bound, df_chc_unbound = (
        chc_info_dict["df_bound"],
        chc_info_dict["df_unbound"],
    )
    ref_point_vec_chc = np.array(
        [df_chc_bound["x"].mean(), df_chc_bound["y"].mean(), df_chc_bound["z"].mean()]
    )
    chc_info_dict["phi"] = get_phi_one_two(df_chc_unbound, ref_point_vec_chc)

    df_chen = pd.read_csv(DATA_DIR + "chen_streams/chen_circular_orbit_stream.csv")
    df_chen_bound, df_chen_unbound = (
        df_chen[df_chen["source"] == "prog"],
        df_chen[df_chen["source"] == "stream"],
    )
    ref_point_vec_chen = np.array(
        [
            df_chen_bound["x"].mean(),
            df_chen_bound["y"].mean(),
            df_chen_bound["z"].mean(),
        ]
    )
    chen_phi_info = get_phi_one_two(df_chen, ref_point_vec_chen)

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
        ]
    )
    n_body_info_dict["phi"] = get_phi_one_two(df_n_body_unbound, ref_point_vec_n_body)

    # CHC IOMs have to be convert to specific IOMs
    stream_info_dict = {
        "CHC": {"phi_one": chc_info_dict["phi"][0], "phi_two": chc_info_dict["phi"][1]},
        "Particle Spray (Chen et al. (2024))": {
            "phi_one": chen_phi_info[0],
            "phi_two": chen_phi_info[1],
        },
        r"Direct $N$-body": {
            "phi_one": n_body_info_dict["phi"][0],
            "phi_two": n_body_info_dict["phi"][1],
        },
    }

    plot_phi1_phi2_kde_panels(stream_info_dict)


if __name__ == "__main__":
    main()
