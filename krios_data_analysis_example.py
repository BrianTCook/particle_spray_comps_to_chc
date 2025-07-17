import os
import glob
from pdb import set_trace

import pandas as pd
import numpy as np

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LogNorm, SymLogNorm
from matplotlib.patches import Circle


mpl.style.use(
    "/home/btcook/Desktop/github_repositories/chc_python/matplotlib_style_file"
)

import gala.coordinates as gc
import astropy.coordinates as coord
import astropy.units as u

_ = coord.galactocentric_frame_defaults.set("v4.0")

import h5py


def get_key(filename):
    numerical_part = filename.split("_")[-1]
    numerical_part = numerical_part.replace(".csv", "")
    return float(numerical_part)


DATA_DIR = "/home/btcook/Desktop/krios_ii_paper/validation_tests/orbit_9_test_exec_and_results/"


def load_data_krios():

    all_files = sorted(
        glob.glob(DATA_DIR + "snapshot_job_id_*.csv"),
        key=get_key,
    )
    last_snapshot = all_files[-1]
    df_krios = pd.read_csv(last_snapshot, sep=", ")

    df_krios["E_wrt_cluster"] = df_krios["K_wrt_density_center"] + df_krios["U_nbody"]
    df_krios["E_wrt_host"] = df_krios["K_wrt_host"] + df_krios["U_wrt_host"]
    df_krios["is_bound"] = df_krios["E_wrt_cluster"] < 0.0

    """
    e.g. cluster mass, integrals of motion
    df_conserved_quantities_info = pd.read_csv(
        glob.glob(DATA_DIR + "conserved_quantities_*.csv")[0],
        header=0,
        sep=", ",
        engine="python",
    )
    """

    scf_frame_info = pd.read_csv(
        glob.glob(DATA_DIR + "scf_frame_*.csv")[0], header=0, sep=", ", engine="python"
    )

    time = float(last_snapshot.split("_")[-1].replace(".csv", ""))
    idx_closest = np.argmin(np.abs(scf_frame_info["t"] - time))
    last_snapshot_ref_point = scf_frame_info.iloc[idx_closest].to_dict()

    info_dict = {"df_last_snapshot": df_krios, "ref_point_vec": last_snapshot_ref_point}

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


if __name__ == "__main__":
    """
    This should provide enough info for KRIOS results section analysis:

    σ_φ₂(φ₁) (dispersions),
    μ_φ₁(φ₁), μ_φ₂(φ₁) (PMs),
    δφ₂(φ₁) (stream width),
    \vec{μ} · dφ₂/dφ₁ (disagreement angle)
    """

    # read in KRIOS data
    krios_info_dict = load_data_krios()

    df_last_snapshot = krios_info_dict["df_last_snapshot"]
    df_last_snapshot = df_last_snapshot[~df_last_snapshot["is_bound"]]  # stream only
    ref_point_dict = krios_info_dict["ref_point_vec"]
    ref_point_vec = (ref_point_dict["x"], ref_point_dict["y"], ref_point_dict["z"])

    # compute phi1, phi2 values
    phi_info_dict = get_phi_info(df_last_snapshot, ref_point_vec)
    phi1_vals, phi2_vals = phi_info_dict["phi1"], phi_info_dict["phi2"]
    mu_phi1_vals, mu_phi2_vals = phi_info_dict["mu_phi1"], phi_info_dict["mu_phi2"]

    # bin data
    lower, upper = 5.0, 95.0
    phi1_bins = np.arange(*np.percentile(phi1_vals, [lower, upper]), 4.0)
    phi2_bins = np.arange(*np.percentile(phi2_vals, [lower, upper]), 0.1)

    phi1_min, phi1_max = np.amin(phi1_vals), np.amax(phi1_vals)
    phi2_min, phi2_max = np.amin(phi2_vals), np.amax(phi2_vals)

    # histogram of data, compute centers based on edges
    phi1_binned, phi1_bin_edges = np.histogram(phi1_vals, bins=phi1_bins)
    phi1_bin_centers = 0.5 * (phi1_bin_edges[1:] + phi1_bin_edges[:-1])

    phi2_binned, phi2_bin_edges = np.histogram(phi2_vals, bins=phi2_bins)
    phi2_bin_centers = 0.5 * (phi2_bin_edges[1:] + phi2_bin_edges[:-1])

    # Initialize arrays
    phi2_std_binned = []
    mu_dot_dphi2_dphi1 = []

    # Bin indices
    bin_indices = np.digitize(phi1_vals, phi1_bins)

    # Loop over bins
    for i in range(1, len(phi1_bins)):
        in_bin = bin_indices == i
        if np.count_nonzero(in_bin) < 5:
            # Not enough particles to compute meaningful stats
            phi2_std_binned.append(np.nan)
            mu_dot_dphi2_dphi1.append(np.nan)
            continue

        phi1_bin = phi1_vals[in_bin]
        phi2_bin = phi2_vals[in_bin]
        mu_phi1_bin = mu_phi1_vals[in_bin]
        mu_phi2_bin = mu_phi2_vals[in_bin]

        # Standard deviation of phi2 in bin
        phi2_std = np.std(phi2_bin)
        phi2_std_binned.append(phi2_std)

        # Estimate dφ₂/dφ₁ via linear fit
        slope, _ = np.polyfit(phi1_bin, phi2_bin, 1)

        # Stream direction unit vector (dφ₁, dφ₂)
        stream_dir = np.array([1.0, slope])
        stream_dir /= np.linalg.norm(stream_dir)

        # Compute mean velocity in stream frame in this bin
        mean_mu_phi1 = np.mean(mu_phi1_bin)
        mean_mu_phi2 = np.mean(mu_phi2_bin)
        mean_vel = np.array([mean_mu_phi1, mean_mu_phi2])
        mean_vel /= np.linalg.norm(mean_vel)

        # Dot product = cos(θ), where θ is disagreement angle
        alignment = np.dot(mean_vel, stream_dir)
        mu_dot_dphi2_dphi1.append(alignment)

    # Convert to arrays
    phi2_std_binned = np.array(phi2_std_binned)
    mu_dot_dphi2_dphi1 = np.array(mu_dot_dphi2_dphi1)

    # 1. Sky-plane: φ₂ vs φ₁
    plt.figure()
    plt.scatter(
        phi1_vals,
        phi2_vals,
        s=2,
        alpha=1.0,
        edgecolors=None,
        linewidths=0,
    )
    plt.xlabel(r"$\phi_{1}$ [deg]", fontsize=14)
    plt.ylabel(r"$\phi_{2}$ [deg]", fontsize=14)
    plt.savefig("sky_plane_orbit_9.png", dpi=400)

    # 2. Proper motions: μ_φ₁ and μ_φ₂ vs φ₁ (two-panel plot)
    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(6, 6))
    ax1.scatter(
        phi_info_dict["phi1"],
        phi_info_dict["mu_phi1"],
        s=2,
        alpha=1.0,
        edgecolors=None,
        linewidths=0,
    )
    ax1.set_ylabel(r"$\mu_{\phi_{1}}$ [mas/yr]", fontsize=14)

    ax2.scatter(
        phi_info_dict["phi1"],
        phi_info_dict["mu_phi2"],
        s=2,
        alpha=1.0,
        edgecolors=None,
        linewidths=0,
    )
    ax2.set_xlabel(r"$\phi_{1}$ [deg]", fontsize=14)
    ax2.set_ylabel(r"$\mu_{\phi_{2}}$ [mas/yr]", fontsize=14)

    plt.savefig("proper_motions_orbit_9.png", dpi=400)

    # 3. Stream width: δφ₂ vs φ₁
    plt.figure()
    plt.plot(
        phi1_bin_centers,
        phi2_std_binned,
    )
    plt.xlabel(r"$\phi_{1}$ [deg]", fontsize=14)
    plt.ylabel(r"$\delta \phi_{2}$ [deg]", fontsize=14)
    plt.savefig("stream_width_orbit_9.png", dpi=400)

    # 4. Disagreement angle: mu · dφ₂/dφ₁ vs φ₁
    plt.figure()
    plt.plot(
        phi1_bin_centers,
        mu_dot_dphi2_dphi1,
    )
    plt.xlabel(r"$\phi_{1}$ [deg]", fontsize=14)
    plt.ylabel(r"$\vec{\mu} \cdot \frac{d\phi_{2}}{d\phi_{1}}$", fontsize=14)
    plt.savefig("disagreement_angle_orbit_9.png", dpi=400)
