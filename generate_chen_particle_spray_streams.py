import astropy.units as u
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import gala.potential as gp
import gala.dynamics as gd
from gala.dynamics import mockstream as ms
from gala.units import galactic

import agama  # to calculate action

agama.setUnits(length=1, velocity=1, mass=1)  # working units: 1 Msun, 1 kpc, 1 km/s

actFinder = agama.ActionFinder(
    agama.Potential(
        "/home/btcook/Desktop/github_repositories/CHC/agama/data/PriceWhelan22.ini"
    )
)


# Function to extract coordinates, velocities, E, Lz
def extract_particle_data(
    particle,
    stream_info_dict,
    source_label,
):
    E_wrt_host = stream_info_dict["hamiltonian"].potential.total_energy(
        particle.pos.xyz, particle.vel.d_xyz
    )
    Lz = particle.angular_momentum()[2]

    # phi1, phi2, mu_phi1, mu_phi2, dist, radial velocity
    return pd.DataFrame(
        {
            "x": particle.pos.x.to_value(u.kpc),
            "y": particle.pos.y.to_value(u.kpc),
            "z": particle.pos.z.to_value(u.kpc),
            "vx": particle.vel.d_x.to_value(u.km / u.s),
            "vy": particle.vel.d_y.to_value(u.km / u.s),
            "vz": particle.vel.d_z.to_value(u.km / u.s),
            "E_wrt_host": E_wrt_host.to((u.km / u.s) ** 2.0).value,
            "L_z": Lz.to(u.kpc * u.km / u.s).value,
            "source": source_label,
        }
    )


def create_stream(stream_info_dict):
    """
    Generate leading and trailing tidal stream particles using gala's particle_spray.
    Includes progenitor particle with a `source` label.
    """
    # Stream distribution function (Chen+25 or Fardal)
    dist_func = ms.ChenStreamDF()

    # Stream generator
    gen = ms.MockStreamGenerator(
        dist_func,
        stream_info_dict["hamiltonian"],
        progenitor_potential=stream_info_dict["progenitor"]["prog_pot"],
    )

    # Generate stream and progenitor
    stream, prog = gen.run(
        stream_info_dict["progenitor"]["prog_w0"],
        stream_info_dict["progenitor"]["prog_mass"],
        dt=stream_info_dict["dt"],
        n_steps=stream_info_dict["n_steps"],
    )

    # Build full DataFrame
    df_stream = extract_particle_data(stream, stream_info_dict, "stream")
    df_prog = extract_particle_data(prog, stream_info_dict, "prog")
    df = pd.concat([df_stream, df_prog], ignore_index=True)

    return df


if __name__ == "__main__":

    orbit_type = "eccentric_misaligned"
    progenitor_type = "plummer"

    host_pot = gp.MilkyWayPotential2022(units=galactic)
    host_hamiltonian = gp.Hamiltonian(host_pot)

    N_ORBITS = 8

    if orbit_type == "circular":
        # Define initial circular orbit, using 1.0 kpc or 16.95994177 kpc
        r_circ = 16.69302028
        pos = [r_circ, 0.0, 0.0] * u.kpc

        vT = host_pot.circular_velocity(pos)[0]
        vel = [0, vT.to_value(u.km / u.s), 0] * (u.km / u.s)

        print(np.round(vT, 8))

        period = 2 * np.pi * np.linalg.norm(pos) / np.linalg.norm(vel)
        t_tot = N_ORBITS * period

        print(f"orbital period: {np.round(period.to_value(u.Myr), 3)}")
        print(
            f"angular momentum: {np.round(np.linalg.norm(pos) * np.linalg.norm(vel) / 1000.0, 3)} 10^3 kpc km/s"
        )
        print(
            f"total energy: {np.round(host_pot.total_energy(pos, vel).to_value((u.km/u.s)**2.0) / 1e5, 3)} 10^5 (km/s)^2"
        )

    else:
        pos = [0.98105761, 3.15561992, 9.55769116] * u.kpc
        vel = [-51.13925321, 55.50809281, 261.59895211] * (u.km / u.s)

        phase_space = gd.PhaseSpacePosition(pos=pos.T, vel=vel.T)
        orbit = host_pot.integrate_orbit(phase_space, dt=0.1 * u.Myr, n_steps=10000)

        t_tot = N_ORBITS * orbit.physicsspherical.estimate_period()["r"][0]

    dt = 0.1 * u.Myr

    n_steps = t_tot / dt

    # load King model ICs from CHC
    DATA_DIR = "/home/btcook/Desktop/github_repositories/CHC/data/"
    column_names = ["id", "k", "m", "Reff", "binind", "x", "y", "z", "vx", "vy", "vz"]
    df_plummer_ics = pd.read_csv(
        DATA_DIR + "chc_king_ics_n_10000.csv", names=column_names
    )

    # King model information in N-body units
    positions = df_plummer_ics[["x", "y", "z"]].values
    n_stars = len(positions)
    masses = (1.0 / n_stars) * np.ones(n_stars) * u.Msun

    # set up the progenitor
    prog_mass = 1.0e5 * u.Msun
    prog_scale_radius = 24.5621 * u.pc  # this is r_vir from CHC, r_hm is 20 pc

    if progenitor_type == "scf":
        # compute SCF expansion coefficients using Lowing (2011) convention, N-body units
        NMAX, LMAX = 10, 5
        S, T = gp.scf.compute_coeffs_discrete(
            positions, mass=masses, nmax=NMAX, lmax=LMAX, r_s=1.0
        )

        prog_pot = gp.scf.SCFPotential(
            Snlm=S,
            Tnlm=T,
            m=prog_mass.to_value(u.Msun),
            r_s=prog_scale_radius.to_value(u.kpc),
            units=galactic,
        )

    else:

        prog_pot = gp.PlummerPotential(
            m=prog_mass.to_value(u.Msun),
            b=prog_scale_radius.to_value(u.kpc),
            units=galactic,
        )

    prog_w0 = gd.PhaseSpacePosition(
        pos=pos,
        vel=vel,
    )

    stream_info_dict = {
        "progenitor": {
            "prog_mass": prog_mass,
            "prog_w0": prog_w0,
            "prog_pot": prog_pot,
        },
        "hamiltonian": host_hamiltonian,
        "dt": t_tot / n_steps,
        "n_steps": n_steps,
    }

    df = create_stream(stream_info_dict)
    df.to_csv(
        f"chen_{orbit_type}_orbit_{progenitor_type}_progenitor_stream.csv", index=False
    )
