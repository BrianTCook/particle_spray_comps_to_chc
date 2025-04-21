import astropy.units as u
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import gala.potential as gp
import gala.dynamics as gd
from gala.dynamics import mockstream as ms
from gala.units import galactic


# Function to extract coordinates, velocities, E, Lz
def extract_particle_data(
    particle,
    stream_info_dict,
    source_label,
):
    Etot = stream_info_dict["hamiltonian"].potential.total_energy(
        particle.pos.xyz, particle.vel.d_xyz
    )
    Lz = particle.angular_momentum()[2]

    # phi1, phi2, mu_phi1, mu_phi2, dist, radial velocity
    return pd.DataFrame(
        {
            "x": particle.pos.x.to(u.kpc).value,
            "y": particle.pos.y.to(u.kpc).value,
            "z": particle.pos.z.to(u.kpc).value,
            "vx": particle.vel.d_x.to(u.km / u.s).value,
            "vy": particle.vel.d_y.to(u.km / u.s).value,
            "vz": particle.vel.d_z.to(u.km / u.s).value,
            "E_total": Etot.to((u.km / u.s) ** 2.0).value,
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

    host_pot = gp.BovyMWPotential2014(units=galactic)
    host_hamiltonian = gp.Hamiltonian(host_pot)

    # Define initial circular orbit
    R = 10.0 * u.kpc
    pos = [R.to(u.kpc).value, 0.0, 0.0] * u.kpc

    vT = host_pot.circular_velocity(pos)[0]
    vel = [0, vT.to(u.km / u.s).value, 0] * (u.km / u.s)

    n_steps = 2000
    t_orbit = (2 * np.pi * R / vT).to(u.Myr)
    t_tot = 10.0 * t_orbit

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

    # compute SCF expansion coefficients using Lowing (2011) convention, N-body units
    NMAX, LMAX = 5, 5
    S, T = gp.scf.compute_coeffs_discrete(
        positions, mass=masses, nmax=NMAX, lmax=LMAX, r_s=1.0
    )

    # set up the progenitor
    prog_mass = 1.0e5 * u.Msun
    prog_scale_radius = 24.5621 * u.pc  # this is r_vir from CHC, r_hm is 20 pc

    prog_pot = gp.scf.SCFPotential(
        Snlm=S,
        Tnlm=T,
        m=prog_mass.to(u.Msun).value,
        r_s=prog_scale_radius.to(u.kpc).value,
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
    df.to_csv("chen_circular_orbit_stream.csv", index=False)
