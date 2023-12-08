"""This script is used to build the base collider with Xmask, configuring only the optics. Functions
in this script are called sequentially."""
# ==================================================================================================
# --- Imports
# ==================================================================================================
from cpymad.madx import Madx
import os
import xmask as xm
import xmask.lhc as xlhc
import shutil
import json
import yaml
import logging
import numpy as np
import itertools
import pandas as pd
import tree_maker
import xtrack as xt
import xtrack._temp.lhc_match as lm
import xpart as xp

# Import user-defined optics-specific tools
import optics_specific_tools as ost


# ==================================================================================================
# --- Function for tree_maker tagging
# ==================================================================================================
def tree_maker_tagging(config, tag="started"):
    # Start tree_maker logging if log_file is present in config
    if tree_maker is not None and "log_file" in config:
        tree_maker.tag_json.tag_it(config["log_file"], tag)
    else:
        logging.warning("tree_maker loging not available")


# ==================================================================================================
# --- Function to load configuration file
# ==================================================================================================
def load_configuration(config_path="config.yaml"):
    # Load configuration
    with open(config_path, "r") as fid:
        configuration = yaml.safe_load(fid)

    # Get configuration for the particles distribution and the collider separately
    config_particles = configuration["config_particles"]
    config_mad = configuration["config_mad"]

    return configuration, config_particles, config_mad


# ==================================================================================================
# --- Function to build particle distribution and write it to file
# ==================================================================================================
def build_particle_distribution(config_particles):
    # Define radius distribution
    r_min = config_particles["r_min"]
    r_max = config_particles["r_max"]
    n_r = config_particles["n_r"]
    radial_list = np.linspace(r_min, r_max, n_r, endpoint=False)

    # Filter out particles with low and high amplitude to accelerate simulation
    # radial_list = radial_list[(radial_list >= 4.5) & (radial_list <= 7.5)]

    # Define angle distribution
    n_angles = config_particles["n_angles"]
    theta_list = np.linspace(0, 90, n_angles + 2)[1:-1]

    # Define particle distribution as a cartesian product of the above
    particle_list = [
        (particle_id, ii[1], ii[0])
        for particle_id, ii in enumerate(itertools.product(theta_list, radial_list))
    ]

    # Split distribution into several chunks for parallelization
    n_split = config_particles["n_split"]
    particle_list = list(np.array_split(particle_list, n_split))

    # Return distribution
    return particle_list


def write_particle_distribution(particle_list):
    # Write distribution to parquet files
    distributions_folder = "particles"
    os.makedirs(distributions_folder, exist_ok=True)
    for idx_chunk, my_list in enumerate(particle_list):
        pd.DataFrame(
            my_list,
            columns=["particle_id", "normalized amplitude in xy-plane", "angle in xy-plane [deg]"],
        ).to_parquet(f"{distributions_folder}/{idx_chunk:02}.parquet")


# ==================================================================================================
# --- Function to build collider from mad model
# ==================================================================================================
def build_collider_from_mad(config_mad, sanity_checks=True):
    # Make mad environment
    xm.make_mad_environment(links=config_mad["links"])

    # Start mad
    mad_b1b2 = Madx(command_log="mad_collider.log")

    mad_b4 = Madx(command_log="mad_b4.log")

    # Build sequences
    ost.build_sequence(mad_b1b2, mylhcbeam=1)
    ost.build_sequence(mad_b4, mylhcbeam=4)

    # Apply optics (only for b1b2, b4 will be generated from b1b2)
    ost.apply_optics(mad_b1b2, optics_file=config_mad["optics_file"])

    if sanity_checks:
        mad_b1b2.use(sequence="lhcb1")
        mad_b1b2.twiss()
        ost.check_madx_lattices(mad_b1b2)
        mad_b1b2.use(sequence="lhcb2")
        mad_b1b2.twiss()
        ost.check_madx_lattices(mad_b1b2)

    # Apply optics (only for b4, just for check)
    ost.apply_optics(mad_b4, optics_file=config_mad["optics_file"])
    if sanity_checks:
        mad_b4.use(sequence="lhcb2")
        mad_b4.twiss()
        ost.check_madx_lattices(mad_b1b2)

    # Build xsuite collider
    collider = xlhc.build_xsuite_collider(
        sequence_b1=mad_b1b2.sequence.lhcb1,
        sequence_b2=mad_b1b2.sequence.lhcb2,
        sequence_b4=mad_b4.sequence.lhcb2,
        beam_config=config_mad["beam_config"],
        enable_imperfections=config_mad["enable_imperfections"],
        enable_knob_synthesis=config_mad["enable_knob_synthesis"],
        rename_coupling_knobs=config_mad["rename_coupling_knobs"],
        pars_for_imperfections=config_mad["pars_for_imperfections"],
        ver_lhc_run=config_mad["ver_lhc_run"],
        ver_hllhc_optics=config_mad["ver_hllhc_optics"],
    )
    collider.build_trackers()

    if sanity_checks:
        collider["lhcb1"].twiss(method="4d")
        collider["lhcb2"].twiss(method="4d")
    # Return collider
    return collider


def activate_RF_and_twiss(collider, config_mad, sanity_checks=True):
    # Define a RF system (values are not so immportant as they're defined later)
    print("--- Now Computing Twiss assuming:")
    if config_mad["ver_hllhc_optics"] == 1.6:
        dic_rf = {"vrf400": 16.0, "lagrf400.b1": 0.5, "lagrf400.b2": 0.5}
        for knob, val in dic_rf.items():
            print(f"    {knob} = {val}")
    elif config_mad["ver_lhc_run"] == 3.0:
        dic_rf = {"vrf400": 12.0, "lagrf400.b1": 0.5, "lagrf400.b2": 0.0}
        for knob, val in dic_rf.items():
            print(f"    {knob} = {val}")
    print("---")

    # Rebuild tracker if needed
    try:
        collider.build_trackers()
    except:
        print("Skipping rebuilding tracker")

    for knob, val in dic_rf.items():
        collider.vars[knob] = val

    if sanity_checks:
        for my_line in ["lhcb1", "lhcb2"]:
            ost.check_xsuite_lattices(collider[my_line])

    return collider


def clean():
    # Remove all the temporaty files created in the process of building collider
    os.remove("mad_collider.log")
    os.remove("mad_b4.log")
    shutil.rmtree("temp")
    os.unlink("errors")
    os.unlink("acc-models-lhc")


# ==================================================================================================
# --- Main function for building distribution and collider
# ==================================================================================================
def build_distr_and_collider(config_file="config.yaml"):
    # Get configuration
    configuration, config_particles, config_mad = load_configuration(config_file)

    # Get sanity checks flag
    sanity_checks = configuration["sanity_checks"]

    # Tag start of the job
    tree_maker_tagging(configuration, tag="started")

    # Build particle distribution
    particle_list = build_particle_distribution(config_particles)

    # Write particle distribution to file
    write_particle_distribution(particle_list)

    # Build collider from mad model
    collider = build_collider_from_mad(config_mad, sanity_checks)

    # Twiss to ensure eveyrthing is ok
    collider = activate_RF_and_twiss(collider, config_mad, sanity_checks)

    # Clean temporary files
    clean()

    # Save collider to json
    os.makedirs("collider", exist_ok=True)
    collider.to_json("collider/collider.json")

    # Tag end of the job
    tree_maker_tagging(configuration, tag="completed")


# ==================================================================================================
# --- Function to start generation of colliders
# ==================================================================================================
def match_ip15_phase(collider, tar_mux15, tar_muy15, staged_match=True, solve=True):
    tw0 = collider.twiss()
    mux_15_orig_b1 = tw0["lhcb1"][:, "ip1"].mux[0] - tw0["lhcb1"][:, "ip5"].mux[0]
    muy_15_orig_b1 = tw0["lhcb1"][:, "ip1"].muy[0] - tw0["lhcb1"][:, "ip5"].muy[0]
    mux_15_orig_b2 = tw0["lhcb2"][:, "ip1"].mux[0] - tw0["lhcb2"][:, "ip5"].mux[0]
    muy_15_orig_b2 = tw0["lhcb2"][:, "ip1"].muy[0] - tw0["lhcb2"][:, "ip5"].muy[0]

    refqxb1 = tw0["lhcb1"].qx
    refqyb1 = tw0["lhcb1"].qy
    refqxb2 = tw0["lhcb2"].qx
    refqyb2 = tw0["lhcb2"].qy

    if mux_15_orig_b1 < 0:
        mux_15_orig_b1 += refqxb1
    if muy_15_orig_b1 < 0:
        muy_15_orig_b1 += refqyb1
    if mux_15_orig_b2 < 0:
        mux_15_orig_b2 += refqxb2
    if muy_15_orig_b2 < 0:
        muy_15_orig_b2 += refqyb2

    print(f"{mux_15_orig_b1=}")
    print(f"{muy_15_orig_b1=}")
    print(f"{mux_15_orig_b2=}")
    print(f"{muy_15_orig_b2=}")

    d_mux_15_b1 = mux_15_orig_b1 - tar_mux15
    d_muy_15_b1 = muy_15_orig_b1 - tar_muy15
    d_mux_15_b2 = mux_15_orig_b2 - tar_mux15
    d_muy_15_b2 = muy_15_orig_b2 - tar_muy15

    print(f"{tar_mux15=}, {tar_muy15=}")
    print(f"{d_mux_15_b1=}, {d_muy_15_b1=}")
    print(f"{d_mux_15_b2=}, {d_muy_15_b2=}")

    opts = ost.change_ip15_phase(
        collider,
        dqx=tar_mux15,
        dqy=tar_muy15,
        d_mux_15_b1=d_mux_15_b1,
        d_muy_15_b1=d_muy_15_b1,
        d_mux_15_b2=d_mux_15_b2,
        d_muy_15_b2=d_muy_15_b2,
        staged_match=staged_match,
        solve=solve,
    )

    return collider, opts

def build_phase_colliders(config_file="config.yaml"):
    configuration, config_particles, config_mad = load_configuration(config_file)

    muxs = config_mad["config_ip15_phase"]["muxs"]
    muys = config_mad["config_ip15_phase"]["muys"]

    # Make mad environment
    xm.make_mad_environment(links=config_mad["links"])

    # Start mad
    mad_b1b2 = Madx(command_log="mad_collider.log")

    mad_b4 = Madx(command_log="mad_b4.log")

    # Build sequences
    ost.build_sequence(mad_b1b2, mylhcbeam=1, ignore_cycling=True)
    ost.build_sequence(mad_b4, mylhcbeam=4, ignore_cycling=True)

    # Apply optics (only for b1b2, b4 will be generated from b1b2)
    ost.apply_optics(mad_b1b2, optics_file=config_mad["optics_file"])
    mad_b1b2.use("lhcb1")
    mad_b1b2.twiss()

    # Apply optics (only for b4, just for check)
    ost.apply_optics(mad_b4, optics_file=config_mad["optics_file"])

    mad_b4.use("lhcb2")
    mad_b4.twiss()

    line1 = xt.Line.from_madx_sequence(
        mad_b1b2.sequence.lhcb1,
        allow_thick=True,
        deferred_expressions=True,
        replace_in_expr={"bv_aux": "bvaux_b1"},
    )

    line4 = xt.Line.from_madx_sequence(
        mad_b4.sequence.lhcb2,
        allow_thick=True,
        deferred_expressions=True,
        replace_in_expr={"bv_aux": "bvaux_b2"},
    )

    # Remove solenoids (cannot backtwiss for now)
    for ll in [line1, line4]:
        tt = ll.get_table()
        for nn in tt.rows[tt.element_type == "Solenoid"].name:
            ee_elen = ll[nn].length
            ll.element_dict[nn] = xt.Drift(length=ee_elen)

    collider = xt.Multiline(lines={"lhcb1": line1, "lhcb2": line4})
    collider.lhcb1.particle_ref = xp.Particles(mass0=xp.PROTON_MASS_EV, p0c=7000e9)
    collider.lhcb2.particle_ref = xp.Particles(mass0=xp.PROTON_MASS_EV, p0c=7000e9)

    collider.lhcb1.twiss_default["method"] = "4d"
    collider.lhcb2.twiss_default["method"] = "4d"
    collider.lhcb2.twiss_default["reverse"] = True

    collider.build_trackers()

    # Save collider to json
    os.makedirs("collider/collider_with_phase", exist_ok=True)

    for match_job, (tar_mux15, tar_muy15) in enumerate(itertools.product(muxs, muys)):
        print(f"Building collider {match_job}")
        collider_p = collider.copy()
        match_ip15_phase(collider_p, tar_mux15, tar_muy15, staged_match=True, solve=True)
        # collider_p.to_json(f'collider/collider_with_phase/collider_{tar_mux15:2.3f}_{tar_muy15:2.3f}.json')
        lm.gen_madx_optics_file_auto(
            collider_p, f"collider/collider_with_phase/opt_phase_ip15_{tar_mux15:2.3f}_{tar_muy15:2.3f}.madx"
        )


def build_distr_and_colliders(config_file="config.yaml"):
    configuration, config_particles, config_mad = load_configuration(config_file)

    muxs = config_mad["config_ip15_phase"]["muxs"]
    muys = config_mad["config_ip15_phase"]["muys"]
    
    # Get sanity checks flag
    # sanity_checks = configuration["sanity_checks"]
    sanity_checks = False # For now, optics files from xtrack missing some variables (e.g. qxBIM)

    # Tag start of the job
    tree_maker_tagging(configuration, tag="started")

    # Build particle distribution
    particle_list = build_particle_distribution(config_particles)

    # Write particle distribution to file
    write_particle_distribution(particle_list)
    
    os.makedirs("collider", exist_ok=True)
    for match_job, (tar_mux15, tar_muy15) in enumerate(itertools.product(muxs, muys)):
        optics_file = f"collider/collider_with_phase/opt_phase_ip15_{tar_mux15:2.3f}_{tar_muy15:2.3f}.madx"
        config_mad["optics_file"] = optics_file

        # Build collider from mad model
        collider = build_collider_from_mad(config_mad, sanity_checks)

        # Twiss to ensure eveyrthing is ok
        collider = activate_RF_and_twiss(collider, config_mad, sanity_checks)

        # Clean temporary files
        clean()

        # Save collider to json
        collider.to_json(f"collider/collider_phase_{tar_mux15:2.3f}_{tar_muy15:2.3f}.json")

    # Tag end of the job
    tree_maker_tagging(configuration, tag="completed")


# ==================================================================================================
# --- Script for execution
# ==================================================================================================

if __name__ == "__main__":
    build_phase_colliders()
    # build_distr_and_collider()
    build_distr_and_colliders()
