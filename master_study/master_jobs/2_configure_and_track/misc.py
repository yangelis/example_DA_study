# Imports
import json
import logging
from scipy.constants import c as clight
from scipy.optimize import minimize_scalar
import xtrack as xt
import xtrack._temp.lhc_match as lm
import numpy as np


# Function to generate dictionnary containing the orbit correction setup
def generate_orbit_correction_setup():
    correction_setup = {}
    correction_setup["lhcb1"] = {
        "IR1 left": dict(
            ref_with_knobs={"on_corr_co": 0, "on_disp": 0},
            start="e.ds.r8.b1",
            end="e.ds.l1.b1",
            vary=(
                "corr_co_acbh14.l1b1",
                "corr_co_acbh12.l1b1",
                "corr_co_acbv15.l1b1",
                "corr_co_acbv13.l1b1",
            ),
            targets=("e.ds.l1.b1",),
        ),
        "IR1 right": dict(
            ref_with_knobs={"on_corr_co": 0, "on_disp": 0},
            start="s.ds.r1.b1",
            end="s.ds.l2.b1",
            vary=(
                "corr_co_acbh13.r1b1",
                "corr_co_acbh15.r1b1",
                "corr_co_acbv12.r1b1",
                "corr_co_acbv14.r1b1",
            ),
            targets=("s.ds.l2.b1",),
        ),
        "IR5 left": dict(
            ref_with_knobs={"on_corr_co": 0, "on_disp": 0},
            start="e.ds.r4.b1",
            end="e.ds.l5.b1",
            vary=(
                "corr_co_acbh14.l5b1",
                "corr_co_acbh12.l5b1",
                "corr_co_acbv15.l5b1",
                "corr_co_acbv13.l5b1",
            ),
            targets=("e.ds.l5.b1",),
        ),
        "IR5 right": dict(
            ref_with_knobs={"on_corr_co": 0, "on_disp": 0},
            start="s.ds.r5.b1",
            end="s.ds.l6.b1",
            vary=(
                "corr_co_acbh13.r5b1",
                "corr_co_acbh15.r5b1",
                "corr_co_acbv12.r5b1",
                "corr_co_acbv14.r5b1",
            ),
            targets=("s.ds.l6.b1",),
        ),
        "IP1": dict(
            ref_with_knobs={"on_corr_co": 0, "on_disp": 0},
            start="e.ds.l1.b1",
            end="s.ds.r1.b1",
            vary=(
                "corr_co_acbch6.l1b1",
                "corr_co_acbcv5.l1b1",
                "corr_co_acbch5.r1b1",
                "corr_co_acbcv6.r1b1",
                "corr_co_acbyhs4.l1b1",
                "corr_co_acbyhs4.r1b1",
                "corr_co_acbyvs4.l1b1",
                "corr_co_acbyvs4.r1b1",
            ),
            targets=("ip1", "s.ds.r1.b1"),
        ),
        "IP2": dict(
            ref_with_knobs={"on_corr_co": 0, "on_disp": 0},
            start="e.ds.l2.b1",
            end="s.ds.r2.b1",
            vary=(
                "corr_co_acbyhs5.l2b1",
                "corr_co_acbchs5.r2b1",
                "corr_co_acbyvs5.l2b1",
                "corr_co_acbcvs5.r2b1",
                "corr_co_acbyhs4.l2b1",
                "corr_co_acbyhs4.r2b1",
                "corr_co_acbyvs4.l2b1",
                "corr_co_acbyvs4.r2b1",
            ),
            targets=("ip2", "s.ds.r2.b1"),
        ),
        "IP5": dict(
            ref_with_knobs={"on_corr_co": 0, "on_disp": 0},
            start="e.ds.l5.b1",
            end="s.ds.r5.b1",
            vary=(
                "corr_co_acbch6.l5b1",
                "corr_co_acbcv5.l5b1",
                "corr_co_acbch5.r5b1",
                "corr_co_acbcv6.r5b1",
                "corr_co_acbyhs4.l5b1",
                "corr_co_acbyhs4.r5b1",
                "corr_co_acbyvs4.l5b1",
                "corr_co_acbyvs4.r5b1",
            ),
            targets=("ip5", "s.ds.r5.b1"),
        ),
        "IP8": dict(
            ref_with_knobs={"on_corr_co": 0, "on_disp": 0},
            start="e.ds.l8.b1",
            end="s.ds.r8.b1",
            vary=(
                "corr_co_acbch5.l8b1",
                "corr_co_acbyhs4.l8b1",
                "corr_co_acbyhs4.r8b1",
                "corr_co_acbyhs5.r8b1",
                "corr_co_acbcvs5.l8b1",
                "corr_co_acbyvs4.l8b1",
                "corr_co_acbyvs4.r8b1",
                "corr_co_acbyvs5.r8b1",
            ),
            targets=("ip8", "s.ds.r8.b1"),
        ),
    }

    correction_setup["lhcb2"] = {
        "IR1 left": dict(
            ref_with_knobs={"on_corr_co": 0, "on_disp": 0},
            start="e.ds.l1.b2",
            end="e.ds.r8.b2",
            vary=(
                "corr_co_acbh13.l1b2",
                "corr_co_acbh15.l1b2",
                "corr_co_acbv12.l1b2",
                "corr_co_acbv14.l1b2",
            ),
            targets=("e.ds.r8.b2",),
        ),
        "IR1 right": dict(
            ref_with_knobs={"on_corr_co": 0, "on_disp": 0},
            start="s.ds.l2.b2",
            end="s.ds.r1.b2",
            vary=(
                "corr_co_acbh12.r1b2",
                "corr_co_acbh14.r1b2",
                "corr_co_acbv13.r1b2",
                "corr_co_acbv15.r1b2",
            ),
            targets=("s.ds.r1.b2",),
        ),
        "IR5 left": dict(
            ref_with_knobs={"on_corr_co": 0, "on_disp": 0},
            start="e.ds.l5.b2",
            end="e.ds.r4.b2",
            vary=(
                "corr_co_acbh13.l5b2",
                "corr_co_acbh15.l5b2",
                "corr_co_acbv12.l5b2",
                "corr_co_acbv14.l5b2",
            ),
            targets=("e.ds.r4.b2",),
        ),
        "IR5 right": dict(
            ref_with_knobs={"on_corr_co": 0, "on_disp": 0},
            start="s.ds.l6.b2",
            end="s.ds.r5.b2",
            vary=(
                "corr_co_acbh12.r5b2",
                "corr_co_acbh14.r5b2",
                "corr_co_acbv13.r5b2",
                "corr_co_acbv15.r5b2",
            ),
            targets=("s.ds.r5.b2",),
        ),
        "IP1": dict(
            ref_with_knobs={"on_corr_co": 0, "on_disp": 0},
            start="s.ds.r1.b2",
            end="e.ds.l1.b2",
            vary=(
                "corr_co_acbch6.r1b2",
                "corr_co_acbcv5.r1b2",
                "corr_co_acbch5.l1b2",
                "corr_co_acbcv6.l1b2",
                "corr_co_acbyhs4.l1b2",
                "corr_co_acbyhs4.r1b2",
                "corr_co_acbyvs4.l1b2",
                "corr_co_acbyvs4.r1b2",
            ),
            targets=(
                "ip1",
                "e.ds.l1.b2",
            ),
        ),
        "IP2": dict(
            ref_with_knobs={"on_corr_co": 0, "on_disp": 0},
            start="s.ds.r2.b2",
            end="e.ds.l2.b2",
            vary=(
                "corr_co_acbyhs5.l2b2",
                "corr_co_acbchs5.r2b2",
                "corr_co_acbyvs5.l2b2",
                "corr_co_acbcvs5.r2b2",
                "corr_co_acbyhs4.l2b2",
                "corr_co_acbyhs4.r2b2",
                "corr_co_acbyvs4.l2b2",
                "corr_co_acbyvs4.r2b2",
            ),
            targets=("ip2", "e.ds.l2.b2"),
        ),
        "IP5": dict(
            ref_with_knobs={"on_corr_co": 0, "on_disp": 0},
            start="s.ds.r5.b2",
            end="e.ds.l5.b2",
            vary=(
                "corr_co_acbch6.r5b2",
                "corr_co_acbcv5.r5b2",
                "corr_co_acbch5.l5b2",
                "corr_co_acbcv6.l5b2",
                "corr_co_acbyhs4.l5b2",
                "corr_co_acbyhs4.r5b2",
                "corr_co_acbyvs4.l5b2",
                "corr_co_acbyvs4.r5b2",
            ),
            targets=(
                "ip5",
                "e.ds.l5.b2",
            ),
        ),
        "IP8": dict(
            ref_with_knobs={"on_corr_co": 0, "on_disp": 0},
            start="s.ds.r8.b2",
            end="e.ds.l8.b2",
            vary=(
                "corr_co_acbchs5.l8b2",
                "corr_co_acbyhs5.r8b2",
                "corr_co_acbcvs5.l8b2",
                "corr_co_acbyvs5.r8b2",
                "corr_co_acbyhs4.l8b2",
                "corr_co_acbyhs4.r8b2",
                "corr_co_acbyvs4.l8b2",
                "corr_co_acbyvs4.r8b2",
            ),
            targets=(
                "ip8",
                "e.ds.l8.b2",
            ),
        ),
    }
    return correction_setup


def luminosity_leveling(
    collider,
    config_lumi_leveling,
    config_beambeam,
    additional_targets_lumi=[],
    crab=False,
):
    for ip_name in config_lumi_leveling.keys():
        print(f"\n --- Leveling in {ip_name} ---")

        config_this_ip = config_lumi_leveling[ip_name]
        bump_range = config_this_ip["bump_range"]

        assert config_this_ip[
            "preserve_angles_at_ip"
        ], "Only preserve_angles_at_ip=True is supported for now"
        assert config_this_ip[
            "preserve_bump_closure"
        ], "Only preserve_bump_closure=True is supported for now"

        beta0_b1 = collider.lhcb1.particle_ref.beta0[0]
        f_rev = 1 / (collider.lhcb1.get_length() / (beta0_b1 * clight))

        targets = []
        vary = []

        if "luminosity" in config_this_ip.keys():
            targets.append(
                xt.TargetLuminosity(
                    ip_name=ip_name,
                    luminosity=config_this_ip["luminosity"],
                    crab=crab,
                    tol=1e30,  # 0.01 * config_this_ip["luminosity"],
                    f_rev=f_rev,
                    num_colliding_bunches=config_this_ip["num_colliding_bunches"],
                    num_particles_per_bunch=config_beambeam["num_particles_per_bunch"],
                    sigma_z=config_beambeam["sigma_z"],
                    nemitt_x=config_beambeam["nemitt_x"],
                    nemitt_y=config_beambeam["nemitt_y"],
                    log=True,
                )
            )

            # Added this line for constraints
            targets.extend(additional_targets_lumi)
        elif "separation_in_sigmas" in config_this_ip.keys():
            targets.append(
                xt.TargetSeparation(
                    ip_name=ip_name,
                    separation_norm=config_this_ip["separation_in_sigmas"],
                    tol=1e-4,  # in sigmas
                    plane=config_this_ip["plane"],
                    nemitt_x=config_beambeam["nemitt_x"],
                    nemitt_y=config_beambeam["nemitt_y"],
                )
            )
        else:
            raise ValueError("Either `luminosity` or `separation_in_sigmas` must be specified")

        if config_this_ip["impose_separation_orthogonal_to_crossing"]:
            targets.append(xt.TargetSeparationOrthogonalToCrossing(ip_name="ip8"))
        vary.append(xt.VaryList(config_this_ip["knobs"], step=1e-4))

        # Target and knobs to rematch the crossing angles and close the bumps
        for line_name in ["lhcb1", "lhcb2"]:
            targets += [
                # Preserve crossing angle
                xt.TargetList(
                    ["px", "py"], at=ip_name, line=line_name, value="preserve", tol=1e-7, scale=1e3
                ),
                # Close the bumps
                xt.TargetList(
                    ["x", "y"],
                    at=bump_range[line_name][-1],
                    line=line_name,
                    value="preserve",
                    tol=1e-5,
                    scale=1,
                ),
                xt.TargetList(
                    ["px", "py"],
                    at=bump_range[line_name][-1],
                    line=line_name,
                    value="preserve",
                    tol=1e-5,
                    scale=1e3,
                ),
            ]

        vary.append(xt.VaryList(config_this_ip["corrector_knob_names"], step=1e-7))

        # Match
        collider.match(
            lines=["lhcb1", "lhcb2"],
            ele_start=[bump_range["lhcb1"][0], bump_range["lhcb2"][0]],
            ele_stop=[bump_range["lhcb1"][-1], bump_range["lhcb2"][-1]],
            twiss_init="preserve",
            targets=targets,
            vary=vary,
        )
        
    return collider


def compute_PU(luminosity, num_colliding_bunches, T_rev0, cross_section = 81e-27):
    return (
            luminosity
            / num_colliding_bunches
            * cross_section
            * T_rev0
        )

def luminosity_leveling_ip1_5(
    collider,
    config_collider,
    config_bb,
    crab=False,
):
    # Get Twiss
    twiss_b1 = collider["lhcb1"].twiss()
    twiss_b2 = collider["lhcb2"].twiss()

    def compute_lumi(I):
        luminosity = xt.lumi.luminosity_from_twiss(
            n_colliding_bunches=config_collider["config_lumi_leveling_ip1_5"][
                "num_colliding_bunches"
            ],
            num_particles_per_bunch=I,
            ip_name="ip1",
            nemitt_x=config_bb["nemitt_x"],
            nemitt_y=config_bb["nemitt_y"],
            sigma_z=config_bb["sigma_z"],
            twiss_b1=twiss_b1,
            twiss_b2=twiss_b2,
            crab=crab,
        )
        return luminosity

    def f(I):
        luminosity = compute_lumi(I)

        PU = compute_PU(luminosity, config_collider["config_lumi_leveling_ip1_5"]["num_colliding_bunches"], twiss_b1["T_rev0"])
        penalty_PU = max(
            0,
            (PU - config_collider["config_lumi_leveling_ip1_5"]["constraints"]["max_PU"]) * 1e35,
        )

        return (
            abs(luminosity - config_collider["config_lumi_leveling_ip1_5"]["luminosity"])
            + penalty_PU
        )

    # Do the optimization
    res = minimize_scalar(
        f,
        bounds=(
            1e10,
            float(config_collider["config_lumi_leveling_ip1_5"]["constraints"]["max_intensity"]),
        ),
        method="bounded",
        options={"xatol": 1e7},
    )
    if not res.success:
        logging.warning("Optimization for leveling in IP 1/5 failed. Please check the constraints.")
    else:
        print(
            f"Optimization for leveling in IP 1/5 succeeded with I={res.x:.2e} particles per bunch"
        )
    return res.x


def change_ip15_phase(
    collider,
    dqx,
    dqy,
    d_mux_15_b1,
    d_muy_15_b1,
    d_mux_15_b2,
    d_muy_15_b2,
    staged_match=False,
    solve=False,
):
    default_tol = {
        None: 1e-8,
        "betx": 1e-6,
        "bety": 1e-6,
    }

    optimizers = {}

    print("Matching ip15 phase:")
    opt = lm.change_phase_non_ats_arcs(
        collider,
        d_mux_15_b1=d_mux_15_b1,
        d_muy_15_b1=d_muy_15_b1,
        d_mux_15_b2=d_mux_15_b2,
        d_muy_15_b2=d_muy_15_b2,
        solve=True,
        default_tol=default_tol,
    )
    optimizers["phase_15"] = opt
    arc_periodic_solution = lm.get_arc_periodic_solution(collider)

    optimizers.update({"b1": {}, "b2": {}})
    for bn in ["b1", "b2"]:
        line_name = f"lhc{bn}"

        muxip1_l = collider.varval[f"muxip1{bn}_l"]
        muyip1_l = collider.varval[f"muyip1{bn}_l"]
        muxip1_r = collider.varval[f"muxip1{bn}_r"]
        muyip1_r = collider.varval[f"muyip1{bn}_r"]

        muxip5_l = collider.varval[f"muxip5{bn}_l"]
        muyip5_l = collider.varval[f"muyip5{bn}_l"]
        muxip5_r = collider.varval[f"muxip5{bn}_r"]
        muyip5_r = collider.varval[f"muyip5{bn}_r"]

        muxip2 = collider.varval[f"muxip2{bn}"]
        muyip2 = collider.varval[f"muyip2{bn}"]
        muxip4 = collider.varval[f"muxip4{bn}"]
        muyip4 = collider.varval[f"muyip4{bn}"]
        muxip6 = collider.varval[f"muxip6{bn}"]
        muyip6 = collider.varval[f"muyip6{bn}"]
        muxip8 = collider.varval[f"muxip8{bn}"]
        muyip8 = collider.varval[f"muyip8{bn}"]

        mux12 = collider.varval[f"mux12{bn}"]
        muy12 = collider.varval[f"muy12{bn}"]
        mux45 = collider.varval[f"mux45{bn}"]
        muy45 = collider.varval[f"muy45{bn}"]
        mux56 = collider.varval[f"mux56{bn}"]
        muy56 = collider.varval[f"muy56{bn}"]
        mux81 = collider.varval[f"mux81{bn}"]
        muy81 = collider.varval[f"muy81{bn}"]

        betx_ip1 = collider.varval[f"betxip1{bn}"]
        bety_ip1 = collider.varval[f"betyip1{bn}"]
        betx_ip5 = collider.varval[f"betxip5{bn}"]
        bety_ip5 = collider.varval[f"betyip5{bn}"]

        betx_ip2 = collider.varval[f"betxip2{bn}"]
        bety_ip2 = collider.varval[f"betyip2{bn}"]

        alfx_ip3 = collider.varval[f"alfxip3{bn}"]
        alfy_ip3 = collider.varval[f"alfyip3{bn}"]
        betx_ip3 = collider.varval[f"betxip3{bn}"]
        bety_ip3 = collider.varval[f"betyip3{bn}"]
        dx_ip3 = collider.varval[f"dxip3{bn}"]
        dpx_ip3 = collider.varval[f"dpxip3{bn}"]
        mux_ir3 = collider.varval[f"muxip3{bn}"]
        muy_ir3 = collider.varval[f"muyip3{bn}"]

        alfx_ip4 = collider.varval[f"alfxip4{bn}"]
        alfy_ip4 = collider.varval[f"alfyip4{bn}"]
        betx_ip4 = collider.varval[f"betxip4{bn}"]
        bety_ip4 = collider.varval[f"betyip4{bn}"]
        dx_ip4 = collider.varval[f"dxip4{bn}"]
        dpx_ip4 = collider.varval[f"dpxip4{bn}"]

        alfx_ip6 = collider.varval[f"alfxip6{bn}"]
        alfy_ip6 = collider.varval[f"alfyip6{bn}"]
        betx_ip6 = collider.varval[f"betxip6{bn}"]
        bety_ip6 = collider.varval[f"betyip6{bn}"]
        dx_ip6 = collider.varval[f"dxip6{bn}"]
        dpx_ip6 = collider.varval[f"dpxip6{bn}"]

        alfx_ip7 = collider.varval[f"alfxip7{bn}"]
        alfy_ip7 = collider.varval[f"alfyip7{bn}"]
        betx_ip7 = collider.varval[f"betxip7{bn}"]
        bety_ip7 = collider.varval[f"betyip7{bn}"]
        dx_ip7 = collider.varval[f"dxip7{bn}"]
        dpx_ip7 = collider.varval[f"dpxip7{bn}"]
        mux_ir7 = collider.varval[f"muxip7{bn}"]
        muy_ir7 = collider.varval[f"muyip7{bn}"]

        alfx_ip8 = collider.varval[f"alfxip8{bn}"]
        alfy_ip8 = collider.varval[f"alfyip8{bn}"]
        betx_ip8 = collider.varval[f"betxip8{bn}"]
        bety_ip8 = collider.varval[f"betyip8{bn}"]
        dx_ip8 = collider.varval[f"dxip8{bn}"]
        dpx_ip8 = collider.varval[f"dpxip8{bn}"]

        tw_sq_a81_ip1_a12 = lm.propagate_optics_from_beta_star(
            collider,
            ip_name="ip1",
            line_name=f"lhc{bn}",
            ele_start=f"s.ds.r8.{bn}",
            ele_stop=f"e.ds.l2.{bn}",
            beta_star_x=betx_ip1,
            beta_star_y=bety_ip1,
        )

        tw_sq_a45_ip5_a56 = lm.propagate_optics_from_beta_star(
            collider,
            ip_name="ip5",
            line_name=f"lhc{bn}",
            ele_start=f"s.ds.r4.{bn}",
            ele_stop=f"e.ds.l6.{bn}",
            beta_star_x=betx_ip5,
            beta_star_y=bety_ip5,
        )

        (
            mux_ir2_target,
            muy_ir2_target,
            mux_ir4_target,
            muy_ir4_target,
            mux_ir6_target,
            muy_ir6_target,
            mux_ir8_target,
            muy_ir8_target,
        ) = lm.compute_ats_phase_advances_for_auxiliary_irs(
            line_name,
            tw_sq_a81_ip1_a12,
            tw_sq_a45_ip5_a56,
            muxip1_l,
            muyip1_l,
            muxip1_r,
            muyip1_r,
            muxip5_l,
            muyip5_l,
            muxip5_r,
            muyip5_r,
            muxip2,
            muyip2,
            muxip4,
            muyip4,
            muxip6,
            muyip6,
            muxip8,
            muyip8,
            mux12,
            muy12,
            mux45,
            muy45,
            mux56,
            muy56,
            mux81,
            muy81,
        )

        print(f"Matching IR2 {bn}")
        opt = lm.rematch_ir2(
            collider,
            line_name=f"lhc{bn}",
            boundary_conditions_left=tw_sq_a81_ip1_a12,
            boundary_conditions_right=arc_periodic_solution[f"lhc{bn}"]["23"],
            mux_ir2=mux_ir2_target,
            muy_ir2=muy_ir2_target,
            betx_ip2=betx_ip2,
            bety_ip2=bety_ip2,
            solve=solve,
            staged_match=staged_match,
            default_tol=default_tol,
        )
        optimizers[bn]["ir2"] = opt

        print(f"Matching IR3 {bn}")
        opt = lm.rematch_ir3(
            collider=collider,
            line_name=f"lhc{bn}",
            boundary_conditions_left=arc_periodic_solution[f"lhc{bn}"]["23"],
            boundary_conditions_right=arc_periodic_solution[f"lhc{bn}"]["34"],
            mux_ir3=mux_ir3,
            muy_ir3=muy_ir3,
            alfx_ip3=alfx_ip3,
            alfy_ip3=alfy_ip3,
            betx_ip3=betx_ip3,
            bety_ip3=bety_ip3,
            dx_ip3=dx_ip3,
            dpx_ip3=dpx_ip3,
            solve=solve,
            staged_match=staged_match,
            default_tol=default_tol,
        )
        optimizers[bn]["ir3"] = opt

        print(f"Matching IR4 {bn}")
        opt = lm.rematch_ir4(
            collider=collider,
            line_name=f"lhc{bn}",
            boundary_conditions_left=arc_periodic_solution[f"lhc{bn}"]["34"],
            boundary_conditions_right=tw_sq_a45_ip5_a56,
            mux_ir4=mux_ir4_target,
            muy_ir4=muy_ir4_target,
            alfx_ip4=alfx_ip4,
            alfy_ip4=alfy_ip4,
            betx_ip4=betx_ip4,
            bety_ip4=bety_ip4,
            dx_ip4=dx_ip4,
            dpx_ip4=dpx_ip4,
            solve=solve,
            staged_match=staged_match,
            default_tol=default_tol,
        )
        optimizers[bn]["ir4"] = opt

        print(f"Matching IP6 {bn}")
        opt = lm.rematch_ir6(
            collider=collider,
            line_name=f"lhc{bn}",
            boundary_conditions_left=tw_sq_a45_ip5_a56,
            boundary_conditions_right=arc_periodic_solution[f"lhc{bn}"]["67"],
            mux_ir6=mux_ir6_target,
            muy_ir6=muy_ir6_target,
            alfx_ip6=alfx_ip6,
            alfy_ip6=alfy_ip6,
            betx_ip6=betx_ip6,
            bety_ip6=bety_ip6,
            dx_ip6=dx_ip6,
            dpx_ip6=dpx_ip6,
            solve=solve,
            staged_match=staged_match,
            default_tol=default_tol,
        )
        optimizers[bn]["ir6"] = opt

        print(f"Matching IP7 {bn}")
        opt = lm.rematch_ir7(
            collider=collider,
            line_name=f"lhc{bn}",
            boundary_conditions_left=arc_periodic_solution[f"lhc{bn}"]["67"],
            boundary_conditions_right=arc_periodic_solution[f"lhc{bn}"]["78"],
            mux_ir7=mux_ir7,
            muy_ir7=muy_ir7,
            alfx_ip7=alfx_ip7,
            alfy_ip7=alfy_ip7,
            betx_ip7=betx_ip7,
            bety_ip7=bety_ip7,
            dx_ip7=dx_ip7,
            dpx_ip7=dpx_ip7,
            solve=solve,
            staged_match=staged_match,
            default_tol=default_tol,
        )
        optimizers[bn]["ir7"] = opt

        print(f"Matching IP8 {bn}")
        opt = lm.rematch_ir8(
            collider=collider,
            line_name=f"lhc{bn}",
            boundary_conditions_left=arc_periodic_solution[f"lhc{bn}"]["78"],
            boundary_conditions_right=tw_sq_a81_ip1_a12,
            mux_ir8=mux_ir8_target,
            muy_ir8=muy_ir8_target,
            alfx_ip8=alfx_ip8,
            alfy_ip8=alfy_ip8,
            betx_ip8=betx_ip8,
            bety_ip8=bety_ip8,
            dx_ip8=dx_ip8,
            dpx_ip8=dpx_ip8,
            solve=solve,
            staged_match=staged_match,
            default_tol=default_tol,
        )
        optimizers[bn]["ir8"] = opt

    opt = lm.match_orbit_knobs_ip2_ip8(collider)
    optimizers["orbit_knobs"] = opt

    collider.to_json(f"phase_collider/collider_ip15_{dqx:2.5f}_{dqy:2.5f}.json")
    lm.gen_madx_optics_file_auto(
        collider, f"phase_collider/opt_phase_ip15_{dqx:2.5f}_{dqy:2.5f}.madx"
    )

    return optimizers


if __name__ == "__main__":
    correction_setup = generate_orbit_correction_setup()
    for nn in ["lhcb1", "lhcb2"]:
        with open(f"corr_co_{nn}.json", "w") as fid:
            json.dump(correction_setup[nn], fid, indent=4)
