from xmask.lhc import install_errors_placeholders_hllhc
import numpy as np
import xtrack as xt
import xtrack._temp.lhc_match as lm

def check_madx_lattices(mad):
    assert mad.globals["qxb1"] == mad.globals["qxb2"]
    assert mad.globals["qyb1"] == mad.globals["qyb2"]
    assert mad.globals["qpxb1"] == mad.globals["qpxb2"]
    assert mad.globals["qpyb1"] == mad.globals["qpyb2"]

    assert np.isclose(mad.table.summ.q1, mad.globals["qxb1"], atol=1e-02)
    assert np.isclose(mad.table.summ.q2, mad.globals["qyb1"], atol=1e-02)
    assert np.isclose(mad.table.summ.dq1, mad.globals["qpxb1"], atol=1e-01)
    assert np.isclose(mad.table.summ.dq2, mad.globals["qpyb1"], atol=1e-01)

    df = mad.table.twiss.dframe()
    for my_ip in [1, 2, 5, 8]:
        assert np.isclose(df.loc[f"ip{my_ip}"].betx, mad.globals[f"betx_IP{my_ip}"], rtol=1e-02)
        assert np.isclose(df.loc[f"ip{my_ip}"].bety, mad.globals[f"bety_IP{my_ip}"], rtol=1e-02)

    assert df["x"].std() < 1e-6
    assert df["y"].std() < 1e-6


def check_xsuite_lattices(my_line):
    tw = my_line.twiss(method="6d", matrix_stability_tol=100)
    print(f"--- Now displaying Twiss result at all IPS for line {my_line}---")
    print(tw[:, "ip.*"])
    # print qx and qy
    print(f"--- Now displaying Qx and Qy for line {my_line}---")
    print(tw.qx, tw.qy)


def build_sequence(
    mad,
    mylhcbeam,
    apply_fix=True,
    ignore_cycling=False,
    ignore_CC=True,
):
    # Select beam
    mad.input(f"mylhcbeam = {mylhcbeam}")

    # Build sequence
    mad.input(f"""
      ! Build sequence
      option, -echo,-warn,-info;
      if (mylhcbeam==4){{
        call,file="acc-models-lhc/lhcb4.seq";
      }} else {{
        call,file="acc-models-lhc/lhc.seq";
      }};
      !Install HL-LHC
      call, file=
        "acc-models-lhc/hllhc_sequence.madx";
      ! Get the toolkit
      call,file=
        "acc-models-lhc/toolkit/macro.madx";
      option, -echo, warn,-info;
      """)

    if apply_fix:
        mad.input("""
        l.mbh = 0.001000;
        ACSCA, HARMON := HRF400;
        
        ACSCA.D5L4.B1, VOLT := VRF400/8, LAG := LAGRF400.B1, HARMON := HRF400;
        ACSCA.C5L4.B1, VOLT := VRF400/8, LAG := LAGRF400.B1, HARMON := HRF400;
        ACSCA.B5L4.B1, VOLT := VRF400/8, LAG := LAGRF400.B1, HARMON := HRF400;
        ACSCA.A5L4.B1, VOLT := VRF400/8, LAG := LAGRF400.B1, HARMON := HRF400;
        ACSCA.A5R4.B1, VOLT := VRF400/8, LAG := LAGRF400.B1, HARMON := HRF400;
        ACSCA.B5R4.B1, VOLT := VRF400/8, LAG := LAGRF400.B1, HARMON := HRF400;
        ACSCA.C5R4.B1, VOLT := VRF400/8, LAG := LAGRF400.B1, HARMON := HRF400;
        ACSCA.D5R4.B1, VOLT := VRF400/8, LAG := LAGRF400.B1, HARMON := HRF400;
        ACSCA.D5L4.B2, VOLT := VRF400/8, LAG := LAGRF400.B2, HARMON := HRF400;
        ACSCA.C5L4.B2, VOLT := VRF400/8, LAG := LAGRF400.B2, HARMON := HRF400;
        ACSCA.B5L4.B2, VOLT := VRF400/8, LAG := LAGRF400.B2, HARMON := HRF400;
        ACSCA.A5L4.B2, VOLT := VRF400/8, LAG := LAGRF400.B2, HARMON := HRF400;
        ACSCA.A5R4.B2, VOLT := VRF400/8, LAG := LAGRF400.B2, HARMON := HRF400;
        ACSCA.B5R4.B2, VOLT := VRF400/8, LAG := LAGRF400.B2, HARMON := HRF400;
        ACSCA.C5R4.B2, VOLT := VRF400/8, LAG := LAGRF400.B2, HARMON := HRF400;
        ACSCA.D5R4.B2, VOLT := VRF400/8, LAG := LAGRF400.B2, HARMON := HRF400;
        """)

    mad.input("""
      ! Slice nominal sequence
      exec, myslice;
      """)

    if mylhcbeam < 3:
        mad.input(f"""
      nrj=7000;
      beam,particle=proton,sequence=lhcb1,energy=nrj,npart=1.15E11,sige=4.5e-4;
      beam,particle=proton,sequence=lhcb2,energy=nrj,bv = -1,npart=1.15E11,sige=4.5e-4;
      """)

    install_errors_placeholders_hllhc(mad)

    if not ignore_cycling:
        mad.input("""
        !Cycling w.r.t. to IP3 (mandatory to find closed orbit in collision in the presence of errors)
        if (mylhcbeam<3){
        seqedit, sequence=lhcb1; flatten; cycle, start=IP3; flatten; endedit;
        };
        seqedit, sequence=lhcb2; flatten; cycle, start=IP3; flatten; endedit;
        """)

    # Force ignore CC for 1.6 for now
    if not ignore_CC:
        mad.input("""
        ! Install crab cavities (they are off)
        call, file='acc-models-lhc/toolkit/enable_crabcavities.madx';
        on_crab1 = 0;
        on_crab5 = 0;
        """)

    mad.input("""
        ! Set twiss formats for MAD-X parts (macro from opt. toolkit)
        exec, twiss_opt;
        """)


def apply_optics(mad, optics_file):
    mad.call(optics_file)
    # A knob redefinition
    mad.input("on_alice := on_alice_normalized * 7000./nrj;")
    mad.input("on_lhcb := on_lhcb_normalized * 7000./nrj;")

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

    return optimizers