import numpy as np
import pytest

from cleandog_cad import design_space as ds
from cleandog_cad.hip_layout import HipLayout
from cleandog_cad.hip_layout import solve as occ_solve


@pytest.fixture(scope="module")
def ref():
    return ds.load_reference()


def test_ik_reproduces_the_reference_with_sim_geometry(ref):
    mo = ds.motion_for(ds.SIM_YA, ds.SIM_LEG, ref)
    q = ref.qpos[:, 7:19].reshape(-1, 4, 3).copy()
    q[:, [1, 3], 0] *= -1
    assert np.abs(mo.q_ik - q).max() < 1e-9
    assert not mo.infeasible.any()


def test_fk_matches_the_sim_model(ref):
    import mujoco

    from cleandog_cad.mj_check import RobotParams, StairParams  # sim package, via mj_check's path setup
    from cleandog_sim.scene import load_model

    # The reference was made with the M0 placeholder geometry, not today's sim defaults.
    m = load_model(RobotParams().with_(hip_width=2 * ds.SIM_YA / 1000, abad_offset=(ds.SIM_LEG - ds.SIM_YA) / 1000),
                   StairParams())
    d = mujoco.MjData(m)
    for fr in (0, 5000, 20000, 33333):
        d.qpos[:] = ref.qpos[fr]
        mujoco.mj_kinematics(m, d)
        R = d.xmat[m.body("trunk").id].reshape(3, 3)
        p0 = d.xpos[m.body("trunk").id]
        for i, leg in enumerate(("FL", "FR", "RL", "RR")):
            fb = R.T @ (d.site_xpos[m.site(f"{leg}_foot").id] - p0) * 1000
            fb[1] = abs(fb[1])
            assert fb == pytest.approx(ref.feet[fr, i], abs=1e-6)


def test_ab_ad_only_check_agrees_with_the_occ_hip_layout_study():
    """At the leg plane the OCC study finds for +-34.4 deg, the sampled ab/ad-only check also sits at
    the 3 mm clearance (both measure the same rolled stack against the same box trunk)."""
    occ = occ_solve(HipLayout(body_width=200, abad_range_deg=34.4))
    assert ds.static_roll_clearance(ds.BASE, occ.leg_plane) == pytest.approx(3.0, abs=0.6)


def test_sampled_and_mujoco_clearances_agree(ref):
    from cleandog_cad.mj_check import replay

    y = 190.0
    rp = replay(ds.BASE, y, ref, stride=40)
    pf = ds.sweep_clearance(ds.BASE, y, ref, per_frame=True, ik_only_frames=True).per_frame[rp.frames]
    near = (pf < ds.CULL) & (rp.trunk < 40)
    assert near.sum() > 50
    assert np.abs(pf[near] - rp.trunk[near]).max() < 1.0


def test_sim_model_matches_the_nk_design():
    """The sim's robot (cleandog_sim.robot, adopted N+K) against the CAD study's N+K geometry: for
    random front-left leg poses, MuJoCo's leg-to-trunk distance and the sampled CAD clearance agree."""
    import mujoco

    from cleandog_cad.mj_check import RobotParams
    from cleandog_sim.scene import build_scene_xml

    p = RobotParams()
    des = ds.with_(ds.NK, nose_half=p.trunk_nose_width / 2 * 1000)
    y_leg = p.leg_plane_y * 1000
    assert des.ya == pytest.approx(p.hip_width / 2 * 1000)
    assert des.notch_x == pytest.approx(p.trunk_notch_x * 1000)
    m = mujoco.MjModel.from_xml_string(build_scene_xml(p))
    d = mujoco.MjData(m)
    fl = [g for g in range(m.ngeom) if m.geom(g).name.startswith("FL_")
          and m.geom(g).name[3:] in ("thigh_hub", "thigh", "knee_pulley", "shin")]
    trunk = [g for g in range(m.ngeom) if m.geom(g).name.startswith(("trunk_shell", "skid"))]
    rng = np.random.default_rng(0)
    Q = np.c_[rng.uniform(-0.6, 0.6, 400), rng.uniform(-0.5, 2.6, 400), rng.uniform(-2.7, -0.2, 400)]
    cad = np.full(len(Q), np.inf)
    for frame, pts, inflate in ds.leg_parts(des, y_leg, 1).values():
        if frame != "stack":  # the stack's fixed 6 mm gap would mask the moving parts
            cad = np.minimum(cad, ds.part_clearance(des, frame, pts, inflate, Q, 1))
    ft = np.zeros(6)
    n = 0
    for q, c in zip(Q, cad):
        d.qpos[:] = 0
        d.qpos[2:4] = 1.0, 1.0  # trunk 1 m up, away from the floor
        d.qpos[7:10] = q
        mujoco.mj_kinematics(m, d)
        # Exact distances: contacts within a margin can miss positive box-box distances.
        sim = min(mujoco.mj_geomDistance(m, d, a, b, 0.04, ft) for a in fl for b in trunk) * 1000
        if c < 20:  # the CAD sweep reports only a lower bound above its 25 mm cull
            assert sim == pytest.approx(c, abs=0.5)
            n += 1
    assert n > 20
