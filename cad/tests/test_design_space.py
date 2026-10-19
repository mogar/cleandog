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

    m = load_model(RobotParams(), StairParams())
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
