import mujoco
import numpy as np
import pytest

from cleandog_sim import RobotParams, StairParams, load_model
from cleandog_sim.ladder import pocket_targets, stance_stroke
from cleandog_sim.robot import LEGS, stand_height


@pytest.mark.parametrize("turn_left", [True, False])
def test_stairwell_compiles(turn_left):
    m = load_model(RobotParams(), StairParams(turn_left=turn_left))
    assert m.nu == 14
    assert m.geom("partition").size[1] * 2 == pytest.approx(StairParams().partition_thickness)


def test_single_flight_compiles():
    assert load_model(RobotParams(), StairParams().single_flight()).nu == 14


def test_upper_floor_height():
    """The top tread of the upper flight sits at the full stairwell rise."""
    s = StairParams()
    m = load_model(RobotParams(), s)
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    tops = d.geom_xpos[:, 2] + m.geom_size[:, 2] * (m.geom_type == mujoco.mjtGeom.mjGEOM_BOX)
    wood = [i for i in range(m.ngeom) if m.geom_matid[i] == m.material("wood").id]
    assert max(tops[wood]) == pytest.approx(s.total_rise, abs=1e-6)


def test_ladder_pose_matches_mujoco():
    """The 2D analysis and the generated MJCF agree on where the feet are."""
    from cleandog_sim.ladder import ladder_pose, ladder_qpos

    s = StairParams()
    for knee in (-1, 1):
        p = RobotParams(knee_sign_front=knee, knee_sign_rear=knee)
        pose = ladder_pose(p, s, 3.5 * s.nosing_pitch, 0.03)
        assert all(pose.ok.values())
        m = load_model(p, s)
        d = mujoco.MjData(m)
        d.qpos[:] = ladder_qpos(p, pose)
        mujoco.mj_forward(m, d)
        for leg in LEGS:
            np.testing.assert_allclose(d.site(f"{leg}_foot").xpos[[0, 2]], pose.feet[leg[0]], atol=1e-9)


def test_robot_mass_matches_target():
    p = RobotParams(total_mass_target=9.0)
    m = load_model(p)
    assert m.body_subtreemass[m.body("trunk").id] == pytest.approx(9.0)


@pytest.mark.parametrize("knee_signs", [(-1, -1), (1, 1), (1, -1)])
def test_stand_pose_puts_feet_under_hips(knee_signs):
    p = RobotParams(knee_sign_front=knee_signs[0], knee_sign_rear=knee_signs[1])
    m = load_model(p)
    d = mujoco.MjData(m)
    mujoco.mj_resetDataKeyframe(m, d, 0)
    mujoco.mj_forward(m, d)
    for leg in LEGS:
        foot = d.site(f"{leg}_foot").xpos
        hip = d.body(f"{leg}_thigh").xpos
        assert foot[0] == pytest.approx(hip[0], abs=1e-6)
        assert foot[2] == pytest.approx(p.foot_radius, abs=1e-6)


def test_stands_under_pd_hold():
    p = RobotParams()
    m = load_model(p)
    d = mujoco.MjData(m)
    mujoco.mj_resetDataKeyframe(m, d, 0)
    q0 = d.qpos[7:].copy()
    for _ in range(1500):
        d.ctrl[:] = 40.0 * (q0 - d.qpos[7:]) - 1.0 * d.qvel[6:]
        mujoco.mj_step(m, d)
    assert d.qpos[2] == pytest.approx(stand_height(p), abs=0.02)


def test_stance_stroke_scales_with_leg_length():
    s = StairParams()
    short = RobotParams(thigh_length=0.08, shin_length=0.08)
    plan = RobotParams(thigh_length=0.13, shin_length=0.13)
    corner_short = pocket_targets(s, 4, short.foot_radius)[0]
    corner_plan = pocket_targets(s, 4, plan.foot_radius)[0]
    assert stance_stroke(short, s, 0.06, corner_short, knee_back=True).stroke == 0.0
    assert stance_stroke(plan, s, 0.0, corner_plan, knee_back=True).stroke > 0.15


def test_links_have_thickness():
    """A knee touching the riser face is rejected once link radius counts."""
    from cleandog_sim.ladder import leg_ik

    s = StairParams()
    p = RobotParams(thigh_length=0.2, shin_length=0.2)
    hip = np.array([0.675, 0.934])
    foot = np.array([0.673, 0.808])
    assert not leg_ik(p, s, hip[None], foot[None], knee_back=False).ok[0]
    assert leg_ik(p.with_(thigh_radius=0.0, shin_radius=0.0), s, hip[None], foot[None], knee_back=False).ok[0]


def _contacts(m, d):
    return {frozenset((m.geom(c.geom1).name, m.geom(c.geom2).name)) for c in d.contact[:d.ncon]}


def test_legs_and_hip_stacks_collide_with_the_trunk():
    """Self-collision is on for leg vs trunk (the robot can't pass a leg through itself) and off
    between a leg and its own hip stack, which overlap by construction."""
    m = load_model(RobotParams())
    d = mujoco.MjData(m)
    mujoco.mj_resetDataKeyframe(m, d, 0)
    d.qpos[2] += 0.5
    mujoco.mj_forward(m, d)
    assert not _contacts(m, d)  # standing: nothing touches
    d.qpos[7:10] = (-0.6, 2.5, -2.2)  # FL folded back and tucked in: the foot meets the skid plate
    mujoco.mj_forward(m, d)
    assert frozenset(("FL_foot", "skid")) in _contacts(m, d)
    d.qpos[7:10] = (-1.2, 0.0, -0.5)  # past the ab/ad range: the hip motor meets the nose
    mujoco.mj_forward(m, d)
    assert frozenset(("FL_hip_motor", "trunk_shell_nose")) in _contacts(m, d)


def test_clearance_of_the_stand_pose():
    from cleandog_sim.reference import Clearance

    p = RobotParams()
    m = load_model(p, StairParams())  # keyframe: standing on the floor before the first riser
    d = mujoco.MjData(m)
    mujoco.mj_resetDataKeyframe(m, d, 0)
    c = Clearance(p, StairParams())(d.qpos)
    # Nearest leg part to the trunk: the hip motor, 6 mm from the centre section's end wall.
    assert c["self"] == pytest.approx(6.0, abs=0.05)
    assert c["stack"] == pytest.approx(Clearance.MARGIN * 1000)


def test_notched_planform():
    p = RobotParams()
    m = load_model(p)
    assert m.geom("trunk_shell").size[:2] == pytest.approx((p.trunk_notch_x, p.body_width / 2))
    assert m.geom("trunk_shell_nose").size[1] == pytest.approx(p.trunk_nose_width / 2)
    assert p.overall_width == pytest.approx(0.304)
    box = load_model(p.with_(trunk_notch_x=p.body_length / 2))
    assert box.geom("trunk_shell").size[:2] == pytest.approx((p.body_length / 2, p.body_width / 2))
