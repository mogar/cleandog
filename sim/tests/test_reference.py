"""Round-trip reference and RL scene: fast checks (no planning, no training)."""

import math

import mujoco
import numpy as np
import pytest

from cleandog_sim import StairParams
from cleandog_sim.reference import (LandingCrawl, Reference, concat, hold, mj_to_pin, pin_to_mj,
                                    support_margin)
from cleandog_sim.transitions import morphology

P = morphology(0.16, "back", 0.25)


def test_support_margin_square():
    sq = [np.array(v) for v in ((0, 0), (1, 0), (1, 1), (0, 1))]
    assert support_margin(np.array([0.5, 0.5]), sq) == pytest.approx(0.5)
    assert support_margin(np.array([0.9, 0.5]), sq) == pytest.approx(0.1)
    assert support_margin(np.array([1.2, 0.5]), sq) == pytest.approx(-0.2)
    tri = sq[:3]  # order must not matter
    assert support_margin(np.array([0.9, 0.1]), tri[::-1]) == pytest.approx(0.1)


def test_quaternion_order_roundtrip():
    q = np.arange(21, dtype=float)
    assert np.allclose(mj_to_pin(pin_to_mj(q)), q)
    assert pin_to_mj(q)[3] == q[6]  # MuJoCo w first, Pinocchio w last


def test_round_trip_assembly_reverses_and_returns():
    q = np.random.default_rng(0).normal(size=(10, 21))
    up = Reference(0.02, q, np.ones((10, 4), bool), np.zeros(10, int), ["a"])
    trip = concat([hold(q[0], [True] * 4, 0.1, 0.02, "start"), up])
    trip = concat([trip, trip.reversed(lambda n: n + " down")])
    assert np.allclose(trip.qpos[0], trip.qpos[-1])  # a round trip ends where it starts
    assert trip.segment_names == ["start", "a", "a down", "start down"]


def test_landing_waypoints_end_facing_back():
    from cleandog_sim.dynamics import pin_model

    s = StairParams()
    crawl = LandingCrawl(P, s, pin_model(P))
    start, goal = (1.75, 0.0, 0.0), (1.85, 1.054, math.pi)
    pts = crawl.waypoints(start, goal)
    assert pts[0] == start
    assert np.allclose(pts[-1][:2], goal[:2]) and math.isclose(pts[-1][2] % (2 * math.pi), math.pi)
    steps = np.diff(np.array(pts), axis=0)
    assert np.hypot(steps[:, 0], steps[:, 1]).max() <= crawl.step_len + 1e-9
    assert np.abs(steps[:, 2]).max() <= crawl.step_yaw + 1e-9
    # Trunk centre stays on the landing, at least a turning radius from its walls.
    x_top = (s.flights[0].n_risers - 1) * s.run
    xs = np.array(pts)[:, 0]
    assert xs.min() >= x_top and xs.max() <= x_top + s.flights[0].top_depth


def test_rl_scene_has_sensors_and_matches_robot():
    pytest.importorskip("mujoco_playground")
    from cleandog_sim.rl import rl_scene_xml

    m = mujoco.MjModel.from_xml_string(rl_scene_xml(P, StairParams()))
    for leg in ("FL", "FR", "RL", "RR"):
        assert m.sensor(f"{leg}_touch").type == mujoco.mjtSensor.mjSENS_TOUCH
        assert m.sensor(f"{leg}_vel").dim == 3
    assert m.nu == 14 and m.opt.timestep == pytest.approx(0.004)


def test_foot_touch_sensor_reads_standing_weight():
    """Standing on the floor, the four foot touch zones carry the robot's weight."""
    pytest.importorskip("mujoco_playground")
    from cleandog_sim.rl import rl_scene_xml
    from cleandog_sim.robot import stand_pose

    m = mujoco.MjModel.from_xml_string(rl_scene_xml(P, None))
    d = mujoco.MjData(m)
    mujoco.mj_resetDataKeyframe(m, d, 0)
    kp = 60.0
    q_ref = np.array(list(stand_pose(P).values()))
    for _ in range(500):
        d.ctrl[:] = kp * (q_ref - d.qpos[7:]) - 1.0 * d.qvel[6:]
        mujoco.mj_step(m, d)
    touch = sum(d.sensordata[m.sensor_adr[m.sensor(f"{leg}_touch").id]] for leg in ("FL", "FR", "RL", "RR"))
    weight = m.body_subtreemass[m.body("trunk").id] * 9.81
    assert touch == pytest.approx(weight, rel=0.1)
