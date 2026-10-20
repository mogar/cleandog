import numpy as np
import pinocchio as pin
import pytest

from cleandog_sim import RobotParams, StairParams
from cleandog_sim import dynamics as D
from cleandog_sim.render import planner_to_world_qpos
from cleandog_sim.stairs import Rect
from cleandog_sim.terrain import Surface, Terrain
from cleandog_sim.transitions import Planner, Scenario


@pytest.fixture(scope="module")
def flat_plan():
    p = RobotParams()
    t = Terrain([Rect(-3, 3, -0.05, 0)], [Surface(-3, 3, 0, 0)], None)
    pl = Planner(p, Scenario("lower", "bottom", "up", t, 8, -1.0, (0,), (0,)))
    path, status = pl.plan(goal=lambda st: st[0] > -0.85, time_limit=60)
    assert status == "ok"
    return p, pl, path


def test_reference_feet_hit_footholds(flat_plan):
    p, pl, path = flat_plan
    model = D.pin_model(p)
    data = model.createData()
    for ph in D.plan_phases(pl, path):
        # Without the planner's sideways sway: a full 5 cm sway can be out of 3D reach where a
        # hip is near its limit, which reference.flight_reference handles by scaling the sway.
        base = ph.base_end.copy()
        base[1] = 0.0
        q = D.reference_q(p, model, base, ph.feet)
        pin.framesForwardKinematics(model, data, q)
        for leg in ph.stance:
            err = data.oMf[model.getFrameId(f"{leg}_foot")].translation - ph.feet[leg]
            assert np.abs(err).max() < 2e-3


def test_solve_flat_walk(flat_plan):
    p, pl, path = flat_plan
    r = D.solve(p, pl, path, max_iter=100)
    assert r.converged
    assert r.friction_used.max() <= 0.5 + 1e-3
    assert np.abs(r.tau[:, :12]).max() < 17.0
    assert r.foot_err < 0.01


def test_mirror_mapping_swaps_legs_and_keeps_pitch_joints():
    p = RobotParams()
    model = D.pin_model(p)
    q = pin.neutral(model)
    q[:3] = (0.3, 0.02, 0.25)
    q[7:19] = np.arange(12) * 0.1 - 0.5  # distinct value per leg joint
    up = planner_to_world_qpos(q, StairParams(), "lower", "up")
    down = planner_to_world_qpos(q, StairParams(), "lower", "down")
    assert up[0] == pytest.approx(0.3) and down[0] == pytest.approx(-0.3)
    legs_up, legs_down = up[7:19].reshape(4, 3), down[7:19].reshape(4, 3)
    for a, b in ((0, 1), (1, 0), (2, 3), (3, 2)):  # FL<->FR, RL<->RR
        assert legs_down[a][0] == pytest.approx(-legs_up[b][0])  # ab/ad negated
        np.testing.assert_allclose(legs_down[a][1:], legs_up[b][1:])  # pitch joints unchanged
