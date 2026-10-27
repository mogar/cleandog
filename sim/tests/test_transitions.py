import numpy as np
import pytest

from cleandog_sim import RobotParams, StairParams
from cleandog_sim.stairs import Rect
from cleandog_sim.terrain import Surface, Terrain, flight_terrain, foothold_candidates
from cleandog_sim.transitions import Planner, Scenario, all_scenarios, turn_footprint


def flat_scenario():
    t = Terrain([Rect(-3, 3, -0.05, 0)], [Surface(-3, 3, 0, 0)], None)
    return Scenario("lower", "bottom", "up", t, 8, -1.0, (0,), (0,))


def test_crawls_on_flat_floor():
    pl = Planner(RobotParams(), flat_scenario())
    path, status = pl.plan(goal=lambda st: st[0] > -0.6, time_limit=60)
    assert status == "ok"
    for a, b in zip(path[:-1], path[1:]):
        assert sum(i != j for i, j in zip(a[5], b[5])) <= 1  # one foot moves per step


def test_plan_states_are_valid():
    """Every state on a plan passes the planner's own four-foot checks."""
    pl = Planner(RobotParams(), flat_scenario())
    path, _ = pl.plan(goal=lambda st: st[0] > -0.8, time_limit=60)
    for st in path:
        ok, _ = pl.evaluate(pl.pose(*st[:3]), pl.feet_of(st[5]))
        assert ok[0]


def test_mirrored_terrain_descends():
    s = StairParams()
    t = flight_terrain(s, 8, 1.0, False, 0.9, True).mirrored()
    z = t.raw_envelope(np.array([-1.5, -0.5, 0.5]))
    assert z[0] > z[1] > z[2]  # walking toward +x goes downhill


def test_footholds_clear_walls():
    s = StairParams()
    t = flight_terrain(s, 8, 1.0, False, 0.889, True)
    pts, sids = foothold_candidates(t, 0.02)
    wall_x = 7 * s.run + 0.889
    assert pts[sids == 8, 0].max() <= wall_x - 0.02 + 1e-9


def test_scenarios_cover_both_flights_and_directions():
    names = {sc.name for sc in all_scenarios(StairParams(), 0.42)}
    assert len(names) == 8


def test_turn_footprint_fits_landing_with_neck_folded():
    fp = turn_footprint(RobotParams(), neck_option=4)
    assert 2 * fp["radius"] < StairParams().flights[0].top_depth


def test_sway_ik_matches_mujoco():
    """The planner's 3D sway check (Planner.sway_ik) against MuJoCo's forward kinematics: with the
    trunk swayed sideways, the joint angles it finds put the foot on the foothold and the knee
    where its side view says."""
    import mujoco

    from cleandog_sim import RobotParams, load_model
    from cleandog_sim.robot import LEGS
    from cleandog_sim.transitions import Poses

    p = RobotParams()
    t = Terrain([Rect(-3, 3, -0.05, 0)], [Surface(-3, 3, 0, 0)], None)
    pl = Planner(p, Scenario("lower", "bottom", "up", t, 8, -1.0, (0,), (0,)))
    m = load_model(p)
    d = mujoco.MjData(m)
    rng = np.random.default_rng(3)
    checked = 0
    for _ in range(40):
        P = Poses(np.array([[0.0, rng.uniform(0.18, 0.26)]]), np.array([rng.uniform(-0.4, 0.4)]))
        feet = {leg: P.hip(p, leg[0] == "F")[0] + [rng.uniform(-0.08, 0.08), -rng.uniform(0.12, 0.25)] for leg in LEGS}
        y = rng.uniform(-0.05, 0.05)
        iks = {leg: pl.sway_ik(P, leg, feet[leg], np.array([y])) for leg in LEGS}
        if not all(pl.sway_leg_ok(P, leg, feet[leg], np.array([y]), collide=False)[0, 0] for leg in LEGS):
            continue
        d.qpos[:] = 0
        d.qpos[:3] = P.centre[0, 0], y, P.centre[0, 1]
        d.qpos[3:7] = np.cos(P.pitch[0] / 2), 0, -np.sin(P.pitch[0] / 2), 0  # nose up
        for i, leg in enumerate(LEGS):
            ik = iks[leg]
            sign = 1 if leg[1] == "L" else -1  # right legs mirror the ab/ad angle
            d.qpos[7 + 3 * i:10 + 3 * i] = sign * ik["abad"][0, 0], ik["hip"][0, 0], ik["knee"][0, 0]
        mujoco.mj_kinematics(m, d)
        for leg in LEGS:
            ik = iks[leg]
            foot = d.site_xpos[m.site(f"{leg}_foot").id]
            assert foot[[0, 2]] == pytest.approx(feet[leg], abs=1e-6)
            assert foot[1] == pytest.approx(p.leg_plane_y * (1 if leg[1] == "L" else -1), abs=1e-6)
            knee = d.xpos[m.body(f"{leg}_shin").id][[0, 2]]
            side = ik["hip_world"][0] + ik["knee_xz"][0, 0, 0] * ik["ex"][0] + ik["knee_xz"][0, 0, 1] * ik["ez"][0]
            assert knee == pytest.approx(side, abs=1e-6)
        checked += 1
    assert checked >= 10


def test_com_matches_mujoco():
    """The planner's 2D COM (Planner.com_x, which sets every stability margin it plans with)
    against MuJoCo's whole-body COM, for each neck fold: the same masses at the same places,
    including the hip stack's COM above the hip axis."""
    import mujoco

    from cleandog_sim import RobotParams, load_model
    from cleandog_sim.robot import LEGS
    from cleandog_sim.transitions import NECK_OPTIONS, Poses

    p = RobotParams()
    t = Terrain([Rect(-3, 3, -0.05, 0)], [Surface(-3, 3, 0, 0)], None)
    pl = Planner(p, Scenario("lower", "bottom", "up", t, 8, -1.0, (0,), (0,)))
    m = load_model(p)
    d = mujoco.MjData(m)
    neck, head = (m.jnt_qposadr[m.joint(j).id] for j in ("neck_pitch", "head_pitch"))
    rng = np.random.default_rng(5)
    checked = 0
    for _ in range(20):
        P = Poses(np.array([[0.0, rng.uniform(0.18, 0.26)]]), np.array([rng.uniform(-0.7, 0.7)]))
        feet = {leg: P.hip(p, leg[0] == "F")[0] + [rng.uniform(-0.08, 0.08), -rng.uniform(0.12, 0.25)] for leg in LEGS}
        iks = {leg: pl.sway_ik(P, leg, feet[leg], np.zeros(1)) for leg in LEGS}
        knees = {leg: pl.leg_check(P, leg, feet[leg])[1] for leg in LEGS}
        if not all(pl.sway_leg_ok(P, leg, feet[leg], np.zeros(1), collide=False)[0, 0] for leg in LEGS):
            continue
        d.qpos[:] = 0
        d.qpos[:3] = P.centre[0, 0], 0, P.centre[0, 1]
        d.qpos[3:7] = np.cos(P.pitch[0] / 2), 0, -np.sin(P.pitch[0] / 2), 0
        for i, leg in enumerate(LEGS):
            d.qpos[8 + 3 * i:10 + 3 * i] = iks[leg]["hip"][0, 0], iks[leg]["knee"][0, 0]
        for opt, (nr, hr) in enumerate(NECK_OPTIONS):
            d.qpos[neck], d.qpos[head] = nr, hr
            mujoco.mj_kinematics(m, d)
            mujoco.mj_comPos(m, d)
            assert pl.com_x(P, knees, feet, opt)[0] == pytest.approx(d.subtree_com[m.body("trunk").id][0], abs=1e-6)
        checked += 1
    assert checked >= 8


def test_smooth_necks_moves_less_and_keeps_margin():
    """smooth_necks picks folds within its slack of each move's best margin, never below the floor,
    and moves the neck no more than taking the best fold per move."""
    from cleandog_sim.transitions import NECK_OPTIONS

    pl = Planner(RobotParams(), flat_scenario(), prefer=0.025)
    path, status = pl.plan(goal=lambda st: st[0] > -0.6, time_limit=60)
    assert status == "ok"
    folds = pl.smooth_necks(path, slack=0.0025)
    assert len(folds) == len(path)
    best = [folds[0]]
    for k, (a, b) in enumerate(zip(path[:-1], path[1:]), 1):
        m = [min(pl.move_margins(a[5], b, neck=o)) for o in range(len(NECK_OPTIONS))]
        assert m[folds[k]] >= pl.margin
        assert m[folds[k]] >= min(max(m), pl.prefer) - 0.0025 - 1e-9
        best.append(int(np.argmax(m)))
    v = np.array(NECK_OPTIONS)
    motion = lambda seq: np.abs(np.diff(v[seq], axis=0)).max(1).sum()
    assert motion(folds) <= motion(best) + 1e-9
