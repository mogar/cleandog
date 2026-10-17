import numpy as np

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
