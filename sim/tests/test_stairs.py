import math

import numpy as np
import pytest

from cleandog_sim import FlightParams, StairParams
from cleandog_sim.ladder import belly_rest_friction, slope_frame
from cleandog_sim.stairs import flight_profile, solid_mask


def test_plan_geometry():
    s = StairParams()
    assert math.degrees(s.slope) == pytest.approx(42.4, abs=0.1)
    assert s.nosing_pitch == pytest.approx(0.292, abs=0.001)
    assert s.pocket_depth == pytest.approx(0.1455, abs=0.001)


def test_profile_solid_and_free_space():
    s = StairParams()
    shapes = flight_profile(s, FlightParams(n_risers=4, top_depth=1.0))
    probe = lambda x, z: bool(solid_mask(shapes, np.array(x), np.array(z)))
    assert not probe(s.run + 0.05, 2 * s.rise + 0.005)  # just above tread
    assert probe(s.run + 0.05, 2 * s.rise - 0.005)  # inside tread board
    assert not probe(s.run - 0.005, 2 * s.rise - 0.05)  # pocket under the overhang
    assert probe(s.run - s.overhang + 0.002, 2 * s.rise - s.tread_thickness / 2)  # bullnose
    assert not probe(-0.1, 0.01)  # floor in front of the stairs


def test_nosing_line_touches_every_bullnose():
    s = StairParams()
    fr = slope_frame(s)
    r = s.tread_thickness / 2
    for k in range(1, 6):
        centre = np.array([(k - 1) * s.run - s.overhang + r, k * s.rise - r])
        assert np.dot(centre - fr.origin, fr.ev) == pytest.approx(-r, abs=1e-9)


def test_belly_rest_friction():
    s = StairParams()
    assert belly_rest_friction(s, 1.0, 0.15) == math.inf
    assert belly_rest_friction(s, 0.5, math.tan(s.slope) + 0.01) == 0.0
    # A low-friction skid plate holding half the weight needs grippy feet.
    assert belly_rest_friction(s, 0.5, 0.15) > 0.6
