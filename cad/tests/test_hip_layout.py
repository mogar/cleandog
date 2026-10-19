import pytest

from cleandog_cad.hip_layout import HipLayout, analytic_y0, hip_stack, solve
from cleandog_cad.vendor import ROBSTRIDE_02


def test_hip_stack_parts_sit_where_intended():
    lay = HipLayout()
    L, t = ROBSTRIDE_02.length, lay.thigh_hub_t
    spans = [(p.bounding_box().min.Y, p.bounding_box().max.Y) for p in hip_stack(100.0, lay)]
    expected = [(100, 100 + L), (100 + L, 100 + L + t), (100 + L + t, 100 + 2 * L + t)]
    for got, want in zip(spans, expected):
        assert got == pytest.approx(want, abs=0.05)


def test_search_agrees_with_hand_calculation():
    """The disc formula ignores the trunk's finite height, so it can only be equal or more
    conservative than the kernel's distance search."""
    lay = HipLayout(body_width=200, abad_range_deg=40)
    r = solve(lay)
    assert r.clearance >= lay.clearance
    assert r.y0 <= analytic_y0(lay) + 0.5
