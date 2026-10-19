"""Hip layout study report: outputs/<date>_hip-layout/report.md.

Usage: python cad/scripts/hip_layout_report.py
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from cleandog_cad.hip_layout import HipLayout, analytic_y0, assembly, solve  # noqa: E402
from cleandog_cad.render import render  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
COLS = {"trunk": "#c9ccd1", "ab/ad motor": "#1baf7a", "hip-pitch motor": "#2a78d6", "thigh hub": "#eda100",
        "knee motor": "#eb6834"}


def main() -> int:
    out = REPO / "outputs" / f"{dt.date.today().isoformat()}_hip-layout"
    out.mkdir(parents=True, exist_ok=True)
    rows, sweep = [], {}
    for W in (160, 180, 200, 220):
        for rng in (15, 20, 25, 30, 35, 40):
            r = solve(HipLayout(body_width=W, abad_range_deg=rng))
            sweep[(W, rng)] = r
    for W in (160, 180, 200, 220):
        r = sweep[(W, 40)]
        rows.append(f"| {W} | {r.ya:.1f} | {r.abad_offset:.1f} | {r.leg_plane:.1f} | {r.overall_width:.0f} | "
                    f"{r.clearance:.1f} |")
    fig, ax = plt.subplots(figsize=(6.5, 3.4))
    rngs = (15, 20, 25, 30, 35, 40)
    ax.plot(rngs, [sweep[(200, g)].abad_offset for g in rngs], color="#2a78d6", lw=2, marker="o")
    ax.axhline(50, color="#52514e", lw=1, ls=(0, (4, 3)))
    ax.text(15, 52, "sim placeholder (50 mm)", fontsize=8, color="#52514e")
    ax.axvline(34.4, color="#8d8b85", lw=1)
    ax.text(34.6, 60, "used by the\nround trip", fontsize=8, color="#52514e")
    ax.set_xlabel("ab/ad range the hip must clear (± degrees)")
    ax.set_ylabel("ab/ad offset (mm)")
    ax.set_ylim(0, 130)
    ax.set_title("Leg-plane offset from the ab/ad axis (body width 200 mm)", fontsize=10)
    fig.tight_layout()
    fig.savefig(out / "offset_vs_range.png", dpi=150)
    plt.close(fig)
    r = sweep[(200, 40)]
    for ang in (0, -40):
        render([(s, COLS[label], 0.3 if label == "trunk" else 1.0) for s, label in assembly(r, ang)],
               out / f"hip_stack_abad{ang}.png", views=("iso", "front", "top"),
               title=f"Front-left hip stack, ab/ad {ang}° (body 200 mm wide)", tol=0.6)
    hand = analytic_y0(HipLayout(body_width=200, abad_range_deg=40))
    md = f"""# Hip layout with the real motors

Generated {dt.datetime.now():%Y-%m-%d %H:%M} by `cad/scripts/hip_layout_report.py`.

## Question

How far must the leg plane sit from the ab/ad axis, once the real motors are in place? The sim
used a 50 mm placeholder, and with it the thighs passed through the body by up to 21 mm.

## Layout studied

One leg, from the body outward:

1. **Ab/ad motor** (DM-J4310P, Ø57 × 49 mm): inside the body, axis along x, as far out as the body wall allows (3 mm wall gap).
2. **Hip-pitch motor** (RobStride 02, Ø78.5 × 45.4 mm): axis along y, output facing out.
3. **Thigh hub:** a 24 mm plate whose mid-plane is the leg plane.
4. **Knee motor** (RobStride 02): output facing in, bolted to the thigh hub.

Mounting the knee motor on the thigh makes the knee angle equal the motor's angle.

The hip stack (hip motor, hub, knee motor) rolls with the ab/ad joint and must clear the trunk by
3 mm at every angle in the range. A bisection search over the hip motor's position, using exact
distance queries from the CAD kernel, finds the closest stack that clears. A hand formula (a disc
rolling past a wall) bounds it from the conservative side: at 200 mm and ±40° the kernel needs
the hip motor face at {r.y0:.1f} mm from the centreline, against {hand:.1f} mm by hand.

## Result

| Body width (mm) | Ab/ad axis from centre | Ab/ad offset | Leg plane from centre | Overall width (knee motors) | Min clearance |
| --- | --- | --- | --- | --- | --- |
{chr(10).join(rows)}

(±40° ab/ad range. The round trip uses ±34.4°.)

![](offset_vs_range.png)

![](hip_stack_abad0.png)

![](hip_stack_abad-40.png)

- **The offset is about 116–118 mm whatever the body width,** more than twice the sim's 50 mm.
  Two things set it, and the ab/ad range barely does:
  - the hip motor's 45 mm length (the leg plane sits outboard of the whole motor);
  - the ab/ad motor's radius inside the body wall.

  Even ±15° only gets it to 104 mm.
- **Overall width** comes out at about 45–51 cm with the knee motors, against about 34 cm in the sim.
  The 93 cm flights have room.

## Consequences for the sim (to re-run)

- **Legs vs body:** the leg plane sits 66–86 mm outside the body wall, so the 21 mm overlap goes
  away. The sim should still get leg-to-body collision turned on, to prove it.
- **Wider feet:** feet sit about 37 cm apart instead of 24 cm. That gives more side-to-side
  stability.
- **Landing turn:** turning in place needs a bigger circle, about 50 cm against the 89 cm landing,
  instead of 46 cm.
- **Mass placement:** about 0.8 kg per hip (two motors) now sits 13–17 cm from the centreline, so
  the robot's roll inertia goes up.
- **Ab/ad torque:** standing load acts about 12 cm out from the ab/ad axis, about 2.5 Nm per leg,
  inside the DM-J4310P's 3.5 Nm rated torque.

## Not decided here

- the knee drive from hub to knee (belt or linkage);
- hub and bracket structure;
- where the ab/ad motor sits along x relative to the hip.
"""
    (out / "report.md").write_text(md)
    print(out / "report.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
