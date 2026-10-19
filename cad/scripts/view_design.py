"""Show one design-space design in OCP CAD Viewer (VS Code), with the real vendor motor models.

Usage (with the OCP CAD Viewer panel open in VS Code):
  conda activate cleandog
  python cad/scripts/view_design.py NK [--frame 17416] [--dry-run]

The leg plane comes from the latest design_space_run.py results. --frame poses the legs as in that
frame of the round trip (foothold IK for this design); the default is the stand pose.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from cleandog_cad import design_space as ds
from cleandog_cad.design_cad import robot

COLOURS = {"trunk": "#c9ccd1", "ab/ad motor": "#1baf7a", "hip-pitch motor": "#2a78d6", "thigh hub": "#eda100",
           "knee motor": "#eb6834", "belt": "#3a3a38", "thigh": "#52514e", "shin": "#8d8b85"}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("design", help="design key, e.g. B, N, K, NK")
    ap.add_argument("--frame", type=int, help="reference frame to pose the legs at")
    ap.add_argument("--dry-run", action="store_true", help="build and list the parts; don't send to the viewer")
    args = ap.parse_args(argv)

    runs = sorted((ds.REPO / "outputs").glob("*_design-space/results.json"))
    if not runs:
        sys.exit("run cad/scripts/design_space_run.py first")
    res = json.loads(Path(runs[-1]).read_text())[args.design]
    des = next(d for d in ds.DESIGNS if d.key == args.design)
    if des.notch:
        des = ds.with_(des, nose_half=res["nose_half"])
    q_legs = None
    if args.frame is not None:
        mo = ds.motion_for(des.ya, res["y_leg"], ds.load_reference())
        q_legs = {leg: tuple(mo.q_ik[args.frame, i]) for i, leg in enumerate(("FL", "FR", "RL", "RR"))}
    parts = robot(des, res["y_leg"], q_legs)
    print(f"{res['label']}: {res['overall_width']:.0f} mm wide, leg plane {res['y_leg']:.1f} mm")
    names = [f"{i:02d} {label}" for i, (_, label) in enumerate(parts)]
    if args.dry_run:
        for (s, _), n in zip(parts, names):
            bb = s.bounding_box()
            print(f"  {n:22s} y [{bb.min.Y:7.1f}, {bb.max.Y:7.1f}]  z [{bb.min.Z:7.1f}, {bb.max.Z:7.1f}]")
        return 0
    from ocp_vscode import show

    show(*[s for s, _ in parts], names=names, colors=[COLOURS[label] for _, label in parts],
         alphas=[0.35 if label == "trunk" else 1.0 for _, label in parts])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
