"""Show the hip layout study in OCP CAD Viewer (VS Code).

Usage (with the OCP CAD Viewer panel open in VS Code):
  conda activate cleandog
  python cad/scripts/view_hip.py [--width 200] [--abad 40] [--no-ghosts]

Shows both front legs (real vendor motor models) on the placeholder trunk, at ab/ad 0°, plus
translucent copies of each hip stack rolled to the ends of the ab/ad range. Every part is named
in the viewer's tree, so it can be hidden or measured on its own. --dry-run builds everything and
prints the parts without contacting the viewer.
"""

from __future__ import annotations

import argparse

from build123d import Plane

from cleandog_cad.hip_layout import HipLayout, assembly, solve

COLOURS = {"trunk": "#c9ccd1", "ab/ad motor": "#1baf7a", "hip-pitch motor": "#2a78d6", "thigh hub": "#eda100",
           "knee motor": "#eb6834"}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--width", type=float, default=200.0, help="trunk width, mm")
    ap.add_argument("--abad", type=float, default=40.0, help="± ab/ad range the hip must clear, degrees")
    ap.add_argument("--no-ghosts", action="store_true", help="hide the stacks rolled to the range ends")
    ap.add_argument("--dry-run", action="store_true", help="build and list the parts; don't send to the viewer")
    args = ap.parse_args(argv)

    res = solve(HipLayout(body_width=args.width, abad_range_deg=args.abad))
    print(f"body {args.width:.0f} mm, ab/ad ±{args.abad:.0f}°: ab/ad offset {res.abad_offset:.1f} mm, "
          f"overall width {res.overall_width:.0f} mm, clearance {res.clearance:.1f} mm")

    objs, names, colours, alphas = [], [], [], []

    def add(shape, name, colour, alpha=1.0):
        objs.append(shape)
        names.append(name)
        colours.append(colour)
        alphas.append(alpha)

    for side, mirror in (("left", False), ("right", True)):
        poses = [(0.0, "", 1.0)] if args.no_ghosts else [(0.0, "", 1.0), (args.abad, f" ab/ad +{args.abad:.0f}°", 0.18),
                                                          (-args.abad, f" ab/ad -{args.abad:.0f}°", 0.18)]
        for ang, suffix, alpha in poses:
            for shape, label in assembly(res, ang):
                if label == "trunk" and (side == "right" or ang != 0.0):
                    continue  # one trunk is enough
                if label == "ab/ad motor" and ang != 0.0:
                    continue  # it doesn't move with the roll
                s = shape.mirror(Plane.XZ) if mirror else shape
                add(s, label if label == "trunk" else f"front-{side} {label}{suffix}", COLOURS[label],
                    0.35 if label == "trunk" else alpha)

    if args.dry_run:
        for o, n in zip(objs, names):
            bb = o.bounding_box()
            print(f"  {n:40s} x [{bb.min.X:7.1f}, {bb.max.X:7.1f}]  y [{bb.min.Y:7.1f}, {bb.max.Y:7.1f}]  "
                  f"z [{bb.min.Z:7.1f}, {bb.max.Z:7.1f}]")
        return 0

    from ocp_vscode import show

    show(*objs, names=names, colors=colours, alphas=alphas)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
