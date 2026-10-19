"""Off-the-shelf parts: vendor CAD placed in a standard frame, plus the data from their drawings.

Every motor uses one frame:
- output axis is +Z;
- the output flange face (the face you bolt to) is at z = 0;
- the housing front face sits at z = -boss_h and the housing ends at z = -length.

Sources, retrieval dates and the drawings themselves are in ../vendor/<part>/.
"""

from __future__ import annotations

import functools
from dataclasses import dataclass, field
from pathlib import Path

from build123d import Axis, Location, Rot, Shape, export_brep, import_brep, import_step

CAD_DIR = Path(__file__).resolve().parents[1]
VENDOR_DIR = CAD_DIR / "vendor"
CACHE_DIR = CAD_DIR / ".cache"  # git-ignored: BREP copies of vendor STEP files (much faster to load)


@dataclass(frozen=True)
class BoltCircle:
    n: int
    pcd: float  # pitch circle diameter, mm
    thread: str  # e.g. "M3"
    depth: float  # usable thread depth, mm (don't exceed: the manual warns about the casing thread)
    face: str  # "output" (rotates) | "front" (housing, output side) | "rear" (housing, back)


@dataclass(frozen=True)
class Motor:
    name: str
    step: str  # path under vendor/
    # Rigid placement taking the vendor model into the standard frame.
    to_standard: Location
    outer_d: float  # housing diameter, mm
    length: float  # total axial length incl. output boss, mm
    boss_d: float  # output boss / flange diameter, mm
    boss_h: float  # how far the output stands proud of the housing front face, mm
    mass: float  # kg
    bolts: tuple[BoltCircle, ...] = field(default_factory=tuple)
    notes: str = ""

    @property
    def body_length(self) -> float:
        """Housing length behind the front face (the output boss excluded)."""
        return self.length - self.boss_h

    def shape(self) -> Shape:
        return _load(self.step).moved(self.to_standard)


@functools.lru_cache(maxsize=None)
def _load(step: str) -> Shape:
    src = VENDOR_DIR / step
    cache = CACHE_DIR / (src.stem + ".brep")
    if cache.exists() and cache.stat().st_mtime >= src.stat().st_mtime:
        return import_brep(cache)
    shape = import_step(src)
    CACHE_DIR.mkdir(exist_ok=True)
    export_brep(shape, cache)
    return shape


# RobStride 02. Drawing in RS02_user_manual_251112.pdf, p. 2. The vendor model has the output
# axis on +Z with the output boss top at z = +4.0, so it only needs shifting down.
ROBSTRIDE_02 = Motor(
    name="RobStride 02",
    step="robstride-02/RS02.step",
    to_standard=Location((0, 0, -4.0)),
    outer_d=78.5,
    length=45.4,  # STEP bounding box (drawing: 41.5 body + boss)
    boss_d=43.0,
    boss_h=4.0,
    mass=0.380,  # manual "mechanical characteristic"; retailers list 405 g
    bolts=(
        BoltCircle(6, 24.0, "M4", 7.0, "output"),
        BoltCircle(9, 73.0, "M3", 8.0, "front"),
    ),
    notes="48 V rated, 24-60 V. 17 Nm peak / 6 Nm rated, 410 rpm no-load. Overload: 17 Nm 10 s, "
          "11 Nm 100 s, 9 Nm 370 s (manual p. 4).",
)

# Damiao DM-J4310P-2EC (V1.1 model, 48 V variant intended). Drawing in
# DM-J4310P-2EC_drawing_2d.pdf. The vendor model's axis is +Y with the output face at y = 0
# and the body toward +Y, so rotate -Y onto -Z.
DM_J4310P = Motor(
    name="Damiao DM-J4310P-2EC",
    step="damiao-dm-j4310p-2ec/DM-J4310P-2EC-V1.1_3d_20260228.stp",
    to_standard=Location((0, 0, 0)) * Rot(-90, 0, 0),
    outer_d=57.0,
    length=49.05,
    boss_d=37.5,
    boss_h=0.25,
    mass=0.330,
    bolts=(
        BoltCircle(6, 24.25, "M3", 6.0, "output"),
        BoltCircle(6, 50.0, "M3", 5.0, "front"),
        BoltCircle(4, 38.0, "M3", 4.0, "rear"),
    ),
    notes="48 V variant: 12.5 Nm peak / 3.5 Nm rated, 450 rpm no-load (200 rpm at 24 V). "
          "Output has a Ø9.5 pilot and a Ø4 dowel hole.",
)

MOTORS = {"robstride_02": ROBSTRIDE_02, "dm_j4310p": DM_J4310P}


def axis_check(m: Motor) -> dict:
    """Bounding box of a motor in the standard frame; expect z from -length to 0."""
    bb = m.shape().bounding_box()
    return dict(z_min=bb.min.Z, z_max=bb.max.Z, d=max(bb.size.X, bb.size.Y))


__all__ = ["Axis", "BoltCircle", "Motor", "MOTORS", "ROBSTRIDE_02", "DM_J4310P", "axis_check"]
