"""Parameter sets for the parametric robot and the stairwell.

All lengths in metres, masses in kg, torques in Nm, speeds in rad/s at the
joint output. Values marked ESTIMATE are placeholders until M1 sysID or a
datasheet confirms them.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace


@dataclass(frozen=True)
class ActuatorSpec:
    name: str
    peak_torque: float
    rated_torque: float
    gear_ratio: float
    mass: float
    no_load_speed: float  # rad/s at output, at pack voltage
    armature: float  # reflected rotor inertia, kg m^2 (rotor J * ratio^2)
    frictionloss: float  # Coulomb friction at output, Nm
    damping: float  # viscous friction at output, Nm s/rad

    def torque_limit(self, speed: float) -> float:
        """Linear torque-speed envelope: full peak torque at stall, zero at no-load speed."""
        return max(0.0, self.peak_torque * (1.0 - abs(speed) / self.no_load_speed))


# RobStride 02: torque, ratio, mass from listing. Speed/armature/friction ESTIMATE.
ROBSTRIDE_02 = ActuatorSpec(
    name="robstride_02",
    peak_torque=17.0,
    rated_torque=6.0,
    gear_ratio=7.75,
    mass=0.405,
    no_load_speed=410 * 2 * math.pi / 60,  # ESTIMATE ~410 rpm at 48 V
    armature=4.0e-5 * 7.75**2,  # ESTIMATE rotor J
    frictionloss=0.15,  # ESTIMATE
    damping=0.02,  # ESTIMATE
)

# Damiao DM-J4310P: torque, ratio from listing. Mass/speed/armature/friction ESTIMATE.
DM_J4310P = ActuatorSpec(
    name="dm_j4310p",
    peak_torque=12.5,
    rated_torque=3.5,
    gear_ratio=10.0,
    mass=0.30,  # ESTIMATE, not listed
    no_load_speed=200 * 2 * math.pi / 60,  # ESTIMATE
    armature=2.5e-5 * 10.0**2,  # ESTIMATE
    frictionloss=0.15,  # ESTIMATE
    damping=0.02,  # ESTIMATE
)

ACTUATORS = {a.name: a for a in (ROBSTRIDE_02, DM_J4310P)}


@dataclass(frozen=True)
class RobotParams:
    """The ~10 parameters M0 sweeps, plus fixed mass-budget items."""

    # Body (x forward, y left, z up; origin at body centre)
    body_length: float = 0.42
    body_width: float = 0.22
    body_height: float = 0.08
    hip_spacing: float = 0.292  # front-rear hip pitch-axis distance; plan's guess = nosing pitch
    hip_width: float = 0.14  # left-right ab/ad axis distance
    hip_z: float = 0.0  # hip axis height relative to body centre
    abad_offset: float = 0.05  # lateral offset from ab/ad axis to leg plane

    # Legs
    thigh_length: float = 0.13
    shin_length: float = 0.13
    foot_radius: float = 0.02
    thigh_radius: float = 0.018  # collision capsule radius
    shin_radius: float = 0.013
    # Sign of knee flexion about +y. -1: knee points backward (Go1/Spot style),
    # +1: knee points forward. M0 sweeps the front/rear combinations.
    knee_sign_front: int = -1
    knee_sign_rear: int = -1

    # Skid plate under the belly
    skid_thickness: float = 0.012
    skid_lip_radius: float = 0.03

    # Actuators: ab/ad, hip pitch, knee (knee motor sits at the hip, belt to knee)
    abad_actuator: str = "dm_j4310p"
    hip_actuator: str = "robstride_02"
    knee_actuator: str = "robstride_02"
    knee_belt_ratio: float = 1.0

    # Neck/head (M0 uses it for the reach check and as a dummy mass)
    include_neck: bool = True
    neck_length: float = 0.15
    head_length: float = 0.12
    head_mass: float = 0.45  # head shell + jaw actuator + camera
    neck_actuator: str = "dm_j4310p"

    # Mass budget (excluding actuators, which come from ActuatorSpec)
    total_mass_target: float = 8.5
    thigh_mass: float = 0.15
    shin_mass: float = 0.08
    foot_mass: float = 0.03
    hip_bracket_mass: float = 0.08
    neck_link_mass: float = 0.12

    # Contact friction (robot side wins via geom priority)
    foot_friction: float = 0.6  # rubber on finished hardwood, ESTIMATE
    skid_friction: float = 0.15  # UHMW/acetal on wood, ESTIMATE

    def actuator(self, role: str) -> ActuatorSpec:
        return ACTUATORS[getattr(self, f"{role}_actuator")]

    @property
    def leg_reach(self) -> float:
        return self.thigh_length + self.shin_length

    @property
    def hip_above_belly(self) -> float:
        """Hip axis height above the skid plate's underside."""
        return self.hip_z + self.body_height / 2 + self.skid_thickness

    def with_(self, **kw) -> "RobotParams":
        return replace(self, **kw)


@dataclass(frozen=True)
class FlightParams:
    n_risers: int
    top_depth: float  # depth of the surface at the top of the flight (landing or floor)
    # What bounds each side, looking uphill: "wall", "balusters" or "open".
    left: str = "wall"
    right: str = "wall"


@dataclass(frozen=True)
class StairParams:
    """Morgan's stairwell: 8 risers, a 180-degree landing, then 7 risers.

    The two flights sit side by side with a partition wall between them.
    The landing is 78" across both flights and 35" deep. Its far end and both
    sides are walled. The lower flight's outer side has balusters; every other
    side is a wall.
    """

    rise: float = 0.197  # 7.75"
    run: float = 0.216  # 8.5", riser face to riser face
    overhang: float = 0.019  # 0.75"
    tread_thickness: float = 0.025  # 1", nosing is a full bullnose of this thickness
    width: float = 0.927  # 36.5", clear tread width of each flight
    flights: tuple[FlightParams, ...] = field(
        default_factory=lambda: (
            FlightParams(n_risers=8, top_depth=0.889, left="wall", right="balusters"),  # 35" landing
            FlightParams(n_risers=7, top_depth=1.5),  # upper floor
        )
    )
    landing_width: float = 1.981  # 78", across both flights incl. the partition
    turn_left: bool = True  # ASSUMED: upper flight returns on the climber's left
    wall_thickness: float = 0.02
    wall_height: float = 1.0
    baluster_radius: float = 0.016
    balusters_per_tread: int = 2
    baluster_inset: float = 0.03  # baluster centre in from the tread edge
    handrail_height: float = 0.86  # 34" above the nosing line
    friction: float = 0.5  # hardwood; robot geoms override via priority

    @property
    def slope(self) -> float:
        return math.atan2(self.rise, self.run)

    @property
    def nosing_pitch(self) -> float:
        return math.hypot(self.rise, self.run)

    @property
    def pocket_depth(self) -> float:
        """Perpendicular depth of a riser/tread corner below the nosing line (no overhang)."""
        return self.rise * self.run / self.nosing_pitch

    @property
    def total_rise(self) -> float:
        return self.rise * sum(f.n_risers for f in self.flights)

    @property
    def partition_thickness(self) -> float:
        """Gap between the two flights, filled by the partition wall."""
        return self.landing_width - 2 * self.width

    def single_flight(self, n_risers: int = 8, top_depth: float = 1.0) -> "StairParams":
        """One straight flight with walls, for analyses that ignore the landing."""
        return replace(self, flights=(FlightParams(n_risers, top_depth),))

    def open_sides(self) -> "StairParams":
        """Same stairs with walls and balusters removed, for unobstructed renders."""
        return replace(self, flights=tuple(replace(f, left="open", right="open") for f in self.flights))

    def with_(self, **kw) -> "StairParams":
        return replace(self, **kw)
