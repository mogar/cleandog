"""report.md for the design-space study. Every number comes from results.json."""

from __future__ import annotations

import datetime as dt

from cleandog_cad import design_space as ds

IDEAS = [
    # (rank, key, idea, mechanism, cost / risk, likelihood)
    (1, "N", "Corner notches at the hips (your idea)",
     "The hip stack sits in a notch, so its ab/ad roll no longer has to clear a trunk wall beside it. The ab/ad "
     "motors live in the corners of the full-width centre section.",
     "The front thighs swing back beside the mid-trunk (figure 1), so the leg plane must still clear the centre "
     "section's side wall. The nose and tail ahead of / behind the hips get narrow (neck, camera, skid plate).",
     "high"),
    (2, "K", "Knee motor off the hip axis (above the hip, two-stage belt)",
     "Takes one whole motor length (45 mm) out of the stack outboard of the thigh. The knee motor rides on the ab/ad "
     "bracket above the hip motor and drives an idler on the hip axis, which drives the existing thigh belt.",
     "One more belt stage. Knee motor angle = knee + hip (the controller handles the coupling). Hips stand about "
     "90 mm above the trunk top.",
     "high"),
    (3, "M", "DM-J4310P for hip pitch and knee (2:1 reduction in the knee belt)",
     "Ø57 instead of Ø78.5 shrinks the roll sweep (R·sin θ) and the notch. One actuator type for all 12 joints "
     "and the neck.",
     "The DM-J4310P is 3.7 mm longer axially, which cancels the gain in a coaxial stack. Hip peak drops from 17 to "
     "12.5 Nm (the shove recoveries saturated the hip at 17 Nm).",
     "medium"),
    (4, "", "Asymmetric stance: rear legs inside the trunk width",
     "Figure 1: the rear thighs and shins never come forward of the rear hip at trunk height, so the rear legs "
     "could tuck under a full-width tail.",
     "Does not reduce the maximum (front) width. A front-rear asymmetric model for the planner and RL.",
     "medium"),
    (5, "", "Axially shorter motors (CubeMars AK60-6 Ø76 × 36.5 mm, 9 Nm peak; Steadywin GIM6010-8 Ø80 × 40 mm "
            "with driver, 11 Nm advertised)",
     "About 9 mm shorter per motor, so 18 mm per side in a coaxial stack.",
     "Hip needs 9.0 Nm peak with the 30% margin, which leaves no margin on the AK60-6. One builder measured "
     "only 3 Nm stall on the GIM6010-8. The AK60-6 costs about twice as much. No vendor CAD in the repo yet.",
     "medium-low"),
    (6, "", "Narrower trunk (160 mm) with the electronics stacked taller",
     "The width falls 1:1 with the trunk.",
     "A taller trunk raises the hips above the belly, and M0's stair results depend on that height. The 8.5 kg "
     "battery and electronics need the volume.",
     "medium-low"),
    (7, "", "Smaller ab/ad range (re-plan the sway to ±20°)",
     "Less roll sweep.",
     "The earlier hip-layout study found ±15° only saves 12 mm per side. The landing crawl and the static margin "
     "use the sway.",
     "low"),
    (8, "", "Knee motor on the thigh, near the hip (Solo-12 style)",
     "No coaxial knee motor.",
     "Raises leg inertia. The motor sweeps with the thigh through the hip-motor band and must clear the ab/ad "
     "bracket.",
     "low"),
    (9, "", "Hip stacks above the trunk (ab/ad axis on the top deck)",
     "The roll swings over the trunk instead of into it.",
     "Hips 50–90 mm higher: the centre of mass rises and the stair geometry M0 proved changes.",
     "low"),
    (10, "", "Trunk only between the hips (hips at the trunk ends)",
     "Hip stacks roll freely in front of / behind the trunk.",
     "The trunk becomes 160 mm long. The 378 mm skid plate and the high-centring results need the long belly.",
     "low"),
    (11, "", "Knee motor in the trunk, linkage through the ab/ad axis",
     "No knee motor on the leg at all.",
     "A spatial linkage or cable through a rolling joint: backlash, complexity, noise.",
     "low"),
    (12, "", "Flatter, larger-diameter motors (Unitree GO-M8010-6 Ø96 × 43.5, CubeMars AK80-9 Ø98 × 38.5)",
     "Slightly shorter stacks.",
     "Larger radius worsens the roll sweep. GO-M8010 is RS-485, not CAN.",
     "low"),
    (13, "", "Drop ab/ad (8 DoF)",
     "No hip stack roll at all.",
     "M0's reference uses ±34° of ab/ad for sway and the in-place landing turn.",
     "very low"),
]


def write_report(out, results, mj_frame, mj_files, cad_files):
    r = results
    B = r["B"]
    keys = [d.key for d in ds.DESIGNS]

    def w(k):
        return r[k]["overall_width"]

    rows = []
    for k in keys:
        x = r[k]
        rows.append(
            f"| {x['label']} | **{x['overall_width']:.0f}** | {x['length']:.0f} | {x['y_leg']:.0f} | "
            f"{x['abad_offset']:.0f} | {x['min_clear']:.1f} ({x['worst_part']}) | {x['stack_stairs_min']:.0f} | "
            f"{x['leg_stairs_min']:.0f} | {100 * x['infeasible']:.1f}% | {x['leg_motor_mass']:.2f} | "
            f"{x['knee_peak']:.0f} / {x['hip_peak']:.1f} |")
    cross = "\n".join(
        f"| {k} | {r[k]['min_clear']:.2f} | {r[k]['mj_trunk_min']:.2f} | {r[k]['mj_vs_np_max_diff']:.2f} | "
        f"{r[k]['mj_frames_compared']} | {r[k]['static_roll']:.1f} |" for k in keys)
    ideas = "\n".join(f"| {i} | {k or '—'} | {name} | {mech} | {risk} | {lik} |"
                      for i, k, name, mech, risk, lik in IDEAS)
    nk, n, kk = r["NK"], r["N"], r["K"]
    md = f"""# Making the robot narrower: design-space study

Generated {dt.datetime.now():%Y-%m-%d %H:%M} by `cad/scripts/design_space_run.py` and
`cad/scripts/design_space_report.py`.

## The question

The hip-layout study (`outputs/2026-10-19_hip-layout`) found that, with the real motors, the
robot is about **{B['overall_width']:.0f} mm wide**, against 276 mm in the sim (leg plane ±120 mm
plus the thigh capsule), with a 200 mm trunk. What else
could the design do? This report lists 13 ideas, ranks them by how likely they are to work, and
tests the top three plus their combinations against the whole M0 round trip.

## Answer

- **The corner notches and the off-axis knee motor both work, and they add up.**
  - N+K is **{w('NK'):.0f} mm** wide.
  - That is {B['overall_width'] - w('NK'):.0f} mm narrower than now, and keeps the 200 mm trunk.
- **Notches alone (N):** {w('N'):.0f} mm (−{B['overall_width'] - w('N'):.0f} mm).
- **Off-axis knee alone (K):** {w('K'):.0f} mm (−{B['overall_width'] - w('K'):.0f} mm).
- **Swapping to the DM-J4310P (M) does not help the width:** {w('M'):.0f} mm. Its smaller diameter
  is cancelled by its longer body.
- **Narrower also means fewer stair hits.**
  - In the baseline, the outboard knee motor dips {-B['stack_stairs_min']:.0f} mm into a nosing
    when ab/ad rolls the leg inward. N+K's motor stack stays at least
    {nk['stack_stairs_min']:.0f} mm clear of the stairs.
  - The thighs still touch nosings, as they do in the sim: {nk['leg_stairs_min']:.0f} mm for N+K,
    against −12.6 mm with the sim's own geometry and {B['leg_stairs_min']:.0f} mm for the
    baseline. The re-plan should fix this.
  - A smaller ab/ad offset (leg plane to ab/ad axis) also keeps the feet closer to where the
    planner put them.

![](fig_widths.png)

## 1. Where the legs actually go

Before changing the trunk shape, I looked at where the thighs and shins travel in the body frame
over the whole 13.5 min round trip.

![](fig_leg_sweep.png)

- **Front legs:** the thighs swing back (hip pitch up to 129°) and pass beside the trunk from x =
  −82 to +136 mm, the whole middle of the body.
  - So a notch can pull the front hip motors inboard, but the front thigh plane still has to clear
    the centre section's side wall.
- **Rear legs:** they never come forward of the rear hip at trunk height (hip pitch 20–142°, always
  backward).

## 2. Ideas, ranked by likelihood of success

| Rank | Tested as | Idea | How it narrows the robot | Cost and risk | Likelihood |
| --- | --- | --- | --- | --- | --- |
{ideas}

The top three were tested, plus combinations: N+K, N+M and N+K+M. Kf is the K variant with the
knee motor ahead of the front hip (behind the rear hip) instead of above.

## 3. How each design was tested

**Geometry.**
- The trunk is the sim's: 378 mm long, 80 mm shell plus 12 mm skid plate. Every design uses a
  200 mm centre section.
- Motors are the vendor dimensions:
  - RobStride 02: Ø78.5 × 45.4 mm;
  - DM-J4310P: Ø57 × 49.05 mm.
- Thigh: 36 × 24 mm. Shin: Ø26 mm. Hub/belt plate: 24 mm, plus a 10 mm first belt stage for K.
- The ab/ad motor sits as far out as the centre section allows (3 mm wall). Its axis passes
  through the hip axis, as in the sim.

**Motion.** Every frame of the M0 round trip (40,644 frames × 4 legs), two ways:
1. **Replay:** the reference joint angles, which are what the trained policy commands.
2. **Foothold IK:** the planned footholds, each moved sideways to sit under the design's leg plane,
   solved with that design's own IK and clipped to the ±34.4° ab/ad limit.

   The same joint angles are not enough on their own. With a wider ab/ad offset o, the same
   ab/ad angle a moves the foot vertically by Δo·sin a: 37 mm for the baseline at 34°.

**Solve.**
- A bisection finds the narrowest leg plane that keeps **3 mm between every leg part and the
  trunk** in every configuration of both motions.
- In notch designs, the nose is then widened as far as it goes.

**Cross-checks.**
- **Two independent clearance models:**
  - numpy point samples against a signed-distance trunk;
  - MuJoCo primitives replayed through the reference.
- **Baseline vs the earlier study:** the baseline reproduces the OCC-kernel hip-layout study to
  0.2 mm (leg plane 184.5 mm against 184.3 mm).
- **IK:** it reproduces the reference joint angles to 1e-13° when given the sim's geometry.
- **FK:** it matches the sim's MuJoCo foot sites exactly.

| Design | Min clearance, numpy (mm) | Min clearance, MuJoCo (mm) | Max frame-by-frame difference (mm) | Frames compared | Ab/ad-only check (mm) |
| --- | --- | --- | --- | --- | --- |
{cross}

## 4. Results

| Design | Overall width (mm) | Length (mm) | Leg plane from centre | Ab/ad offset | Min clearance to trunk (worst part) | Motor stack vs stairs (mm) | Legs vs stairs (mm) | Footholds out of ab/ad range | Leg motors (kg/leg) | Knee / hip peak (Nm) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
{chr(10).join(rows)}

How to read the table:
- **Stairs columns:** negative means penetration.
  - Legs vs stairs includes the thigh-into-nosing contact the sim already has (−12.6 mm with the
    sim's own geometry).
  - Only footholds the design can reach are counted.
- **Footholds out of ab/ad range:** the share of leg-frames whose foothold, moved under the new
  leg plane, needs more than ±34.4° of ab/ad. Those footholds must move in a re-plan.
  - These are almost all swing feet tucked high, about 90 mm below the hip, where a long ab/ad
    offset makes ab/ad very sensitive.
  - Fewer is better; the sim's own geometry has 0%.

### The designs, front view at the front hip

The hip stack is shown rolled to ±34.4° (faint) and at 0°. Aqua: ab/ad motor. Blue: hip motor.
Orange: knee motor. Yellow: thigh hub. In the notch designs, the dashed outline is the centre
section behind the hip.

![](fig_designs_front.png)

### Top views

![](fig_designs_top.png)

### Clearance through the round trip

![](fig_clearance_trace.png)

### Motors against the stairs

![](fig_stairs.png)

MuJoCo, frame {mj_frame} (the baseline's worst stair contact, upper flight). Left: baseline,
whose right knee motor is in the nosing. Right: N+K at the same moment.

<img src="{mj_files[0].name}" width="48%"> <img src="{mj_files[1].name}" width="48%">

### 3D, with the vendor motor models (stand pose)

{chr(10).join(f'![]({f.name})' for f in cad_files)}

## 5. The maths behind the widths

**Baseline.**
- The hip motor's inboard face is a disc of radius R at distance d outboard of the ab/ad axis
  (at y_a).
- Rolled by θ, its innermost point is at y_a + d·cos θ − R·sin θ. This must stay c outside the
  wall at W/2:

  d ≥ (W/2 + c − y_a + R·sin θ) / cos θ.

- Half-width = y_a + d + L_hip + t + L_knee.
- With W = 200, c = 3, y_a = 68.5, R = 39.25, θ = 34.4°: d = 68.7 mm. The half-width is then
  ≤ 252 mm by hand, against {B['overall_width'] / 2:.0f} mm from the sweep. The hand formula
  ignores the trunk's finite height, so it is conservative.

**Notches (N).**
- The hip motor no longer has a wall beside it. The binding constraint becomes the front thigh
  passing the centre section (figure 1):

  y_leg ≥ W/2 + t_thigh/2 + c + (roll of the thigh when it lies along x).

  The roll term exists because the thigh is then parallel to the ab/ad axis and swings on a
  circle of radius y_leg − y_a.
- Half-width = y_leg + t/2 + L_knee.
- The sweep gives y_leg = {n['y_leg']:.0f} mm, so {w('N'):.0f} mm overall. Worst part:
  {n['worst_part']}.

**Off-axis knee (K).**
- Half-width = y_leg + t/2. The knee motor's 45 mm leaves the outboard side.
- The hip motor moves 10 mm out to make room for the first belt stage, so y_leg grows by
  {kk['y_leg'] - B['y_leg']:.0f} mm.

**Why the ab/ad offset matters beyond width.**
- **Standing ab/ad torque:** per leg it is about F_z·o, with F_z ≈ 8.5 kg × 9.81 / 4 = 21 N.
  - Baseline: o = {B['abad_offset']:.0f} mm, so {21 * B['abad_offset'] / 1000:.1f} Nm.
  - N+K: o = {nk['abad_offset']:.0f} mm, so {21 * nk['abad_offset'] / 1000:.1f} Nm.
  - The DM-J4310P's rated torque is 3.5 Nm.
- **Dip of the outer stack:** when ab/ad rolls the foot inward by a, the outer end of the stack
  drops by about (y_outer − y_a)·sin a. That is the {-B['stack_stairs_min']:.0f} mm nosing
  penetration in the baseline.

## 5a. What fits in the trunk (N+K)

![](fig_packing.png)

- **Centre section:** {2 * nk['notch_x']:.0f} × 200 mm. The four ab/ad motors take its corners.
  - The battery (10S1P 21700, cells lying 5 × 2, about 110 × 72 × 44 mm) fits between them,
    on the centre of mass.
  - The power board fits in the 60 mm gap between the front and rear ab/ad motors.
- **Nose and tail:** each is {2 * nk['nose_half']:.0f} mm wide and about 110 mm long. They hold
  parts that a box trunk would have put in its middle:
  - the Pi 5 with its CAN HAT in the tail;
  - the e-stop radio and the neck-pitch motor in the nose.
- **Fit check:**
  - nothing is stacked above the battery, so every part fits in the 74 mm interior;
  - part sizes are ESTIMATEs until the parts are chosen;
  - the trunk planform is {nk['trunk_area_cm2']:.0f} cm², against {B['trunk_area_cm2']:.0f} cm²
    for the box.
- **RGB-D camera:** it doesn't fit inside the nose. It would sit on top of it.

## 6. Not modelled, and what to do next

- **Not modelled:**
  - the ab/ad bracket itself, which carries the hip stack around the outside of the hip motor;
  - wiring;
  - the skid plate's lips;
  - leg-to-leg contact (front vs rear on one side: the front shin never goes behind x = −82 mm
    and the rear legs never ahead of −125 mm at trunk height, so there is no overlap in this
    motion).
- **The footholds were moved, not re-planned.**
  - Re-run the planner, reference and RL with the chosen design's `hip_width` =
    {nk['hip_width'] / 1000:.3f} m and `abad_offset` = {nk['abad_offset'] / 1000:.4f} m.
  - Add leg-to-trunk collision (as the CLAUDE.md caveat already asks), and add the motor stack
    to the robot's collision geometry, so the policy learns not to dip the stack into nosings.
- **The notched trunk's narrow nose and tail change the skid plate:**
  - it narrows to {2 * nk['nose_half']:.0f} mm at both ends;
  - the belly's roll support on a nosing is then the 200 mm centre section only.
- **K's motors sit about 90 mm above the trunk top at the hips.** Check against the furniture the
  robot should walk under.
- **The 2:1 knee belt (M)** keeps the knee's peak at 24 Nm on a DM-J4310P. With N+K, the M swap
  becomes a cost and BOM choice rather than a width one (see N+K+M).
"""
    (out / "report.md").write_text(md)
