# Quiet House-Tidying Robot Dog — Project Plan

Oct 16, 2026 · @Morgan

## Summary

Build a roughly 8–9 kg, 12-DoF quadruped with a mouth-style gripper on a short neck. It clears floors into floor-level bins and climbs your 8 + landing + 7 hardwood stairwell. Estimated total: **$4,300–7,100**, spread over eight milestones that can be bought one at a time.

The core design bet is that a small robot can climb 7¾" risers. It would do this slowly, with its body parallel to the stair slope and a skid plate riding on the nosings. It would not take stairs the way Spot does. Milestone 0 tests that bet in simulation before any money is spent. The robot's final dimensions and actuator choice come out of that study.

Work runs on two tracks. Hardware milestones are purchased one after another. The simulation and software track costs nothing and runs continuously, so the gaps between purchases are never idle.

## Requirements and derived specs

Each of your ten requirements maps to a testable spec. Values marked *M0* are first estimates; the sizing study replaces them.

| Requirement | Derived spec | How it's verified |
| --- | --- | --- |
| Stairs | Ascend and descend 19.7 cm rise × 21.6 cm run (23.5 cm tread including overhang) hardwood, 92.7 cm wide, 8 risers + landing (180° turn, 78" × 35") + 7 risers. Balusters on the open side of the lower flight; walls elsewhere. Speed target ≤ 20 s per step (about 5 min per flight). Nothing is carried on stairs. | M3: 20 consecutive round trips with no intervention |
| Pick up objects | Mouth gripper on a 2–3 DoF neck. Payload 1.0 kg (hardback book) at about 25 cm reach. Lower jaw is a thin scoop, so flat objects can be slid under. Drops into floor-level bins. | M5: pick rate per object class |
| Quiet | Quasi-direct-drive BLDC actuators with FOC; no hobby servos. Low gear ratio, soft feet, motor PWM above 20 kHz. Target: quieter than your Roomba at 1 m. | dBA meter at 1 m, M1 and M2 |
| 45 min runtime | About 100 Wh pack (10S Li-ion) with margin (*M0*) | Timed flat-floor run in M2, full-task run in M6 |
| Cameras | Front RGB-D, a mouth camera, and one inexpensive 2D lidar | M4 |
| Small | Body about 40–45 × 25 cm, mass 8–9 kg (*M0*) | M0 output |
| Not fast | 0.2–0.3 m/s on flat floors | M2 |
| Sourceable parts | Off-the-shelf CAN actuators. Printed brackets. Structural parts as sheet metal or CNC from online services. Source CAD in FreeCAD or Onshape. | BOM review |
| No cloud | Locomotion and safety run onboard. Perception and planning run on your RTX 3090 desktop over the LAN. Open-weight models only. | Network audit: no WAN traffic |
| Budget < $10k | About $4.5–7k total, purchased milestone by milestone | Budget table below |
| Safe around kids (added) | Torque and current limits, speed cap, covered pinch points, wireless and physical e-stop. Falls or lost comms lead to a controlled sit-down. | M2 fault-injection tests |

## Stairwell geometry: why a small robot can climb it

In the body frame, your stairs look like flat ground with 14.5 cm-deep pockets. A robot with legs of roughly 13 cm per segment should be able to walk it. That figure is a hypothesis for M0 to test, not a design value. **M0 result (2026-10-16): 13 cm fails at the transitions; 16 cm is needed** (see the M0 status below).

**Your stairs:** rise 7.75" (19.7 cm), tread 9.25" (23.5 cm) from riser to nosing tip, so the run is 8.5" (21.6 cm); width 36.5" (92.7 cm). The nosing is rounded, 1" (2.5 cm) thick, with 0.75" (1.9 cm) overhang. That is a 42° slope, steeper than current code allows (about 37.8° at 7¾ / 10). The distance from one nosing to the next is 29.2 cm.

**The ladder posture.** Draw a line through all the nosings. If the body is pitched to lie parallel to that line, the treads and risers become triangular pockets below it. Each pocket reaches 19.7 × 21.6 / 29.2 = **14.5 cm** below the line, measured perpendicular to the slope.

In this frame, each step is an ordinary stride of about 30 cm. The swing foot only needs to clear the nosing line, not the riser height. Required leg extension is the hip's height above the nosing line plus the pocket depth used, about 8 + 14.5 ≈ 22.5 cm. That is about 12–13 cm per segment with a working margin.

**What does the work:**

- ~~**Riser bracing.** Plant feet in the inside corner of each pocket. The riser then carries the downhill load: 67% of body weight at 42°. This means the robot doesn't depend on rubber-on-hardwood friction.~~ **Corrected in M0 (2026-10-16):** no stair surface faces uphill, so the risers can't hold the robot from sliding downhill. Any uphill force comes from friction. Standing on the treads needs no friction, but resting on a slippery skid plate pushes the robot downhill: at 50% belly load with UHMW (μ 0.15), the feet need μ ≈ 0.67. Holding the full weight on the plate needs a plate μ of at least tan 42° = 0.91. M0 compares a grippy or ratcheting belly against never loading the belly.
- **Skid plate on the nosings.** The belly can rest on the nosing line between strides, so the robot inches up like a sled. A belly plate of low-friction UHMW or acetal is fine, since you've ruled out carrying anything on stairs.
- **Hip spacing near 29.2 cm.** If the front and rear hips sit one nosing-pitch apart, all four legs work at the same posture in their own pockets. M0 will sweep this value.
- **Low center of mass.** On a 42° slope, the center of mass shifts downhill by its height above the nosing line × sin 42°. A short, flat body keeps it over the support polygon.

**The hard parts are the transitions, not the flights.**

- At the top of each flight the slope turns convex. The skid plate can high-center on the last nosing while the front legs reach flat floor.
- At the bottom it turns concave. The front legs must reach up before the body can pitch.
- The landing is two of each.
- On the way down, M0 will compare descending rear-first (a toddler going backward) with head-first. Rear-first keeps the center of mass on the uphill side. **M0 result: descend rear-first.** It is the ascent played backwards, so it is feasible wherever ascent is. Head-first descent was not found for the recommended design.

**Energy is a non-issue.** Lifting 9 kg up the full stairwell (15 risers, 2.96 m) takes about 260 J. That is under 0.1% of a 100 Wh pack. Stairs are limited by peak torque, actuator heating, and contact reliability, not battery.

## Mechanical architecture

**Recommendation:** design your own leg and frame at the size M0 picks, borrowing from three open designs. None of the existing ones fits as-is. MEVIUS is roughly Go1-sized at about $12k. Solo-12 and Pupper v3 are smaller than your stairs need.

| Source design | What to take from it | Why not use it whole |
| --- | --- | --- |
| [Stanford Pupper v3](https://pupper-v3-documentation.readthedocs.io/en/latest/index.html) | ROS 2 + MuJoCo + RL software stack, CAN-BLDC electronics, URDF/MJCF pipeline | Likely too small for 19.7 cm risers; M0 will confirm. CAD source format still to be checked. |
| [ODRI Solo-12](https://www.mintlify.com/open-dynamic-robot-initiative/open_robot_actuator_hardware/overview) | Belt-reduction actuators (quiet). SolidWorks source CAD, Eagle electronics, drawings for machined parts. | 2.5 kg, no payload headroom |
| [MEVIUS](https://github.com/haraduka/mevius) | Sheet-metal structure ordered online. RL training pipeline, two-stage training recipe. | 15.5 kg, AK70-10 actuators, about $12k |

### Actuators

The RobStride 02 is the leading candidate for hip pitch and knee. A smaller Damiao unit is the candidate for hip ab/ad and the neck. Twelve leg actuators would cost about $1,800–2,200. Final torque and speed requirements come out of M0.

| Actuator | Peak / rated torque | Ratio | Mass | Listed price | Fit |
| --- | --- | --- | --- | --- | --- |
| [RobStride 02](https://aifitlab.com/products/robstride-02-motor) | 17 / 6 Nm | 7.75:1 | 405 g | $145–189 | Hip pitch, knee. Low ratio is good for noise and backdrivability. CAN 1 Mbps, 24–60 V. |
| [Damiao DM-J4310P](https://aifitlab.com/products/damiao-dm-j4310p-2ec-motor) | 12.5 / 3.5 Nm | 10:1 | not listed | $155 | Hip ab/ad, neck joints |
| [Damiao DM-J4340](https://aifitlab.com/products/damiao-dm-j4340-2ec-servo-motor) | 27 / 9 Nm | 40:1 | about 362 g | $155 | Rejected for legs: 40:1 is louder and slow (36 rpm nominal) |
| [CubeMars AK70-10](https://www.roboticscenter.ai/learn/robot-joint-motor-comparison) | about 25 Nm (as used on MEVIUS) | 10:1 | — | about $570 | Fallback only if M0 demands more torque; 3–4× the cost |

Several listings showed backorders. Buy the M1 actuators as the exact model you'll use in M2.

### Legs, feet and belly

- **Three DoF per leg.** Hip ab/ad and hip pitch sit at the body. The knee motor is also mounted at the hip and drives the knee through a belt or linkage. This keeps leg inertia low and avoids putting a gearbox at the foot, where it would be loud.
- **Segments:** 16 cm each, with knees pointing backward (M0 result; was 12–15 cm). Thigh and shin as CNC aluminum or carbon tube, joints as printed or machined housings.
- **Feet:** rubber hemispheres over a foam core, for grip on hardwood and quiet touchdown. They are shaped to seat into the riser corner.
- **Skid plate:** full-length UHMW or acetal belly plate with rounded front and rear lips, so it slides onto and off the nosings without catching.

### Mouth and neck

- **Two-DoF neck plus a one-DoF jaw.** Neck pitch at the body and head pitch, with yaw coming from the body itself. It folds flat and low for stairs.
- **Scoop lower jaw:** a thin, stiff spatula that slides under flat objects such as a pencil, a closed book, or a sock. The upper jaw clamps down on top, with compliant pads.
- **Reach for bins:** the body rears up and the neck extends to clear the bin rim. A bin rim height under about 20 cm keeps this easy (see open questions).
- **Load check:** a 1 kg book at 25 cm puts about 2.5 Nm plus the head's own weight on neck pitch, well inside a DM-J4310P.
- **Mouth camera** looking along the jaw, for the final approach to a grasp.

### Noise plan, in rough order of impact

1. **Footfall impacts.** Soft feet, plus an RL reward term that penalizes touchdown velocity and swing height.
2. **Gear whine.** Lowest practical ratio (7.75:1), belt-driven knee, avoid spur stages.
3. **PWM whine.** Run motor PWM at 20 kHz or above. Configurability on RobStride and Damiao drivers is still to be confirmed.
4. **Fans and resonance.** Passively cooled compute, damped panels.

Measure each with an SPL meter at 1 m on the M1 test stand and in M2. Your Roomba is the reference.

## Electronics, power, sensors and safety

A 10S Li-ion pack of about 100–150 Wh, a Raspberry Pi 5, and four CAN buses cover the robot. These are first-pass budgets for M0 to refine.

### Mass and power budget (estimates)

| Subsystem | Mass | Avg. power | Notes |
| --- | --- | --- | --- |
| 8 × RobStride 02 + 4 × DM-J4310P (legs) | 4.4 kg | 30–60 W | Holding torque costs I²R power even when standing still |
| Neck and jaw (2–3 actuators + structure) | 0.9 kg | 5 W | Idle most of the time |
| Frame, legs, skid plate | 1.5 kg | — |  |
| Battery (10S Li-ion with BMS) | 0.7–0.9 kg | — | 10S1P of 21700 cells ≈ 36 V × 4–5 Ah ≈ 145–180 Wh |
| Compute, sensors, power electronics | 0.6 kg | 15–25 W |  |
| **Total** | **≈ 8–9 kg** | **≈ 60–100 W** | 45 min at 100 W = 75 Wh, so 145 Wh leaves wide margin for aging and cold |

Ten-series Li-ion (36 V nominal, 42 V full) sits inside the RobStride 24–60 V range. Choose the 48 V variants of the Damiao units.

### Compute and buses

- **Onboard:** Raspberry Pi 5, the same computer Pupper v3 uses. It runs the ros2\_control loop, the locomotion policy, state estimation and safety. The robot must stay safe if Wi-Fi drops.
- **Alternative:** Jetson Orin Nano class, if M4 shows you want perception onboard. It costs more power.
- **CAN:** one bus per leg, with the neck sharing one. Classical CAN at 1 Mbps carries roughly 8,000 frames/s. Fifteen actuators × command + reply × 500 Hz is 15,000 frames/s, so a single bus is too few. Use a 4-channel CAN interface (USB or SPI HAT) on SocketCAN.
- **Desktop (RTX 3090, 24 GB):** perception models, grasp selection, map building, task planning. It talks to the robot over Wi-Fi.

### Sensors

- IMU on the main board
- Front RGB-D camera, pitched down for stair and floor perception. An OAK-D-class camera computes depth onboard and saves Pi CPU.
- Mouth camera
- 2D lidar, about $100 class, for SLAM
- Foot contact estimated from motor current; no foot sensors

### Safety

- Current and torque caps in actuator firmware. A 0.3 m/s speed cap.
- Covered pinch points at knee and neck.
- **E-stop:** a physical button plus a wireless stop on its own radio, separate from Wi-Fi. Sequence: actuators go to damping mode, the robot settles, then motor power is cut through a contactor.
- **Watchdogs:** lost commands, a fall detected by the IMU, or an over-temperature all trigger a controlled sit-down.
- **Battery:** BMS with temperature sensing. Charge only on the dock, in a spot away from anything flammable. This matters because the robot runs and charges unattended at night.
- **Development tether:** an overhead harness on the stairwell for all of M3.

## Software architecture

Use ROS 2 with ros2\_control and rmw\_zenoh. Everything needed to stay upright runs on the robot; everything heavy runs on the desktop. The same controllers run against MuJoCo or the real hardware by swapping one ros2\_control hardware plugin.

**Middleware**

- **Distro:** ROS 2 Lyrical (the 2026 LTS) if mujoco\_ros2\_control, Nav2 and rmw\_zenoh all have binaries for it. Otherwise Jazzy.
- **Transport:** rmw\_zenoh. Its [shared-memory transport](https://github.com/airacingtech/rmw_zenoh) applies to every message on the same machine, not only loaned ones, which gives you near zero-copy for camera and point-cloud topics. It also bridges cleanly to the desktop over Wi-Fi through a Zenoh router.
- **Parameters:** ROS 2 parameters live per node; there is no central parameter server as in ROS 1. Use YAML loaded by launch files. Add a small blackboard node only if you find you need global parameters.
- **Introspection:** MCAP rosbags, Foxglove (Pupper v3's tooling already uses it), and the ros2 CLI.

**On the robot (Pi 5, C++)**

- ros2\_control hardware interface over SocketCAN for RobStride and Damiao, one bus per leg. This is the first code you write, in M1.
- Locomotion: an RL policy as a ros2\_control controller at about 50 Hz, with joint PD running at the actuators. Separate policies or modes for flat walking, stair ascent and stair descent.
- State estimator: IMU plus leg kinematics.
- Safety supervisor: watchdogs, e-stop, sit-down behavior.

**On the desktop (RTX 3090)**

- **Perception:** open-vocabulary detection and segmentation of floor objects, all open-weight models running locally.
- **Mapping:** slam\_toolbox (2D lidar) or RTAB-Map (RGB-D), one map per floor, with the stairwell as a linking edge.
- **Navigation:** Nav2. "Take stairs" is a behavior-tree action that hands control to the stair policy.
- **Manipulation:** grasp pose from depth plus object mask. Pick and bin-drop policies trained from sim and teleop demos; the LeRobot tooling is a candidate.
- **Task layer:** BehaviorTree.CPP (which Nav2 already uses). Room sweep → detect → pick → bin → repeat → dock.

## Simulation stack and workflow

Use MuJoCo as the single physics world, with MJCF as the source of truth for the model. MuJoCo Playground on your 3090 handles RL training. Isaac Lab is the second opinion and the photoreal option later. Everything here is open source.

| Tool | Role | Why |
| --- | --- | --- |
| MuJoCo + MuJoCo Playground (MJX / MuJoCo Warp) | Gait and stair RL with thousands of parallel environments on the 3090 | MuJoCo Warp is a solver inside [Newton](https://developer.nvidia.com/blog/train-a-quadruped-locomotion-policy-and-simulate-cloth-manipulation-with-nvidia-isaac-lab-and-newton) and also powers Playground. Pupper v3 and MEVIUS already train this way. |
| [mujoco\_ros2\_control](https://github.com/PathOn-AI/mujoco_ros2_control) | Runs the real ROS 2 stack (controllers, Nav2, task tree) against sim | It's a ros2\_control SystemInterface with MJCF/URDF conversion, so sim and robot share code |
| Crocoddyl (C++) | Fast trajectory-optimization feasibility checks in M0 | Answers "physically possible?" in minutes, where an RL run takes hours |
| Isaac Lab / Isaac Sim | Cross-check of stair policies. Photoreal renders for perception training later. | Runs on the 3090. Newton is an experimental backend in Isaac Lab 3.0 beta. |

### Workflow

1. **Parametric robot generator.** A Python script emits MJCF and URDF from about ten parameters: segment lengths, hip spacing, body size, masses, skid plate, and an actuator model. The actuator model covers torque-speed curve, current limit, thermal model, friction and latency. M1 measurements replace the datasheet guesses.
2. **Your stairwell as a model.** Exact geometry: 8 steps, landing, 7 steps, rounded nosings, hardwood friction. Add randomized variants (rise 17–20 cm, tread 21–28 cm, friction 0.3–0.8) so policies don't overfit.
3. **Feasibility sweep (Crocoddyl).** Climb one step in ladder posture across the size grid. This prunes infeasible sizes.
4. **RL on the survivors.** Train flat walking, stair ascent, and stair descent (rear-first vs head-first). Rewards penalize touchdown velocity for noise and torque-squared for heat.
5. **Pick the morphology.** Choose the smallest one with ≥ 95% round-trip success on randomized stairs and ≥ 30% torque margin.
6. **Tidying world.** Model your actual rooms with bins and object assets. Books, shoes, toys and pencils start as rigid bodies. Socks and jackets begin as rigid or composite approximations and move to MuJoCo flex cloth later.
7. **Run the full task in sim** through mujoco\_ros2\_control: map, sweep, pick, bin, climb, repeat.

Known weak spot: sliding contact (belly on nosing) and cloth are where sim is least trustworthy. M1 measures the first on a real nosing, and the cloth objects get extra real-world testing in M5.

## Milestones

Eight milestones, M0–M7. Hardware is bought one milestone at a time. The software track continues between purchases: while you wait on M2 parts, the tidying sim (M6 software) can already be in progress.

&#91;embedded content: roadmap · hardware and sim tracks\]

Each sim result gates the hardware above it. The M1 leg rig also feeds back into the sim: its measured actuator model replaces datasheet guesses before stair training.

**M0 — Sim sizing and stair feasibility** · $0

- **Build:** parametric generator, stairwell model, Crocoddyl sweep, RL ascent/descent, neck-reach check (book from floor to bin rim).
- **Exit:**
  - A chosen morphology: segment lengths, hip spacing, mass budget.
  - An actuator torque/speed spec with ≥ 30% margin.
  - ≥ 95% simulated round trips on your stairwell.
  - A go/no-go on the Pupper v3 testbed (below).
- **Status (2026-10-16): exit items 1, 2 and 4 met; item 3 (RL round trips) is next.** Reports are in `outputs/` (git-ignored); regenerate them with `sim/scripts/`.
  - **Morphology:** 16 cm thigh and shin, knees backward, hip spacing 25 cm, trunk about 38 × 22 cm, 8.5 kg budget.
    - A quasi-static free-gait planner with verified swing paths completes all four uphill-facing transitions: floor→flight, flight→landing, landing→flight, flight→upper floor.
    - 13 cm segments are proven infeasible at 29.2 cm hip spacing; 14 cm was not resolved even with a 10× search budget.
  - **Knees must point downhill:** backward knees work facing uphill, forward knees facing downhill. So the robot climbs forwards and descends rear-first.
  - **Landing turn:** turning in place needs a 46 cm circle with the neck folded, which fits the 89 cm landing. With the neck straight out it needs 86 cm.
  - **Actuators:** Crocoddyl whole-body trajectories (μ = 0.5 friction cones, about 1 min per transition) confirm the RobStride 02 / DM-J4310P choice with the ≥ 30% margin:
    - knee needs 10.7 Nm peak (13.9 with margin) against 17 Nm, and 2.9 Nm RMS (3.8) against 6 Nm rated;
    - hip pitch needs 6.9 Nm peak and 1.4 Nm RMS;
    - ab/ad needs 2.8 Nm peak;
    - joint speeds stay under 5 rad/s.
  - **Pupper v3 testbed: no-go.** The chosen size (16 cm segments) is well above Pupper scale.
  - **Bin rim:** 20 cm confirmed for the 16 cm robot.

**M1 — One leg on a test stand** · $600–1,000

- **Buy:** 3 leg actuators plus 1 spare, a CAN interface, bench supply, leg parts (printed, plus CNC tubes).
- **Build:** a two-step stair mock to your exact dimensions in hardwood with a matching rounded nosing, plus a weighted sled carrying the skid plate.
- **Write:** the ros2\_control CAN hardware interface.
- **Exit:**
  - Actuator sysID (torque constant, friction, latency, thermal) fed back into the sim.
  - The leg replays a sim stair stride on the mock.
  - dBA at 1 m measured.
  - Belly-sliding force on the nosing measured against sim.
- **Pace:** aim for about 3–4 weeks after parts arrive.

**M2 — Locomotion base, flat floors** · $2,000–2,800

- **Buy:** 9 more leg actuators, frame (sheet metal and CNC), battery with BMS, Pi 5, IMU, 4-channel CAN, e-stop hardware, a dummy neck mass.
- **Exit:**
  - The sim-trained policy walks hardwood and carpet at 0.2 m/s.
  - Quieter than the Roomba at 1 m.
  - 45-minute flat-floor endurance run.
  - Fault-injection tests pass: e-stop, comms loss, push-over.

**M3 — Stairs** · $150–300

- **Buy:** production skid plate, stair feet, camera mount.
- **Exit:** 20 consecutive tethered round trips of the real stairwell with no intervention, then 10 untethered with a spotter.

**M4 — Perception and navigation** · $300–600

- **Buy:** RGB-D camera, 2D lidar.
- **Exit:** maps both floors, navigates room to room, and takes the stairs as a behavior-tree action, with all heavy compute on the desktop.

**M5 — Mouth and neck** · $400–700

- **Buy:** 2–3 neck and jaw actuators, mouth camera, printed head.
- **Exit:** each object class picked ≥ 90% in sim and ≥ 70% in reality on the first attempt, and dropped in a bin.

**M6 — Tidying autonomy** · $0–200

- **Exit:** clears a staged room of 15 mixed objects into bins within one 45-minute charge, on either floor.

**M7 — Dock and overnight operation** · $150–300

- **Buy:** charging dock with contacts.
- **Exit:** a week of unattended nightly runs. Logs are reviewed in Foxglove each morning.

**Optional testbed — Pupper v3 kit** · about $1,300–2,000

- **Gated by M0.** Buy it only if M0's morphology lands close to Pupper scale. If it does, consider forking Pupper v3 directly; that would fold M1 and M2 into one modification job.
- **Why it transfers well:** same ROS 2 + MuJoCo + CAN-BLDC architecture, with a [RL controller and sim mode](https://pupper-v3-documentation.readthedocs.io/en/latest/development/modifying_code.html) built in.
- **If M0 lands well above Pupper scale, skip it.** The Go2-class commercial robots fail your transfer test: different protocols, sensors and mechanisms.

## Budget

Estimated total: **$4,300–7,100** including 20% contingency, well under the $10k ceiling. M2 is the one large purchase. The optional Pupper v3 testbed would add about $1,300–2,000.

| Milestone | Low | High | Main cost |
| --- | --- | --- | --- |
| M0 Sim sizing | $0 | $0 | Your desktop |
| M1 Leg + test stand | $600 | $1,000 | 4 actuators, CAN interface, stair mock |
| M2 Locomotion base | $2,000 | $2,800 | 9 actuators, frame, battery, compute |
| M3 Stairs | $150 | $300 | Skid plate, feet |
| M4 Perception + nav | $300 | $600 | RGB-D camera, lidar |
| M5 Mouth + neck | $400 | $700 | 2–3 actuators, camera |
| M6 Tidying autonomy | $0 | $200 | Bins, test objects |
| M7 Dock | $150 | $300 | Dock and contacts |
| Contingency (20%) | $720 | $1,180 | Broken parts, redesigns |
| **Total** | **$4,320** | **$7,080** |  |

Actuator prices are from current US listings: RobStride 02 at $145–189 and Damiao DM-J4310P at $155. Other figures are estimates until each milestone's BOM is written.

## Risks and open questions

The biggest risk is that the small-robot stair bet fails in M0. If that happens, the fallback is a longer-legged robot of about 12–15 kg at higher cost, and you'd know before buying anything.

| Risk | Mitigation |
| --- | --- |
| Ladder posture can't handle the top or bottom transitions at small size | M0 tests these transitions explicitly. Fallback: longer segments. |
| Rounded hardwood nosings are slippery | Brace the feet against the risers, so grip doesn't depend on friction. Test on the M1 stair mock. |
| Sim misjudges belly sliding and cloth | Measure on the M1 sled. Cloth gets real-world trials in M5. |
| Falls on stairs damage the robot | Overhead tether through all of M3. Sacrificial bumpers. |
| Actuators backordered or discontinued | Buy M1 and M2 units from the same batch and keep a spare. |
| Motor PWM whine | Confirm drivers allow PWM ≥ 20 kHz before the M2 purchase. |

**Open questions**

- [x] Tread depth answered: 9.25" is riser to nosing tip, so the run is 8.5" and the slope is 42°.
- [x] Landing: 180° turn, 78" × 35". The flights sit side by side with a 5" partition. Stair counts are risers (8 + 7). *Still assumed:* the upper flight returns on the climber's left.
- [x] Bin rim height: **proposed 20 cm (8")**, open top at least 45 × 30 cm. At a level stance with the neck straight forward, the jaw clears this rim with a book held flat. The static limit is about 46 cm. Bins are chosen for the project.
- [ ] Is Pupper v3's CAD available as source (Onshape or STEP), not only STL?
- [ ] Can your printer run CF-nylon? That needs an enclosure and a hardened nozzle. Otherwise use PETG for brackets and CNC for the load paths.

**Sources**

- [MEVIUS2 paper](https://arxiv.org/html/2603.22031v1)
- [Pupper v3 docs](https://pupper-v3-documentation.readthedocs.io/en/latest/index.html)
- [ODRI hardware overview](https://www.mintlify.com/open-dynamic-robot-initiative/open_robot_actuator_hardware/overview)
- [RobStride 02 listing](https://aifitlab.com/products/robstride-02-motor)
- [Damiao DM-J4310P listing](https://aifitlab.com/products/damiao-dm-j4310p-2ec-motor)
- [Joint motor price comparison](https://www.roboticscenter.ai/learn/robot-joint-motor-comparison)
- [rmw\_zenoh](https://github.com/airacingtech/rmw_zenoh)
- [mujoco\_ros2\_control](https://github.com/PathOn-AI/mujoco_ros2_control)
- [Isaac Lab and Newton](https://developer.nvidia.com/blog/train-a-quadruped-locomotion-policy-and-simulate-cloth-manipulation-with-nvidia-isaac-lab-and-newton)
