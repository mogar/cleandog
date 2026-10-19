# Damiao DM-J4310P-2EC (hip ab/ad, neck)

Retrieved 2026-10-19 from Damiao's official repository <https://github.com/dmBots/DM-J4310P-2EC>
(branch `master`). Distributor pages (AIFIT Lab, OpenELAB) link to it.

| File | Source path in the repo |
| --- | --- |
| `DM-J4310P-2EC_drawing_2d.pdf` | `drawings/2d/4310 减速电机-rau 3505-2250717.PDF` |
| `DM-J4310P-2EC_user_manual_V1.3_2026-09-14.pdf` | `manual/DM-J4310P-2EC Geared Motor User Manual V1.3 2026-09-14.pdf` |
| `DM-J4310P-2EC-V1.1_3d_20260228.stp` | `models/3d/DM-J4310P-2EC-V1.1_3d_20260228_1.stp` (35 MB) |
| `curves/*.png` | `test-data/performance-curves/` (torque and speed efficiency, temperature rise) |

## Key data (drawing and distributor tables)

- **Size and mass:** Ø57 × 49.05 mm and about 330 g. The output face stands 0.25 mm proud of the housing.
- **Output:**
  - 6 × M3 holes, 6 mm deep, on a Ø24.25 mm bolt circle;
  - a Ø9.5 mm pilot;
  - a Ø4 mm dowel hole;
  - Ø27 and Ø37.5 mm features.
- **Housing, output side:** 6 × M3 holes, 5 mm deep, on Ø50 mm, plus a Ø3 mm dowel hole.
- **Housing, rear:** 4 × M3 holes, 4 mm deep, on Ø38 mm.
- **Torque and speed:**
  - 12.5 Nm peak (transient), 3.5 Nm rated;
  - gear ratio 10:1;
  - 120 rpm rated;
  - no-load speed 450 rpm on the **48 V** variant and 200 rpm on the 24 V one. Buy the 48 V variant: the robot runs a 10S (36 V) pack.
- **Interface:** CAN at 1 Mbps, UART for configuration, MIT / position / speed control modes, crossed-roller output bearing.

## CAD frame

`cleandog_cad.vendor.DM_J4310P` rotates the model (vendor axis +Y, output at y = 0) so the output
faces +Z at z = 0.
