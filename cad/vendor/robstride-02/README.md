# RobStride 02 (hip pitch, knee)

Retrieved 2026-10-19. The links were found on the Pakronics product page and point to Seeed
Studio's file host. RobStride's own download center is <https://www.robstride.com>.

| File | Source |
| --- | --- |
| `RS02.step` | <https://files.seeedstudio.com/products/RobStride/Product%20Literature/RS02/RS02.step> |
| `RS02_user_manual_251112.pdf` | <https://files.seeedstudio.com/products/RobStride/Product%20Literature/RS02/RS02User%20Manual251112.pdf> |
| `RobStride_product_specification_20250626.pdf` | <https://files.seeedstudio.com/products/RobStride/> (product family spec, 2025-06-26) |

## Key data (manual, pp. 2–4)

- **Size and mass:** Ø78.5 mm housing, 45.4 mm overall including a 4 mm output boss (Ø43), and 380 g ± 3 g. Retailers list 405 g.
- **Output flange:** 6 × M4 holes, 7 mm deep, on a Ø24 mm bolt circle.
- **Housing front face:** 9 × M3 holes, 8 mm deep, on a Ø73 mm bolt circle, plus 3 × Ø3 holes. The manual warns not to exceed the thread depth.
- **Electrical:**
  - 48 V rated, 24–60 V range;
  - 7 A phase current at rated load, 23 A peak;
  - torque constant 1.22 Nm/Arms.
- **Torque and speed:**
  - 17 Nm peak, 6 Nm rated;
  - 410 rpm no load, 360 rpm at rated load;
  - gear ratio 7.75:1.
- **Thermal overload limits at 25 °C:**

  | Torque | Allowed for |
  | --- | --- |
  | 17 Nm | 10 s |
  | 15 Nm | 18 s |
  | 13 Nm | 35 s |
  | 11 Nm | 100 s |
  | 9 Nm | 370 s |
  | 7 Nm | 1000 s |
  | 6 Nm | continuous |

- **Interface:** CAN at 1 Mbps, XT30PB power connector, 14-bit dual encoders.

## CAD frame

`cleandog_cad.vendor.ROBSTRIDE_02` places the model with its output on +Z and the output flange
face at z = 0. The vendor STEP only needs a −4 mm shift.
