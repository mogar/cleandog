"""Write MJCF scenes to sim/models/ for inspection in the MuJoCo viewer.

Usage: python scripts/build_models.py
       python -m mujoco.viewer --mjcf=models/stairwell.xml
"""

from pathlib import Path

from cleandog_sim import RobotParams, StairParams, build_scene_xml

OUT = Path(__file__).resolve().parent.parent / "models"


def main() -> None:
    OUT.mkdir(exist_ok=True)
    robot = RobotParams()
    scenes = {"flat": None, "stairwell": StairParams(), "single_flight": StairParams().single_flight()}
    for name, stairs in scenes.items():
        path = OUT / f"{name}.xml"
        path.write_text(build_scene_xml(robot, stairs))
        print(path)


if __name__ == "__main__":
    main()
