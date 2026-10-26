# SkyRoute Urban 3D - UAV Path Planning Microproject

A native OpenGL desktop simulation for autonomous UAV navigation in a procedural downtown street scene. It uses a white, high-visibility presentation with city roads, intersections, building facades, street trees, lamps, moving traffic, and a crossing UAV.

## Run

Double-click run_skyroute.bat. Python 3.10 or newer is required. Pyglet 2.1.19 is bundled in vendor for an offline launch. If the vendor folder is absent after cloning the source from GitHub, the launcher installs the version pinned in requirements.txt.

## Sensor avoidance demo

Choose Run Sensor Avoidance or press C. The drone scans ahead with visible forward range-sensor beams. When a moving UAV enters the flight corridor, the sensor reports its distance and the planner computes a clear route before the drone reaches it. The aircraft does not need to make contact. The avoidance counter records sensor-triggered reroutes.

Choose Autonomous Route or press A to plan a safe route from the start. Press R to reset and Space to pause or resume.

## Camera views

Press 1 for orbit, 2 for chase, 3 for pilot view, and 4 for overhead. Right-drag to orbit the free camera and scroll to zoom. Click the camera label at the upper right to cycle through the views.

## Included simulation features

- 3D A* route planning on a voxel grid, with dynamic-object prediction and clearance margins.
- Moving cars, a moving crossing UAV, and a sidewalk robot.
- Forward range-sensor fan and a visible detection line to the current hazard.
- Pre-contact avoidance and live route replanning.
- Four named camera views, plus free orbit and zoom.
- Live altitude, speed, route state, and sensor-avoidance count.

The scene is procedural and designed to make the path-planning behavior easy to see. Flight dynamics are simplified; this is a project demonstration, not a flight controller or a high-fidelity aerodynamic model.

## Project files

- skyroute3d.py - simulation, city scene, sensor logic, and 3D renderer.
- run_skyroute.bat - Windows launcher.
- requirements.txt - pinned Pyglet dependency for source checkouts.
