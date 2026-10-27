# SkyRoute Urban 3D — UAV Path Planning Microproject

SkyRoute is a native OpenGL desktop simulation for an autonomous drone mission in a generated downtown. The urban region is 4.2 km long by 1.05 km wide (4.41 km²), with 80 blocks, marked roads, sidewalks, detailed building facades and rooftops, moving cars, pedestrians, and a 12-hazard aerial course. The presentation keeps the bright white canvas while using richer city geometry and several camera views.

## Mission

The drone starts on a plaza launch pad, climbs to its flight corridor, then flies about 1.5 km to a rooftop helipad. The route has 12 moving hazards: crossing and inspection UAVs, plus suspended construction platforms. Sensor Mode first computes an A* resultant route through the marked hazard corridor. A simulated nine-beam LiDAR scan supplies noisy obstacle position samples, and a three-axis constant-velocity Kalman filter estimates position and velocity. The simulator predicts future obstacle positions and activates D* Lite when predicted clearance between the drone and an obstacle surface falls below 3 m. It then replans around the detected hazard and rejoins the remaining A* route. Unseen course hazards remain unknown until sensed. Safe Mode plans an A* clearance route around the known obstacle map before flight. Both modes land on the rooftop pad and stop the rotors instead of hovering.

## Run

Double-click `run_skyroute.bat`. Python 3.10 or newer is required. Pyglet 2.1.19 is bundled in `vendor/pyglet` for offline launch. If the vendor folder is absent after cloning, the launcher installs the version pinned in `requirements.txt`.

## Controls

- `C` — start sensor mission.
- `A` — start preplanned safe-route mission.
- `R` — return the drone to its launch pad and reset the scene.
- `Space` — pause or resume.
- `1` — orbit view; right-drag to orbit and scroll to zoom.
- `2` — chase view.
- `3` — pilot view.
- `4` — overhead view following the drone.
- `5` — full city overview.

Both Sensor Mode and Safe Mode start in chase view. The camera label at the top right also cycles through the views when clicked.

## Simulation details

- City footprint: 4.2 km × 1.05 km (4.41 km²), with a 20-by-4 block grid.
- 80 varied buildings with multi-floor windows, reflective glazing, roof parapets and rooftop equipment.
- 64 moving cars, 152 varied roadside trees, 14 sidewalk pedestrians and 12 moving course hazards (nine UAVs and three suspended work platforms). Each course object crosses the route with its own speed and direction, drifts on a second axis, and changes altitude smoothly, making movement visible from the chase camera. Seeded parameters keep each run repeatable without synchronizing the obstacles.
- Sensor Mode follows the curved left-side A* resultant path through the hazard field from takeoff. Only when the sensors predict collision risk does D* Lite update the map and compute a local reroute to a checkpoint beyond that hazard; it then resumes the marked A* corridor. The original A* route stays visible as a faint reference during rerouting. The HUD counts unique course hazards sensed out of 12.
- LiDAR uses nine forward-facing horizontal beams across a 64° field of view, 180 m range, and seeded 1.5 m standard-deviation measurement noise. The Kalman filter estimates all three position and velocity components from those returns; collision-risk checks use the estimates, not the obstacle's scripted future position.
- The evaluation HUD reports flown path length, physical collision count and collisions per kilometre, number of replans, average planner computation time, and mission success. Risk warnings and avoided contacts remain separate from physical collisions.
- To vary the experiment, change `COURSE_HAZARD_COUNT` (1–12) and `HAZARD_SPEED_SCALE` in `skyroute3d.py`. The LiDAR range, beam angles, and measurement noise are configurable beside them.
- Safe Mode uses A* to plan a clearance route around known obstacles before takeoff.
- D* Lite follows Koenig and Likhachev, AAAI 2002: [D* Lite paper](https://ocs.aaai.org/Library/AAAI/2002/aaai02-072.php).
- Mission stages: launch, vertical takeoff, cruise, landing approach, touchdown and motors idle.
- Four-plus camera angles with a separate full-city view.

The city is procedural and has a Central Park-scale footprint; it is not a georeferenced New York map or the CARLA Unreal Engine renderer. Flight and sensor behavior are simplified for a microproject demonstration, not a flight controller.

## Project files

- `skyroute3d.py` — city simulation and renderer; uses the AI model in `skyroute_ai_model.py`.
- `skyroute_ai_model.py` — report-ready, standard-library-only LiDAR, Kalman, collision-risk, A*, and D* Lite algorithms.
- `run_skyroute.bat` — Windows launcher.
- `requirements.txt` — pinned Pyglet dependency.
- `vendor/pyglet` — bundled windowing and OpenGL library.
