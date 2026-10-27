# SkyRoute Urban 3D — UAV Path Planning Microproject

SkyRoute is a native OpenGL desktop simulation for an autonomous drone mission in a generated downtown. The urban region is 4.2 km long by 1.05 km wide (4.41 km²), with 80 blocks, marked roads, sidewalks, detailed building facades and rooftops, moving cars, pedestrians, and a 12-hazard aerial course. The presentation keeps the bright white canvas while using richer city geometry and several camera views.

## Mission

The drone starts on a plaza launch pad, climbs to its flight corridor, then flies about 1.5 km to a rooftop helipad. The route has 12 moving hazards: crossing and inspection UAVs, plus suspended construction platforms. Sensor Mode generates up to four A* corridor candidates: direct, north, south, and higher-altitude. The AI selects the lowest estimated cost using route length, turn smoothness, and altitude effort; it follows that selected A* resultant through the moving-hazard field. A simulated nine-beam LiDAR scan supplies noisy obstacle position samples, and a three-axis constant-velocity Kalman filter estimates position and velocity. LiDAR tracks obstacles from farther away, but only triggers a reroute when the estimated gap from the drone hull to a conflicting obstacle surface is about 2 m. D* Lite then computes a local detour to a point 60 m beyond the predicted encounter, and the drone rejoins the selected A* path. Unseen course hazards remain unknown to the planner until sensed. Safe Mode plans an A* clearance route around the known obstacle map before flight. Both modes land on the rooftop pad and stop the rotors instead of hovering.

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
- Sensor Mode builds up to four A* candidates (direct, north, south, and high). The AI selects the lowest-cost candidate using distance, turn smoothness, and altitude effort; the selected resultant passes through the dynamic hazard zone. Alternative candidates remain visible as faint route lines.
- LiDAR continuously tracks obstacles at range, while collision-risk alerts are gated until the estimated drone-hull-to-obstacle-surface gap is about 2 m. If the projected motion still conflicts with the route, D* Lite detours around the sensed obstacle, rejoins the selected A* path 60 m beyond the predicted encounter, and follows it to the goal. The grid planner permits expansion from the drone's physically clear start position even when its snapped grid cell falls inside the hazard inflation zone. If D* cannot extract a route, the mission falls back to A* recovery to the same rejoin point, then to the goal, rather than stopping in a hold state. The HUD counts unique course hazards sensed out of 12.
- LiDAR uses nine forward-facing horizontal beams across a 64° field of view, 180 m range, and seeded 1.5 m standard-deviation measurement noise. The Kalman filter estimates all three position and velocity components from those returns; collision-risk checks use the estimates, not the obstacle's scripted future position. The 2 m trigger is measured from the estimated drone hull to the object's surface; the 3 m predictive envelope remains part of the future collision check.
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
