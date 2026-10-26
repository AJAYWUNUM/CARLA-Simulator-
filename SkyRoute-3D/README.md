# SkyRoute Urban 3D — UAV Path Planning Microproject

SkyRoute is a native OpenGL desktop simulation for an autonomous drone mission in a generated downtown. The urban region is 4.2 km long by 1.05 km wide (4.41 km²), with 80 blocks, marked roads, sidewalks, detailed building facades and rooftops, moving cars, pedestrians, and a 12-hazard aerial course. The presentation keeps the bright white canvas while using richer city geometry and several camera views.

## Mission

The drone starts on a plaza launch pad, climbs to its flight corridor, then flies about 1.5 km to a rooftop helipad. The route has 12 moving hazards: crossing UAVs, sensor balloons, and suspended construction platforms. Sensor Mode uses D* Lite's resultant route as its active route from the start, while leaving unseen course hazards unknown. When sensors predict a collision risk, D* Lite incrementally updates the obstacle map and replaces the active route with a new resultant path around the detected hazard. Safe Mode plans an A* clearance route around the known obstacle map before flight. Both modes land on the rooftop pad and stop the rotors instead of hovering.

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
- 64 moving cars, 152 varied roadside trees, 14 sidewalk pedestrians and 12 moving course hazards. The hazards use seeded variation in direction, crossing axis, speed and distance so their motion is repeatable but not synchronized.
- Sensor Mode follows the D* Lite resultant path from takeoff. When the sensors predict collision risk, D* Lite updates the map and computes a new resultant path that becomes active; the HUD identifies it as the route in use. The HUD counts unique course hazards sensed out of 12.
- Safe Mode uses A* to plan a clearance route around known obstacles before takeoff.
- D* Lite follows Koenig and Likhachev, AAAI 2002: [D* Lite paper](https://ocs.aaai.org/Library/AAAI/2002/aaai02-072.php).
- Mission stages: launch, vertical takeoff, cruise, landing approach, touchdown and motors idle.
- Four-plus camera angles with a separate full-city view.

The city is procedural and has a Central Park-scale footprint; it is not a georeferenced New York map or the CARLA Unreal Engine renderer. Flight and sensor behavior are simplified for a microproject demonstration, not a flight controller.

## Project files

- `skyroute3d.py` — simulation, city generation, sensor logic and 3D renderer.
- `run_skyroute.bat` — Windows launcher.
- `requirements.txt` — pinned Pyglet dependency.
- `vendor/pyglet` — bundled windowing and OpenGL library.

