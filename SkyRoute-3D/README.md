# SkyRoute Urban 3D — UAV Path Planning Microproject

SkyRoute is a native OpenGL desktop simulation for an autonomous drone mission in a generated downtown. The urban region is 4.2 km long by 1.05 km wide (4.41 km²), with 80 blocks, marked roads, sidewalks, detailed building facades and rooftops, moving cars, pedestrians, and a 12-hazard aerial course. The presentation keeps the bright white canvas while using richer city geometry and several camera views.

## Mission

The drone starts on a plaza launch pad, climbs to its flight corridor, then flies about 1.5 km to a rooftop helipad. The route has 12 moving hazards: crossing UAVs, sensor balloons, and suspended construction platforms. Both launch choices fly the same direct sensor-controlled course; they differ only in the initial camera view. Forward sensors predict moving intersections, make a short local sidestep around each sensed hazard, then rejoin the direct line. The mission descends onto the landing pad and stops the rotors; the landed drone remains at the pad instead of hovering.

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

The camera label at the top right also cycles through the views when clicked.

## Simulation details

- City footprint: 4.2 km × 1.05 km (4.41 km²), with a 20-by-4 block grid.
- 80 varied buildings with multi-floor windows, reflective glazing, roof parapets and rooftop equipment.
- 22 moving vehicles, 14 sidewalk pedestrians and 12 moving course hazards. The hazards use seeded variation in direction, crossing axis, speed and distance so their motion is repeatable but not synchronized.
- Forward sensors predict near-future intersections and make a short local sidestep around each detected hazard before rejoining the original direct course. Upcoming detections are queued if they enter sensor range during a sidestep; the HUD counts unique course hazards sensed out of 12.
- Mission stages: launch, vertical takeoff, cruise, landing approach, touchdown and motors idle.
- Four-plus camera angles with a separate full-city view.

The city is procedural and has a Central Park-scale footprint; it is not a georeferenced New York map or the CARLA Unreal Engine renderer. Flight and sensor behavior are simplified for a microproject demonstration, not a flight controller.

## Project files

- `skyroute3d.py` — simulation, city generation, sensor logic and 3D renderer.
- `run_skyroute.bat` — Windows launcher.
- `requirements.txt` — pinned Pyglet dependency.
- `vendor/pyglet` — bundled windowing and OpenGL library.
