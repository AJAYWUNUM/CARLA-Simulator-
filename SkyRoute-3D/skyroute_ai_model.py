"""SkyRoute AI model: LiDAR, Kalman tracking, collision risk, A* and D* Lite.

This module contains the reportable algorithm layer without the OpenGL city or
windowing code. It uses only Python's standard library and accepts obstacles
through a small attribute interface (``name``, ``kind``, ``center`` and
``size``), so the same model can be reused in another simulator.

The tracking state is [position, velocity] for each Cartesian axis. The
constant-velocity transition is x(k+1) = F(dt) x(k), and LiDAR measures
position only. Collision assessment projects the filtered state forward and
compares its closest approach with the drone's swept safety envelope.
"""
from __future__ import annotations

import heapq
import math
import random
from typing import Callable, Iterable, Sequence

Vector3 = tuple[float, float, float]
Cell3 = tuple[int, int, int]


def _sub(a: Sequence[float], b: Sequence[float]) -> Vector3:
    return tuple(float(a[i]) - float(b[i]) for i in range(3))


def _dot(a: Sequence[float], b: Sequence[float]) -> float:
    return sum(float(a[i]) * float(b[i]) for i in range(3))


def _norm(vector: Sequence[float]) -> float:
    return math.sqrt(_dot(vector, vector))


def _scale(vector: Sequence[float], scalar: float) -> Vector3:
    return tuple(float(value) * scalar for value in vector)


def _unit(vector: Sequence[float]) -> Vector3:
    magnitude = _norm(vector)
    return _scale(vector, 1.0 / magnitude) if magnitude > 1e-9 else (0.0, 0.0, 0.0)


def obstacle_surface_clearance(obstacle: object, drone_position: Sequence[float],
                               obstacle_position: Sequence[float] | None = None,
                               drone_radius: float = 0.0) -> float:
    """Estimate the gap from the drone hull to an obstacle's nearest surface."""
    center = obstacle.center if obstacle_position is None else obstacle_position
    if obstacle.kind in ("aircraft", "sphere"):
        return max(0.0, _norm(_sub(drone_position, center))
                   - float(obstacle.size) - float(drone_radius))
    half_x, half_y, half_z = obstacle.size
    gap = (
        max(0.0, abs(float(drone_position[0]) - float(center[0])) - float(half_x)),
        max(0.0, abs(float(drone_position[1]) - float(center[1])) - float(half_y)),
        max(0.0, abs(float(drone_position[2]) - float(center[2])) - float(half_z)),
    )
    return max(0.0, _norm(gap) - float(drone_radius))


def score_path_cost(path: Sequence[Sequence[float]], turn_weight: float = 24.0,
                    climb_weight: float = 1.5) -> float:
    """Score route length plus distance-equivalent turn and altitude effort."""
    if not path:
        return math.inf
    if len(path) == 1:
        return 0.0
    length_cost = sum(_norm(_sub(b, a)) for a, b in zip(path, path[1:]))
    altitude_effort = sum(abs(float(b[2]) - float(a[2])) for a, b in zip(path, path[1:]))
    turn_cost = 0.0
    for a, b, c in zip(path, path[1:], path[2:]):
        incoming = _unit(_sub(b, a))
        outgoing = _unit(_sub(c, b))
        alignment = max(-1.0, min(1.0, _dot(incoming, outgoing)))
        turn_cost += float(turn_weight) * (1.0 - alignment)
    return length_cost + float(climb_weight) * altitude_effort + turn_cost


def choose_best_path(candidates: dict[str, Sequence[Sequence[float]]]):
    """Return the lowest-cost named route and scores for all usable candidates."""
    scores = {name: score_path_cost(path) for name, path in candidates.items()
              if len(path) >= 2}
    if not scores:
        return None, [], {}
    selected = min(scores, key=scores.get)
    return selected, list(candidates[selected]), scores


class AxisKalmanFilter:
    """One-dimensional constant-velocity Kalman filter.

    ``measurement_variance`` is LiDAR position-noise variance. Process noise
    represents unmodelled acceleration, allowing the velocity estimate to
    adapt when a moving obstacle changes direction.
    """

    def __init__(self, position: float, measurement_variance: float = 4.0,
                 acceleration_variance: float = 9.0) -> None:
        self.position = float(position)
        self.velocity = 0.0
        self.p00, self.p01, self.p10, self.p11 = 25.0, 0.0, 0.0, 100.0
        self.measurement_variance = max(1e-9, float(measurement_variance))
        self.acceleration_variance = max(0.0, float(acceleration_variance))

    def update(self, measurement: float, dt: float) -> None:
        dt = max(1e-3, float(dt))
        q = self.acceleration_variance
        old00, old01, old10, old11 = self.p00, self.p01, self.p10, self.p11

        # Predict with F=[[1, dt], [0, 1]] and white-acceleration process noise.
        self.position += self.velocity * dt
        self.p00 = old00 + dt * (old01 + old10) + dt * dt * old11 + q * dt**4 * 0.25
        self.p01 = old01 + dt * old11 + q * dt**3 * 0.5
        self.p10 = old10 + dt * old11 + q * dt**3 * 0.5
        self.p11 = old11 + q * dt * dt

        # Correct from a scalar position observation H=[1, 0].
        pred00, pred01, pred10, pred11 = self.p00, self.p01, self.p10, self.p11
        innovation = float(measurement) - self.position
        innovation_variance = max(1e-9, pred00 + self.measurement_variance)
        gain_position = pred00 / innovation_variance
        gain_velocity = pred10 / innovation_variance
        self.position += gain_position * innovation
        self.velocity += gain_velocity * innovation
        self.p00 = max(1e-9, (1.0 - gain_position) * pred00)
        self.p01 = (1.0 - gain_position) * pred01
        self.p10 = pred10 - gain_velocity * pred00
        self.p11 = max(1e-9, pred11 - gain_velocity * pred01)


class ObstacleTrack:
    """Three independent position/velocity filters for one detected object."""

    def __init__(self, measurement: Sequence[float], time_seconds: float,
                 measurement_variance: float = 4.0) -> None:
        self.axes = [AxisKalmanFilter(measurement[i], measurement_variance) for i in range(3)]
        self.last_seen_time = float(time_seconds)

    def update(self, measurement: Sequence[float], time_seconds: float) -> None:
        dt = max(1e-3, float(time_seconds) - self.last_seen_time)
        for index, axis_filter in enumerate(self.axes):
            axis_filter.update(measurement[index], dt)
        self.last_seen_time = float(time_seconds)

    @property
    def position(self) -> Vector3:
        return tuple(axis.position for axis in self.axes)

    @property
    def velocity(self) -> Vector3:
        return tuple(axis.velocity for axis in self.axes)

    def predict(self, horizon_seconds: float) -> Vector3:
        horizon = max(0.0, float(horizon_seconds))
        return tuple(axis.position + axis.velocity * horizon for axis in self.axes)


class LiDARKalmanModel:
    """Forward LiDAR fan with noisy returns and persistent Kalman tracks."""

    def __init__(self, beam_angles: Sequence[float] = (-32, -24, -16, -8, 0, 8, 16, 24, 32),
                 beam_half_angle: float = 5.0, max_range: float = 180.0,
                 measurement_std: float = 1.5, seed: int = 41218) -> None:
        if not beam_angles:
            raise ValueError("At least one LiDAR beam angle is required")
        self.beam_angles = tuple(float(angle) for angle in beam_angles)
        self.beam_half_angle = max(0.0, float(beam_half_angle))
        self.max_range = max(0.0, float(max_range))
        self.measurement_std = max(0.0, float(measurement_std))
        self.seed = int(seed)
        self.rng = random.Random(self.seed)
        self.tracks: dict[object, ObstacleTrack] = {}
        self.return_points: list[tuple[object, Vector3]] = []

    def reset(self) -> None:
        """Clear tracks and restart the seeded sensor noise sequence."""
        self.tracks.clear()
        self.return_points.clear()
        self.rng = random.Random(self.seed)

    def scan(self, obstacles: Iterable[object], origin: Sequence[float],
             forward: Sequence[float], time_seconds: float) -> list[tuple[object, ObstacleTrack]]:
        """Cast each beam, keep its nearest return, and update visible tracks.

        An obstacle supplies ``center`` and either a scalar ``size`` (sphere)
        or a three-value half-extent tuple (box). The detected object centroid
        is perturbed by Gaussian LiDAR noise before it enters the filter.
        """
        forward_xy = _unit((forward[0], forward[1], 0.0))
        if _norm(forward_xy) < 1e-8:
            forward_xy = (1.0, 0.0, 0.0)

        nearest_by_beam: dict[float, tuple[float, object]] = {}
        for obstacle in obstacles:
            delta = _sub(obstacle.center, origin)
            horizontal_range = math.hypot(delta[0], delta[1])
            range_3d = _norm(delta)
            if range_3d < 1e-6 or range_3d > self.max_range:
                continue

            if horizontal_range < 1e-6:
                angle = 0.0
            else:
                forward_dot = forward_xy[0] * delta[0] + forward_xy[1] * delta[1]
                cross_track = forward_xy[0] * delta[1] - forward_xy[1] * delta[0]
                angle = math.degrees(math.atan2(cross_track, forward_dot))
            nearest_beam = min(self.beam_angles, key=lambda sample: abs(sample - angle))
            if angle < self.beam_angles[0] - self.beam_half_angle:
                continue
            if angle > self.beam_angles[-1] + self.beam_half_angle:
                continue
            if abs(angle - nearest_beam) > self.beam_half_angle:
                continue

            size = obstacle.size
            if isinstance(size, (tuple, list)):
                vertical_extent = float(size[2])
                radius_xy = math.hypot(float(size[0]), float(size[1]))
            else:
                vertical_extent = radius_xy = float(size)
            if abs(delta[2]) > vertical_extent + 8.0:
                continue

            angular_radius = math.degrees(math.asin(min(0.98, radius_xy / max(horizontal_range, 1e-6))))
            for beam in self.beam_angles:
                beam_delta = angle - beam
                if abs(beam_delta) > self.beam_half_angle + angular_radius:
                    continue
                ray_distance = max(0.0, horizontal_range * math.cos(math.radians(beam_delta)) - radius_xy)
                if ray_distance > self.max_range:
                    continue
                current = nearest_by_beam.get(beam)
                if current is None or ray_distance < current[0]:
                    nearest_by_beam[beam] = (ray_distance, obstacle)

        visible = sorted({obstacle for _, obstacle in nearest_by_beam.values()},
                         key=lambda item: item.name)
        detections: list[tuple[object, ObstacleTrack]] = []
        variance = max(1e-9, self.measurement_std**2)
        for obstacle in visible:
            measurement = tuple(float(obstacle.center[i]) + self.rng.gauss(0.0, self.measurement_std)
                                for i in range(3))
            track = self.tracks.get(obstacle)
            if track is None:
                track = ObstacleTrack(measurement, time_seconds, variance)
                self.tracks[obstacle] = track
            else:
                track.update(measurement, time_seconds)
            detections.append((obstacle, track))

        self.return_points = sorted(((obstacle, track.position) for obstacle, track in detections),
                                    key=lambda item: _norm(_sub(item[1], origin)))
        return detections


def assess_collision_risk(detections: Iterable[tuple[object, ObstacleTrack]],
                          drone_position: Sequence[float], forward: Sequence[float],
                          cruise_speed: float, sensor_range: float, drone_radius: float,
                          safety_clearance: float, cooldowns: dict | None = None,
                          time_seconds: float = 0.0,
                          current_trigger_distance: float | None = None):
    """Return the nearest predicted conflict as ``(obstacle, position, t, drone_future)``.

    A conflict is reported only when the constant-velocity obstacle estimate
    predicts the drone's projected route to pass within the obstacle's shape,
    drone radius, and configured safety clearance.
    """
    forward = _unit(forward)
    if _norm(forward) < 1e-8:
        return None
    cooldowns = cooldowns or {}
    nearest = None
    nearest_distance = float(sensor_range) + 1.0
    nearest_prediction = None
    speed = max(1.0, float(cruise_speed))
    max_horizon = float(sensor_range) / speed

    for obstacle, track in detections:
        if cooldowns.get(obstacle, 0.0) > time_seconds:
            continue
        if (current_trigger_distance is not None and
                obstacle_surface_clearance(obstacle, drone_position, track.position,
                                           drone_radius) > float(current_trigger_distance)):
            continue
        delta = _sub(track.position, drone_position)
        along = _dot(delta, forward)
        if along <= 0.0 or along > sensor_range:
            continue
        horizon = min(max(along / speed, 0.0), max_horizon)
        predicted = None
        for _ in range(3):
            predicted = track.predict(horizon)
            projected = _sub(predicted, drone_position)
            predicted_along = _dot(projected, forward)
            if predicted_along <= 0.0 or predicted_along > sensor_range:
                predicted = None
                break
            horizon = min(max(predicted_along / speed, 0.0), max_horizon)
        if predicted is None:
            continue

        future_drone = tuple(float(drone_position[i]) + forward[i] * speed * horizon for i in range(3))
        miss = _sub(predicted, future_drone)
        future_along = _dot(miss, forward)
        side_vector = tuple(miss[i] - forward[i] * future_along for i in range(3))
        side_distance = _norm(side_vector)
        margin = float(drone_radius) + float(safety_clearance)
        if obstacle.kind in ("aircraft", "sphere"):
            vertical_extent = lateral_extent = float(obstacle.size) + margin
        else:
            hx, hy, hz = obstacle.size
            vertical_extent = float(hz) + margin
            lateral_extent = max(float(hx), float(hy)) + margin
        if abs(miss[2]) > vertical_extent or side_distance > lateral_extent or abs(future_along) > 10.0:
            continue

        current_range = _norm(delta)
        if current_range < nearest_distance:
            nearest = obstacle
            nearest_distance = current_range
            nearest_prediction = (predicted, horizon, future_drone)

    if nearest_prediction is None:
        return None
    return (nearest, *nearest_prediction)


def astar_3d(start: Sequence[float], goal: Sequence[float], world_min: Sequence[float],
             world_max: Sequence[float], step: float,
             collides: Callable[[Sequence[float], float], object],
             segment_clear: Callable[[Sequence[float], Sequence[float]], bool]) -> list[Vector3]:
    """Grid-based 26-neighbour A* with line-of-sight path smoothing."""
    step = float(step)
    if step <= 0.0:
        raise ValueError("Grid step must be greater than zero")
    dims = tuple(int(round((world_max[i] - world_min[i]) / step)) + 1 for i in range(3))

    def to_cell(point: Sequence[float]) -> Cell3:
        return tuple(int(round((point[i] - world_min[i]) / step)) for i in range(3))

    def to_point(cell: Cell3) -> Vector3:
        return tuple(float(world_min[i]) + cell[i] * step for i in range(3))

    start_cell, goal_cell = to_cell(start), to_cell(goal)
    if collides(start, 0.25) or collides(goal, 0.25):
        return []
    offsets = [(dx, dy, dz) for dx in (-1, 0, 1) for dy in (-1, 0, 1)
               for dz in (-1, 0, 1) if (dx, dy, dz) != (0, 0, 0)]
    opened = [(0.0, 0.0, start_cell)]
    came_from: dict[Cell3, Cell3] = {}
    costs = {start_cell: 0.0}
    closed: set[Cell3] = set()

    while opened:
        _, current_cost, current = heapq.heappop(opened)
        if current in closed:
            continue
        if current == goal_cell:
            cells = [current]
            while current in came_from:
                current = came_from[current]
                cells.append(current)
            cells.reverse()
            raw = [tuple(start)] + [to_point(cell) for cell in cells[1:-1]] + [tuple(goal)]
            smooth = [raw[0]]
            anchor = 0
            while anchor < len(raw) - 1:
                furthest = anchor + 1
                for candidate in range(len(raw) - 1, anchor, -1):
                    if segment_clear(raw[anchor], raw[candidate]):
                        furthest = candidate
                        break
                smooth.append(raw[furthest])
                anchor = furthest
            return smooth

        closed.add(current)
        for offset in offsets:
            nxt = tuple(current[i] + offset[i] for i in range(3))
            if any(nxt[i] < 0 or nxt[i] >= dims[i] for i in range(3)) or nxt in closed:
                continue
            point = to_point(nxt)
            if collides(point, 0.53):
                continue
            edge_cost = math.sqrt(sum(value * value for value in offset)) * step
            new_cost = current_cost + edge_cost
            if new_cost >= costs.get(nxt, math.inf):
                continue
            costs[nxt] = new_cost
            came_from[nxt] = current
            heuristic = math.sqrt(sum((point[i] - goal[i]) ** 2 for i in range(3)))
            heapq.heappush(opened, (new_cost + heuristic, new_cost, nxt))
    return []


class DStarLitePlanner:
    """Incremental 6-neighbour D* Lite planner for changing occupancy grids."""

    def __init__(self, dims: Cell3, step: float, start: Cell3, goal: Cell3) -> None:
        self.dims, self.step = dims, float(step)
        self.start = self.last_start = start
        self.goal = goal
        self.km = 0.0
        self.g: dict[Cell3, float] = {}
        self.rhs: dict[Cell3, float] = {goal: 0.0}
        self.blocked: set[Cell3] = set()
        self.queue: list[tuple[float, float, int, Cell3, int]] = []
        self.versions: dict[Cell3, int] = {}
        self.serial = 0
        self.initialized = False
        self._push(goal)

    @staticmethod
    def value(table: dict[Cell3, float], node: Cell3) -> float:
        return table.get(node, math.inf)

    def heuristic(self, a: Cell3, b: Cell3) -> float:
        return self.step * math.sqrt(sum((a[i] - b[i]) ** 2 for i in range(3)))

    def calculate_key(self, node: Cell3) -> tuple[float, float]:
        best = min(self.value(self.g, node), self.value(self.rhs, node))
        return best + self.heuristic(self.start, node) + self.km, best

    def neighbors(self, node: Cell3):
        for offset in ((1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)):
            neighbor = tuple(node[i] + offset[i] for i in range(3))
            if all(0 <= neighbor[i] < self.dims[i] for i in range(3)):
                yield neighbor

    def edge_cost(self, a: Cell3, b: Cell3) -> float:
        return self.heuristic(a, b)

    def _push(self, node: Cell3, version: int | None = None) -> None:
        if version is None:
            version = self.versions.get(node, 0) + 1
            self.versions[node] = version
        k1, k2 = self.calculate_key(node)
        self.serial += 1
        heapq.heappush(self.queue, (k1, k2, self.serial, node, version))

    def _top_key(self) -> tuple[float, float]:
        while self.queue:
            k1, k2, _, node, version = self.queue[0]
            if self.versions.get(node) != version:
                heapq.heappop(self.queue)
                continue
            return k1, k2
        return math.inf, math.inf

    def _pop(self):
        while self.queue:
            k1, k2, _, node, version = heapq.heappop(self.queue)
            if self.versions.get(node) == version:
                return (k1, k2), node
        return (math.inf, math.inf), None

    def update_vertex(self, node: Cell3) -> None:
        if node != self.goal:
            if node in self.blocked:
                self.rhs[node] = math.inf
            else:
                self.rhs[node] = min(
                    (self.edge_cost(node, neighbor) + self.value(self.g, neighbor)
                     for neighbor in self.neighbors(node) if neighbor not in self.blocked),
                    default=math.inf,
                )
        version = self.versions.get(node, 0) + 1
        self.versions[node] = version
        if self.value(self.g, node) != self.value(self.rhs, node):
            self._push(node, version)

    def update_blocked(self, new_blocked: Iterable[Cell3]) -> None:
        new_blocked = set(new_blocked)
        changed = self.blocked ^ new_blocked
        self.blocked = new_blocked
        if not self.initialized:
            affected = set(self.blocked)
            for node in self.blocked:
                affected.update(self.neighbors(node))
            self.initialized = True
        else:
            affected = set(changed)
            for node in changed:
                affected.update(self.neighbors(node))
        for node in affected:
            self.update_vertex(node)

    def move_start(self, new_start: Cell3) -> None:
        if new_start != self.start:
            self.km += self.heuristic(self.last_start, new_start)
            self.last_start = new_start
            self.start = new_start

    def compute_shortest_path(self) -> None:
        start_key = self.calculate_key(self.start)
        while self._top_key() < start_key or self.value(self.rhs, self.start) != self.value(self.g, self.start):
            old_key, node = self._pop()
            if node is None:
                break
            new_key = self.calculate_key(node)
            if old_key < new_key:
                self._push(node)
            elif self.value(self.g, node) > self.value(self.rhs, node):
                self.g[node] = self.value(self.rhs, node)
                for predecessor in self.neighbors(node):
                    self.update_vertex(predecessor)
            else:
                self.g[node] = math.inf
                self.update_vertex(node)
                for predecessor in self.neighbors(node):
                    self.update_vertex(predecessor)
            start_key = self.calculate_key(self.start)

    def extract_path(self) -> list[Cell3]:
        if self.start in self.blocked or self.goal in self.blocked:
            return []
        self.compute_shortest_path()
        if math.isinf(self.value(self.g, self.start)):
            return []
        path = [self.start]
        seen = {self.start}
        current = self.start
        for _ in range(math.prod(self.dims)):
            if current == self.goal:
                return path
            choices = []
            for neighbor in self.neighbors(current):
                if neighbor in self.blocked:
                    continue
                score = self.edge_cost(current, neighbor) + self.value(self.g, neighbor)
                if not math.isinf(score):
                    choices.append((score, self.heuristic(neighbor, self.goal), neighbor))
            if not choices:
                return []
            current = min(choices)[2]
            if current in seen:
                return []
            seen.add(current)
            path.append(current)
        return []


__all__ = [
    "AxisKalmanFilter", "ObstacleTrack", "LiDARKalmanModel",
    "obstacle_surface_clearance", "score_path_cost", "choose_best_path",
    "assess_collision_risk", "astar_3d", "DStarLitePlanner",
]
