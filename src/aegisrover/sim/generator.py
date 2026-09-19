"""Seed-driven batch scenario generation with physically feasible fault layouts.

Writing every scenario by hand does not scale across arena sizes, robot counts and
fault combinations. This module turns a *parameter range* plus a seed into a batch
of :class:`~aegisrover.sim.scenarios.ScenarioSpec` objects.

Three guarantees make generated batches useful in a test suite:

* **Reproducible.** Every scenario gets a deterministic child seed derived from the
  master seed and its batch index, and every random draw goes through that child RNG.
  The same request always yields byte-identical specs (same scenario digest).
* **Feasible.** A layout is rejected when robots overlap, teleport targets leave the
  arena or land on another robot, commanded twists violate the drive envelope or its
  minimum turning radius, or the configured fault sequence cannot fit the available
  time window. Infeasible *requests* (ranges that cannot be satisfied even with
  retries) raise before a single scenario is emitted.
* **Comparable.** Each spec carries a machine-readable manifest (parameters, layout
  and fault signature) in ``notes`` and :func:`diff_generated` reports exactly which
  of those fields, events or poses differ between two scenarios.

Only event kinds understood by
:func:`aegisrover.sim.scenarios._apply_event` are emitted (``set_twist``, ``stop``,
``teleport``), so generated specs replay without extra vocabulary.
"""
from __future__ import annotations

import hashlib
import json
import math
import random
from dataclasses import dataclass, field, replace
from typing import Iterable, Sequence

from aegisrover.sim.scenarios import ScenarioSpec, compute_digest

__all__ = (
    'GeneratorConfig', 'GenerationParams', 'GeneratedScenario', 'GeneratedBatch',
    'GeneratorError', 'generate_batch', 'diff_generated', 'batch_coverage',
    'save_batch', 'validate_spec', 'ROBOT_RADIUS',
)

#: Conservative collision radius used for layout feasibility checks. The kinematic
#: :class:`~aegisrover.sim.world.World` itself has no obstacle model, so this is the
#: envelope the generator guarantees rather than something the engine enforces.
ROBOT_RADIUS = 0.25


class GeneratorError(ValueError):
    """Raised when a generation request is infeasible or generation exhausts retries."""


@dataclass(frozen=True)
class DriveEnvelope:
    """Physical speed envelope of a robot.

    ``min_turn_radius`` couples linear and angular speed: a twist is only feasible
    when ``abs(linear) / abs(angular) >= min_turn_radius`` (straight motion exempt).
    """

    max_linear: float = 1.0
    max_angular: float = 1.0
    min_turn_radius: float = 0.0

    def allows(self, linear: float, angular: float) -> bool:
        if abs(linear) > self.max_linear + 1e-9 or abs(angular) > self.max_angular + 1e-9:
            return False
        if self.min_turn_radius > 0.0 and abs(angular) > 1e-12:
            if abs(linear) / abs(angular) < self.min_turn_radius - 1e-9:
                return False
        return True

    def validate(self) -> None:
        if self.max_linear <= 0 or self.max_angular <= 0 or self.min_turn_radius < 0:
            raise GeneratorError('drive envelope limits must be non-negative')
        # The tightest achievable turn at maximum angular speed must be feasible at
        # some linear speed the envelope allows; otherwise no turning command exists.
        if self.min_turn_radius > 0 and self.min_turn_radius * self.max_angular > self.max_linear + 1e-9:
            raise GeneratorError(
                f'min turn radius {self.min_turn_radius} is unreachable: '
                f'max_angular {self.max_angular} needs linear >= '
                f'{self.min_turn_radius * self.max_angular:.3f} > max_linear {self.max_linear}')


@dataclass(frozen=True)
class GeneratorConfig:
    """Ranges sampled independently for each scenario in a batch."""

    seed: int = 0
    count: int = 10
    duration_range: tuple[float, float] = (5.0, 20.0)
    step: float = 0.1
    arena_range: tuple[float, float] = (4.0, 10.0)
    robots_range: tuple[int, int] = (1, 3)
    faults_range: tuple[int, int] = (0, 4)
    envelope: DriveEnvelope = field(default_factory=DriveEnvelope)
    #: Fraction of the timeline reserved for fault events (events avoid the edges).
    event_window: tuple[float, float] = (0.1, 0.9)
    #: Probability weight of each fault kind; normalised internally.
    kind_weights: dict = field(default_factory=lambda: {'set_twist': 0.6,
                                                        'teleport': 0.2, 'stop': 0.2})
    #: Minimum gap between events touching the same robot, in seconds.
    min_event_gap: float = 0.5
    spawn_margin: float = ROBOT_RADIUS
    max_attempts: int = 200

    def validate(self) -> None:
        if self.count <= 0:
            raise GeneratorError('count must be positive')
        if self.step <= 0:
            raise GeneratorError('step must be positive')
        _check_range(self.duration_range, 'duration_range', positive=True)
        if self.duration_range[0] < self.step:
            raise GeneratorError('shortest duration must cover at least one step')
        _check_range(self.arena_range, 'arena_range', positive=True)
        _check_int_range(self.robots_range, 'robots_range', minimum=1)
        _check_int_range(self.faults_range, 'faults_range', minimum=0)
        lo, hi = self.event_window
        if not (0.0 <= lo < hi <= 1.0):
            raise GeneratorError('event_window fractions must satisfy 0 <= lo < hi <= 1')
        if self.min_event_gap < 0:
            raise GeneratorError('min_event_gap must be non-negative')
        if self.spawn_margin < 0:
            raise GeneratorError('spawn_margin must be non-negative')
        if self.max_attempts <= 0:
            raise GeneratorError('max_attempts must be positive')
        weights = {k: float(v) for k, v in self.kind_weights.items()}
        unknown = set(weights) - {'set_twist', 'teleport', 'stop'}
        if unknown:
            raise GeneratorError(f'unsupported fault kinds: {sorted(unknown)}')
        if not weights or sum(weights.values()) <= 0:
            raise GeneratorError('kind_weights needs at least one positive weight')
        self.envelope.validate()
        # Worst-case layout feasibility: the most robots on the smallest arena must
        # admit margin-separated points. Use the guaranteed axis-aligned grid
        # capacity (a conservative sufficient condition; rejection sampling handles
        # the denser irregular cases).
        min_arena = self.arena_range[0]
        most_robots = self.robots_range[1]
        usable = max(0.0, min_arena - 2 * self.spawn_margin)
        # Axis-aligned grid capacity: points sit at margin + i*2r inside the
        # usable length. A grid that only fills the length exactly is rejected
        # (rejection sampling would practically never hit exact tangencies), so an
        # extra spacing of slack is required once the capacity is reached.
        spacing = 2 * ROBOT_RADIUS
        per_row = math.floor((usable - spacing) / spacing + 1e-9) + 1 if usable >= spacing else 1
        if per_row * per_row < most_robots:
            raise GeneratorError(
                f'arena {min_arena:g}m cannot fit {most_robots} robots with margin '
                f'{self.spawn_margin:g}')
        # Worst-case scheduling feasibility on the shortest timeline.
        min_duration = self.duration_range[0]
        window = (self.event_window[1] - self.event_window[0]) * min_duration
        slots = int(window // max(self.min_event_gap, self.step)) + 1
        if slots < self.faults_range[1]:
            raise GeneratorError(
                f'{self.faults_range[1]} faults cannot fit a {min_duration:g}s window '
                f'with gap {self.min_event_gap:g}s')


@dataclass(frozen=True)
class GenerationParams:
    """The concrete parameter draw one scenario was generated from."""

    index: int
    child_seed: int
    duration: float
    arena: float
    robot_count: int
    fault_count: int
    step: float

    def to_dict(self) -> dict:
        return {'index': self.index, 'seed': self.child_seed, 'master_seed': None,
                'duration': self.duration, 'arena': self.arena,
                'robots': self.robot_count, 'faults': self.fault_count, 'step': self.step}


@dataclass(frozen=True)
class GeneratedScenario:
    spec: ScenarioSpec
    params: GenerationParams
    #: Ordered ``(time, kind, robot)`` triples used as the compact fault signature.
    fault_signature: tuple[tuple, ...]

    def manifest(self) -> dict:
        return {
            'scenario_id': self.spec.scenario_id,
            'digest': self.spec.digest,
            'params': _with_master(self.params),
            'fault_signature': [list(item) for item in self.fault_signature],
        }


@dataclass(frozen=True)
class GeneratedBatch:
    master_seed: int
    config: GeneratorConfig
    scenarios: tuple[GeneratedScenario, ...]

    def manifests(self) -> tuple[dict, ...]:
        return tuple(_with_master_scn(scn, self.master_seed) for scn in self.scenarios)

    def manifest_json(self) -> str:
        """Stable JSON of the whole batch; identical for identical requests."""
        return json.dumps({'master_seed': self.master_seed,
                           'scenarios': list(self.manifests())},
                          sort_keys=True, separators=(',', ':'), ensure_ascii=False)

    def specs(self) -> tuple[ScenarioSpec, ...]:
        return tuple(scn.spec for scn in self.scenarios)

    def diff(self, left_index: int, right_index: int) -> dict:
        return diff_generated(self.scenarios[left_index], self.scenarios[right_index])


def generate_batch(config: GeneratorConfig) -> GeneratedBatch:
    """Generate ``config.count`` feasible scenarios from the configured ranges.

    Raises :class:`GeneratorError` if the request is structurally infeasible or if
    feasibility retries are exhausted (which indicates overly tight ranges).
    """
    config.validate()
    produced: list[GeneratedScenario] = []
    for index in range(config.count):
        child_seed = _child_seed(config.seed, index)
        produced.append(_generate_one(config, index, child_seed))
    return GeneratedBatch(master_seed=config.seed, config=config, scenarios=tuple(produced))


def save_batch(store, batch: GeneratedBatch, *, actor: str = 'generator') -> tuple:
    """Persist every scenario in *batch* through a :class:`ScenarioStore`."""
    return tuple(store.save(scn.spec, actor=actor) for scn in batch.scenarios)


# ---------------------------------------------------------------------------- generation
def _generate_one(config: GeneratorConfig, index: int, child_seed: int) -> GeneratedScenario:
    rng = random.Random(child_seed)
    params = _draw_params(config, index, child_seed, rng)
    name = f'gen_{config.seed}_{index:04d}'
    robots = _layout_robots(config, params, rng)
    events = _layout_faults(config, params, rng, robots)
    spec = ScenarioSpec(
        scenario_id=name,
        revision=0,
        duration=params.duration,
        step=config.step,
        seed=child_seed,
        robots=tuple(robots),
        events=tuple(events),
        notes=json.dumps({'generated': _with_master(params, config.seed)},
                         sort_keys=True, separators=(',', ':'), ensure_ascii=False),
    )
    validate_spec(spec, config.envelope, params.arena)
    spec = replace(spec, digest=compute_digest(spec.normalised()))
    signature = tuple((e['time'], e['kind'], _event_robot(e)) for e in spec.events)
    return GeneratedScenario(spec=spec, params=params, fault_signature=signature)


def _draw_params(config: GeneratorConfig, index: int, child_seed: int,
                 rng: random.Random) -> GenerationParams:
    duration = _quantize(rng.uniform(*config.duration_range), config.step)
    duration = max(duration, config.step)
    arena = round(rng.uniform(*config.arena_range), 6)
    robot_count = rng.randint(*config.robots_range)
    fault_count = rng.randint(*config.faults_range)
    return GenerationParams(index=index, child_seed=child_seed, duration=duration,
                            arena=arena, robot_count=robot_count,
                            fault_count=fault_count, step=config.step)


def _layout_robots(config: GeneratorConfig, params: GenerationParams,
                   rng: random.Random) -> list[dict]:
    margin = config.spawn_margin
    bound = params.arena - margin
    if bound <= margin:
        raise GeneratorError(f'arena {params.arena:g} leaves no spawn area')
    poses: list[tuple[float, float]] = []
    for _ in range(params.robot_count):
        point = _sample_point(rng, margin, bound, poses, attempts=config.max_attempts)
        poses.append(point)
    return [{'name': f'r{i + 1}',
             'pose': [round(x, 6), round(y, 6), round(rng.uniform(-3.14159, 3.14159), 6)],
             'twist': [0.0, 0.0]} for i, (x, y) in enumerate(poses)]


def _layout_faults(config: GeneratorConfig, params: GenerationParams,
                   rng: random.Random, robots: list[dict]) -> list[dict]:
    if params.fault_count == 0:
        return []
    window_lo = params.duration * config.event_window[0]
    window_hi = params.duration * config.event_window[1]
    gap = max(config.min_event_gap, config.step)
    kinds, cumulative = _weighted_kinds(config.kind_weights)
    names = [r['name'] for r in robots]
    occupied: dict[str, list[float]] = {n: [] for n in names}
    events: list[dict] = []

    for _ in range(params.fault_count):
        event = None
        for _ in range(config.max_attempts):
            kind = kinds[_weighted_index(rng, cumulative)]
            robot_name = names[rng.randrange(len(names))]
            time = _quantize(rng.uniform(window_lo, window_hi), config.step)
            candidate = _make_fault(kind, robot_name, time, rng, config, params,
                                    occupied, robots)
            if candidate is not None and _targets_free(occupied, candidate, gap):
                event = candidate
                break
        if event is None:
            raise GeneratorError(
                f'could not place fault within {config.max_attempts} attempts '
                f'(seed {params.child_seed}, index {params.index}); ranges too tight')
        for name in _event_targets(event):
            occupied[name].append(time)
        events.append(event)
    events.sort(key=lambda e: (e['time'], _event_robot(e), e['kind']))
    return events


def _make_fault(kind: str, robot_name: str, time: float, rng: random.Random,
                config: GeneratorConfig, params: GenerationParams,
                occupied: dict, robots: list[dict]) -> dict | None:
    envelope = config.envelope
    if kind == 'stop':
        others = [n for n in occupied if n != robot_name]
        return {'time': time, 'kind': 'stop',
                'payload': {'robots': [robot_name] + (others if rng.random() < 0.3 else [])}}
    if kind == 'set_twist':
        linear, angular = _sample_twist(rng, envelope)
        return {'time': time, 'kind': 'set_twist',
                'payload': {'robot': robot_name, 'linear': linear, 'angular': angular}}
    # teleport: target must stay inside the arena margin and clear every robot's
    # *initial* pose. Moving trajectories are not forward-simulated here, so the
    # generator guarantees geometric feasibility at the instant of the jump only.
    margin = config.spawn_margin
    existing = [tuple(r['pose'][:2]) for r in robots if r['name'] != robot_name]
    try:
        x, y = _sample_point(rng, margin, params.arena - margin, existing,
                             attempts=config.max_attempts)
    except GeneratorError:
        return None
    return {'time': time, 'kind': 'teleport',
            'payload': {'robot': robot_name,
                        'pose': [round(x, 6), round(y, 6),
                                 round(rng.uniform(-3.14159, 3.14159), 6)]}}


def _sample_twist(rng: random.Random, envelope: DriveEnvelope) -> tuple[float, float]:
    for _ in range(64):
        linear = round(rng.uniform(-envelope.max_linear, envelope.max_linear), 6)
        angular = round(rng.uniform(-envelope.max_angular, envelope.max_angular), 6)
        if rng.random() < 0.4:
            angular = 0.0  # pure translation is a common, always-feasible command
        if envelope.allows(linear, angular):
            return linear, angular
    # Fall back to pure translation at a feasible speed.
    return round(envelope.max_linear * (1.0 if rng.random() < 0.5 else -0.5), 6), 0.0


# ---------------------------------------------------------------------------- feasibility
def validate_spec(spec: ScenarioSpec, envelope: DriveEnvelope | None = None,
                  arena: float | None = None) -> None:
    """Reject specs that encode physically impossible combinations.

    Used both as a generation-time assertion and as a validator for hand-written or
    externally loaded specs before they are replayed in a batch.
    """
    if spec.duration <= 0 or spec.step <= 0:
        raise GeneratorError('duration and step must be positive')
    names = [r.get('name') for r in spec.robots]
    if any(not n for n in names) or len(set(names)) != len(names):
        raise GeneratorError('robots need unique non-empty names')
    envelope = envelope or DriveEnvelope()
    positions: dict[str, tuple[float, float]] = {}
    for robot in spec.robots:
        pose = robot.get('pose') or [0.0, 0.0, 0.0]
        twist = robot.get('twist') or [0.0, 0.0]
        if len(pose) != 3 or len(twist) != 2:
            raise GeneratorError(f"robot {robot.get('name')!r} has malformed pose/twist")
        x, y, _ = (float(v) for v in pose)
        if arena is not None and not _inside(x, y, arena):
            raise GeneratorError(f"robot {robot.get('name')!r} spawns outside the arena")
        if not envelope.allows(float(twist[0]), float(twist[1])):
            raise GeneratorError(f"robot {robot.get('name')!r} violates the drive envelope")
        for other, (ox, oy) in positions.items():
            if (x - ox) ** 2 + (y - oy) ** 2 < (2 * ROBOT_RADIUS) ** 2 - 1e-9:
                raise GeneratorError(
                    f"robots {robot.get('name')!r} and {other!r} overlap at spawn")
        positions[robot['name']] = (x, y)

    last_by_robot: dict[str, float] = {}
    for event in spec.events:
        kind, time, payload = event['kind'], float(event['time']), event.get('payload') or {}
        if not 0.0 <= time <= spec.duration:
            raise GeneratorError(f'event {kind!r} at t={time:g} is outside the timeline')
        robot = _event_robot(event)
        if robot is not None:
            if robot not in positions:
                raise GeneratorError(f'event {kind!r} targets unknown robot {robot!r}')
            previous = last_by_robot.get(robot)
            if previous is not None and time < previous - 1e-9:
                raise GeneratorError(f'events for {robot!r} are not time ordered')
            last_by_robot[robot] = time
        if kind == 'set_twist':
            if not envelope.allows(float(payload.get('linear', 0.0)),
                                   float(payload.get('angular', 0.0))):
                raise GeneratorError(
                    f'set_twist at t={time:g} violates the drive envelope')
        elif kind == 'teleport':
            pose = payload.get('pose')
            if not pose or len(pose) != 3:
                raise GeneratorError('teleport needs a 3-element pose')
            x, y, _ = (float(v) for v in pose)
            if arena is not None and not _inside(x, y, arena):
                raise GeneratorError(f'teleport at t={time:g} leaves the arena')
            for other, (ox, oy) in positions.items():
                if other == robot:
                    continue
                if (x - ox) ** 2 + (y - oy) ** 2 < (2 * ROBOT_RADIUS) ** 2 - 1e-9:
                    raise GeneratorError(
                        f'teleport at t={time:g} lands on robot {other!r}')
        elif kind == 'stop':
            for name in payload.get('robots', ()):  # missing => all robots, always valid
                if name not in positions:
                    raise GeneratorError(f'stop targets unknown robot {name!r}')
        else:
            raise GeneratorError(f'unsupported event kind {kind!r}')


def _inside(x: float, y: float, arena: float) -> bool:
    return -ROBOT_RADIUS <= x <= arena + ROBOT_RADIUS and \
        -ROBOT_RADIUS <= y <= arena + ROBOT_RADIUS


# ---------------------------------------------------------------------------- comparison
def diff_generated(left: GeneratedScenario, right: GeneratedScenario) -> dict:
    """Structural diff of two generated scenarios.

    Reports parameter changes, per-robot pose changes, events added/removed by their
    stable ``(time, kind, robot)`` key, and payload changes for shared keys, plus the
    replay digests so a run-level difference is visible even when the structure matches.
    """
    lp, rp = _with_master(left.params), _with_master(right.params)
    changed_params = {k: {'left': lp.get(k), 'right': rp.get(k)}
                      for k in sorted(set(lp) | set(rp)) if lp.get(k) != rp.get(k)}
    left_robots = {r['name']: r for r in left.spec.robots}
    right_robots = {r['name']: r for r in right.spec.robots}
    pose_changes = {}
    for name in sorted(set(left_robots) | set(right_robots)):
        if left_robots.get(name) != right_robots.get(name):
            pose_changes[name] = {'left': left_robots.get(name),
                                  'right': right_robots.get(name)}
    left_events = {_event_key(e): e for e in left.spec.events}
    right_events = {_event_key(e): e for e in right.spec.events}
    added = [list(k) for k in sorted(set(right_events) - set(left_events))]
    removed = [list(k) for k in sorted(set(left_events) - set(right_events))]
    payload_changes = {
        _fmt_key(k): {'left': left_events[k]['payload'], 'right': right_events[k]['payload']}
        for k in sorted(set(left_events) & set(right_events))
        if left_events[k]['payload'] != right_events[k]['payload']
    }
    return {
        'left': left.spec.scenario_id,
        'right': right.spec.scenario_id,
        'digests': {'left': left.spec.digest, 'right': right.spec.digest,
                    'equal': left.spec.digest == right.spec.digest},
        'changed_params': changed_params,
        'pose_changes': pose_changes,
        'events_added': added,
        'events_removed': removed,
        'payload_changes': payload_changes,
    }


def batch_coverage(batch: GeneratedBatch) -> dict:
    """Summarise which (robot_count, fault_count) combinations a batch covers."""
    combos: dict[tuple[int, int], int] = {}
    for scn in batch.scenarios:
        key = (scn.params.robot_count, scn.params.fault_count)
        combos[key] = combos.get(key, 0) + 1
    return {
        'count': len(batch.scenarios),
        'combinations': [{'robots': r, 'faults': f, 'scenarios': n}
                         for (r, f), n in sorted(combos.items())],
        'distinct_fault_signatures': len({scn.fault_signature for scn in batch.scenarios}),
    }


# ---------------------------------------------------------------------------- helpers
def _check_range(value: tuple, label: str, *, positive: bool = False) -> None:
    if len(value) != 2 or value[0] > value[1]:
        raise GeneratorError(f'{label} must be a (low, high) pair with low <= high')
    if positive and value[0] <= 0:
        raise GeneratorError(f'{label} must be positive')


def _check_int_range(value: tuple, label: str, *, minimum: int = 0) -> None:
    if len(value) != 2 or not all(isinstance(v, int) for v in value):
        raise GeneratorError(f'{label} must be two ints')
    if value[0] < minimum or value[0] > value[1]:
        raise GeneratorError(f'{label} invalid: need {minimum} <= low <= high')


def _quantize(value: float, step: float) -> float:
    return round(round(value / step) * step, 9)


def _sample_point(rng: random.Random, low: float, high: float,
                  existing: Sequence[tuple[float, float]], *, attempts: int) -> tuple[float, float]:
    for _ in range(attempts):
        x = round(rng.uniform(low, high), 6)
        y = round(rng.uniform(low, high), 6)
        if all((x - ox) ** 2 + (y - oy) ** 2 >= (2 * ROBOT_RADIUS) ** 2 - 1e-9
               for ox, oy in existing):
            return x, y
    raise GeneratorError(f'could not find non-overlapping point in {attempts} attempts')


def _slot_free(times: list[float], time: float, gap: float) -> bool:
    return all(abs(time - t) + 1e-9 >= gap for t in times)


def _weighted_kinds(weights: dict) -> tuple[list[str], list[float]]:
    kinds = sorted(weights)
    total = sum(float(weights[k]) for k in kinds)
    cumulative, running = [], 0.0
    for kind in kinds:
        running += float(weights[kind]) / total
        cumulative.append(running)
    return kinds, cumulative


def _weighted_index(rng: random.Random, cumulative: list[float]) -> int:
    draw = rng.random()
    for i, boundary in enumerate(cumulative):
        if draw <= boundary:
            return i
    return len(cumulative) - 1


def _event_robot(event: dict) -> str | None:
    payload = event.get('payload') or {}
    if 'robot' in payload:
        return payload['robot']
    robots = payload.get('robots')
    if robots:
        return robots[0]
    return None


def _event_targets(event: dict) -> list[str]:
    payload = event.get('payload') or {}
    if 'robot' in payload:
        return [payload['robot']]
    return list(payload.get('robots') or ())


def _targets_free(occupied: dict, event: dict, gap: float) -> bool:
    return all(_slot_free(occupied[name], event['time'], gap)
               for name in _event_targets(event))


def _event_key(event: dict) -> tuple:
    return (round(float(event['time']), 9), event['kind'], _event_robot(event))


def _fmt_key(key: tuple) -> str:
    return f't={key[0]:g}|{key[1]}|{key[2]}'


def _child_seed(master_seed: int, index: int) -> int:
    digest = hashlib.sha256(f'aegisrover.scenario:{master_seed}:{index}'.encode()).digest()
    return int.from_bytes(digest[:8], 'big')


def _with_master(params: GenerationParams, master: int | None = None) -> dict:
    data = params.to_dict()
    data['master_seed'] = master
    return data


def _with_master_scn(scn: GeneratedScenario, master: int) -> dict:
    manifest = scn.manifest()
    manifest['params']['master_seed'] = master
    return manifest
