"""Seed-driven scenario generation: reproducible batches from parameter ranges.

Hand-writing one scenario per layout and fault combination does not scale, so
this module samples whole batches of ScenarioSpec objects from a single
GenerationConfig: numeric parameters are given as ranges, a master seed drives
all sampling, and every generated scenario carries its provenance in notes.
Fault combinations are expressed with the built-in event vocabulary — ``stop``
models a motor/e-stop fault, ``teleport`` a localisation jump, ``set_twist``
an erroneous command.

Three guarantees define the contract:

- Reproducible: generate_one(config, index) depends only on the config and the
  index, never on batch size or call order. All sampling draws come from one
  rng and only through random(), so the stream is stable across runs,
  platforms and interpreter versions.
- Comparable: scenarios embed (master_seed, index, config_digest) provenance,
  and diff_specs reports path-level differences between any two scenarios.
- Physically valid: samplers enforce Constraints (world bounds, speed limits,
  robot separation, event times inside the run) and every emitted spec is
  re-checked with validate_scenario before it leaves the generator.
"""
from __future__ import annotations

import hashlib
import json
import math
import random
from dataclasses import dataclass
from typing import Iterable

from aegisrover.sim.scenarios import ScenarioSpec, ScenarioStore
from aegisrover.storage.repository import canonical_json

__all__ = ('Constraints', 'GenerationConfig', 'GenerationError', 'Range',
           'ScenarioGenerator', 'diff_specs', 'generate_batch', 'provenance',
           'summarize', 'validate_scenario')

GENERATOR_ID = 'aegisrover.sim.generator'
KNOWN_EVENT_KINDS = ('set_twist', 'stop', 'teleport')
_MISSING = object()


class GenerationError(RuntimeError):
    pass


@dataclass(frozen=True)
class Range:
    """Inclusive numeric range, sampled uniformly. Count ranges must be integral."""
    low: float
    high: float

    def check(self, name: str) -> 'Range':
        if not self.low <= self.high:
            raise GenerationError(f'{name}: low {self.low} exceeds high {self.high}')
        return self


@dataclass(frozen=True)
class Constraints:
    """Physical limits every generated scenario must satisfy."""
    x_min: float = -10.0
    y_min: float = -10.0
    x_max: float = 10.0
    y_max: float = 10.0
    max_linear: float = 2.0      # m/s
    max_angular: float = 2.0     # rad/s
    min_separation: float = 0.5  # m between robot start poses / teleport targets
    max_duration: float = 600.0
    min_step: float = 0.01
    max_step: float = 1.0

    def check(self) -> 'Constraints':
        if not (self.x_min < self.x_max and self.y_min < self.y_max):
            raise GenerationError('bounds must be ordered')
        if self.max_linear <= 0 or self.max_angular <= 0:
            raise GenerationError('speed limits must be positive')
        if self.min_separation < 0:
            raise GenerationError('min_separation must be non-negative')
        if not (0 < self.min_step <= self.max_step):
            raise GenerationError('step limits must be positive and ordered')
        if self.max_duration <= 0:
            raise GenerationError('max_duration must be positive')
        return self


@dataclass(frozen=True)
class GenerationConfig:
    """Sampling parameters for one batch.

    count and scenario_prefix are excluded from the per-scenario seed chain, so
    generate_one(config, i) is identical no matter how large the batch is.
    """
    master_seed: int
    count: int = 8
    scenario_prefix: str = 'gen'
    step: float = 0.1
    duration: Range = Range(5.0, 30.0)
    robot_count: Range = Range(1, 3)
    events_per_robot: Range = Range(1, 4)
    event_kinds: tuple[str, ...] = KNOWN_EVENT_KINDS
    speed: Range = Range(0.2, 1.5)
    yaw_rate: Range = Range(-1.0, 1.0)
    constraints: Constraints = Constraints()


class ScenarioGenerator:
    """Samples reproducible scenario batches from a GenerationConfig."""

    def __init__(self, config: GenerationConfig):
        self._check_config(config)
        self._config = config
        self._n_low = max(1, math.ceil((config.duration.low - 1e-9) / config.step))
        self._n_high = math.floor((config.duration.high + 1e-9) / config.step)
        self._digest = _config_digest(config)

    @property
    def config_digest(self) -> str:
        return self._digest

    def generate(self) -> list[ScenarioSpec]:
        return [self.generate_one(i) for i in range(self._config.count)]

    def generate_one(self, index: int) -> ScenarioSpec:
        if index < 0:
            raise GenerationError('index must be non-negative')
        config = self._config
        seed = _derived_seed(config.master_seed, index)
        rng = random.Random(seed)
        n_steps = _randint(rng, self._n_low, self._n_high)
        robots = self._sample_robots(rng)
        events = self._sample_events(rng, robots, n_steps)
        events.sort(key=lambda e: e['time'])
        notes = json.dumps({'generator': GENERATOR_ID, 'version': 1,
                            'master_seed': config.master_seed, 'index': index,
                            'config_digest': self._digest},
                           sort_keys=True, separators=(',', ':'))
        spec = ScenarioSpec(
            scenario_id=f'{config.scenario_prefix}-{config.master_seed}-{index:04d}',
            revision=0, duration=round(n_steps * config.step, 9), step=config.step,
            seed=seed, robots=tuple(robots), events=tuple(events), notes=notes)
        violations = validate_scenario(spec, config.constraints)
        if violations:
            raise GenerationError(f'generated scenario violates constraints: {violations}')
        return spec

    def save_batch(self, store: ScenarioStore, *, actor: str = 'generator') -> list[ScenarioSpec]:
        return [store.save(spec, actor=actor) for spec in self.generate()]

    # -- sampling --------------------------------------------------------------
    def _sample_robots(self, rng: random.Random) -> list[dict]:
        config = self._config
        n = _randint(rng, int(config.robot_count.low), int(config.robot_count.high))
        robots = []
        for i, (x, y, yaw) in enumerate(_sample_poses(rng, n, config.constraints, ())):
            linear = _round(_uniform(rng, config.speed.low, config.speed.high))
            angular = _round(_uniform(rng, config.yaw_rate.low, config.yaw_rate.high))
            robots.append({'name': f'r{i + 1}', 'pose': [x, y, yaw], 'twist': [linear, angular]})
        return robots

    def _sample_events(self, rng: random.Random, robots: list[dict], n_steps: int) -> list[dict]:
        config = self._config
        events = []
        for robot in robots:
            k = _randint(rng, int(config.events_per_robot.low), int(config.events_per_robot.high))
            for slot in _sample_slots(rng, k, n_steps):
                kind = config.event_kinds[_randint(rng, 0, len(config.event_kinds) - 1)]
                events.append({'time': round(slot * config.step, 9), 'kind': kind,
                               'payload': self._sample_payload(rng, kind, robot, robots)})
        return events

    def _sample_payload(self, rng: random.Random, kind: str, robot: dict, robots: list[dict]) -> dict:
        config = self._config
        if kind == 'set_twist':
            return {'robot': robot['name'],
                    'linear': _round(_uniform(rng, config.speed.low, config.speed.high)),
                    'angular': _round(_uniform(rng, config.yaw_rate.low, config.yaw_rate.high))}
        if kind == 'stop':
            return {'robots': [robot['name']]}
        if kind == 'teleport':
            others = [r['pose'] for r in robots if r['name'] != robot['name']]
            x, y, yaw = _sample_poses(rng, 1, config.constraints, others)[0]
            return {'robot': robot['name'], 'pose': [x, y, yaw]}
        raise GenerationError(f'unknown event kind {kind!r}')

    # -- validation ------------------------------------------------------------
    @staticmethod
    def _check_config(config: GenerationConfig) -> None:
        c = config.constraints.check()
        if config.count < 1:
            raise GenerationError('count must be positive')
        if not (c.min_step <= config.step <= c.max_step):
            raise GenerationError('step outside constraint limits')
        duration = config.duration.check('duration')
        if duration.low < config.step:
            raise GenerationError('duration must be at least one step')
        if duration.high > c.max_duration:
            raise GenerationError('duration exceeds max_duration')
        n_low = max(1, math.ceil((duration.low - 1e-9) / config.step))
        n_high = math.floor((duration.high + 1e-9) / config.step)
        if n_low > n_high:
            raise GenerationError('duration range contains no whole number of steps')
        rc = config.robot_count.check('robot_count')
        if rc.low < 1 or not _is_integral(rc.low) or not _is_integral(rc.high):
            raise GenerationError('robot_count must be a positive integer range')
        ep = config.events_per_robot.check('events_per_robot')
        if ep.low < 0 or not _is_integral(ep.low) or not _is_integral(ep.high):
            raise GenerationError('events_per_robot must be a non-negative integer range')
        if ep.high > n_low:
            raise GenerationError('events_per_robot cannot fit the shortest duration')
        if ep.high > 0:
            if not config.event_kinds:
                raise GenerationError('event_kinds must not be empty')
            unknown = [k for k in config.event_kinds if k not in KNOWN_EVENT_KINDS]
            if unknown:
                raise GenerationError(f'unknown event kinds: {unknown}')
        speed = config.speed.check('speed')
        yaw_rate = config.yaw_rate.check('yaw_rate')
        if max(abs(speed.low), abs(speed.high)) > c.max_linear:
            raise GenerationError('speed range exceeds max_linear')
        if max(abs(yaw_rate.low), abs(yaw_rate.high)) > c.max_angular:
            raise GenerationError('yaw_rate range exceeds max_angular')


def generate_batch(config: GenerationConfig) -> list[ScenarioSpec]:
    return ScenarioGenerator(config).generate()


def validate_scenario(spec: ScenarioSpec, constraints: Constraints) -> list[str]:
    """Physical-constraint violations in a scenario; empty list means valid.

    Teleport targets are checked against other robots' *initial* poses — where
    a robot actually is at event time is a replay-time property, not something
    a static check can know.
    """
    c = constraints
    problems = []
    if not (c.min_step <= spec.step <= c.max_step):
        problems.append(f'step {spec.step} outside [{c.min_step}, {c.max_step}]')
    if not (spec.step <= spec.duration <= c.max_duration):
        problems.append(f'duration {spec.duration} outside [{spec.step}, {c.max_duration}]')
    names = [r.get('name') for r in spec.robots]
    if len(set(names)) != len(names):
        problems.append('robot names must be unique')
    known = set(names)
    positions = {}
    for robot in spec.robots:
        name = robot.get('name')
        pose = robot.get('pose') or [0.0, 0.0, 0.0]
        twist = robot.get('twist') or [0.0, 0.0]
        positions[name] = (float(pose[0]), float(pose[1]))
        if not (c.x_min <= positions[name][0] <= c.x_max and c.y_min <= positions[name][1] <= c.y_max):
            problems.append(f'{name}: initial pose outside bounds')
        if abs(float(twist[0])) > c.max_linear + 1e-9:
            problems.append(f'{name}: initial linear speed exceeds max_linear')
        if abs(float(twist[1])) > c.max_angular + 1e-9:
            problems.append(f'{name}: initial angular speed exceeds max_angular')
    ordered = sorted(positions)
    for i, a in enumerate(ordered):
        for b in ordered[i + 1:]:
            if _distance(positions[a], positions[b]) < c.min_separation - 1e-9:
                problems.append(f'{a} and {b}: initial poses closer than min_separation')
    for event in spec.events:
        kind = event['kind']
        payload = event.get('payload') or {}
        if not (0.0 <= event['time'] <= spec.duration + 1e-9):
            problems.append(f"event at t={event['time']} outside [0, {spec.duration}]")
        if kind not in KNOWN_EVENT_KINDS:
            problems.append(f'unknown event kind {kind!r}')
            continue
        if kind == 'set_twist':
            if payload.get('robot') not in known:
                problems.append(f"set_twist references unknown robot {payload.get('robot')!r}")
            if abs(float(payload.get('linear', 0.0))) > c.max_linear + 1e-9:
                problems.append('set_twist linear exceeds max_linear')
            if abs(float(payload.get('angular', 0.0))) > c.max_angular + 1e-9:
                problems.append('set_twist angular exceeds max_angular')
        elif kind == 'stop':
            for robot in payload.get('robots', ()):
                if robot not in known:
                    problems.append(f'stop references unknown robot {robot!r}')
        elif kind == 'teleport':
            robot = payload.get('robot')
            if robot not in known:
                problems.append(f'teleport references unknown robot {robot!r}')
                continue
            pose = payload.get('pose')
            if not isinstance(pose, (list, tuple)) or len(pose) != 3:
                problems.append('teleport pose must be [x, y, yaw]')
                continue
            target = (float(pose[0]), float(pose[1]))
            if not (c.x_min <= target[0] <= c.x_max and c.y_min <= target[1] <= c.y_max):
                problems.append('teleport target outside bounds')
            for other in sorted(positions):
                if other != robot and _distance(target, positions[other]) < c.min_separation - 1e-9:
                    problems.append(f'teleport target closer than min_separation to {other}')
    return problems


def provenance(spec: ScenarioSpec) -> dict:
    """Generator provenance embedded in a scenario's notes ({} if not generated)."""
    try:
        data = json.loads(spec.notes)
    except (TypeError, ValueError):
        return {}
    if not isinstance(data, dict) or data.get('generator') != GENERATOR_ID:
        return {}
    return data


def summarize(spec: ScenarioSpec) -> dict:
    """Compact, deterministic view of a scenario for batch inspection."""
    kinds: dict[str, int] = {}
    for event in spec.events:
        kinds[event['kind']] = kinds.get(event['kind'], 0) + 1
    return {
        'scenario_id': spec.scenario_id,
        'seed': spec.seed,
        'duration': round(spec.duration, 9),
        'step': round(spec.step, 9),
        'robots': {r.get('name'): {'pose': list(r.get('pose') or ()),
                                   'twist': list(r.get('twist') or ())} for r in spec.robots},
        'event_count': len(spec.events),
        'event_kinds': dict(sorted(kinds.items())),
        'provenance': provenance(spec),
    }


def diff_specs(left: ScenarioSpec, right: ScenarioSpec) -> dict:
    """Path-oriented diff of two scenarios' normalised payloads.

    Robots are matched by name and events by index, so changing one generation
    parameter shows up as a small set of paths instead of two opaque lists.
    """
    changed: dict = {}
    _walk(left.normalised(), right.normalised(), '', changed)
    return {'left': left.scenario_id, 'right': right.scenario_id, 'changed': changed}


# ---------------------------------------------------------------------------
# Sampling primitives. Everything funnels through rng.random() so the draw
# sequence is stable across interpreter versions (unlike randint/choices,
# whose implementations have changed between releases).
def _round(value: float) -> float:
    return round(value, 9)


def _uniform(rng: random.Random, low: float, high: float) -> float:
    return low + (high - low) * rng.random()


def _randint(rng: random.Random, low: int, high: int) -> int:
    return low + int(rng.random() * (high - low + 1))


def _is_integral(value: float) -> bool:
    return float(value).is_integer()


def _distance(a: tuple[float, float], b: tuple[float, float]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _derived_seed(master_seed: int, index: int) -> int:
    material = f'{GENERATOR_ID}:v1:{master_seed}:{index}'.encode()
    return int.from_bytes(hashlib.sha256(material).digest()[:8], 'big')


def _sample_poses(rng: random.Random, n: int, c: Constraints,
                  fixed: Iterable) -> list[tuple[float, float, float]]:
    """n poses inside bounds, each at least min_separation from fixed and each other."""
    placed = [(float(p[0]), float(p[1])) for p in fixed]
    out = []
    for _ in range(n):
        for _attempt in range(1000):
            x = _round(_uniform(rng, c.x_min, c.x_max))
            y = _round(_uniform(rng, c.y_min, c.y_max))
            if all(_distance((x, y), p) >= c.min_separation for p in placed):
                break
        else:
            raise GenerationError(
                f'cannot place {n} poses with {c.min_separation} m separation in the given bounds')
        yaw = _round(_uniform(rng, -math.pi, math.pi))
        placed.append((x, y))
        out.append((x, y, yaw))
    return out


def _sample_slots(rng: random.Random, k: int, n_steps: int) -> list[int]:
    """k distinct step indices in 1..n_steps, drawn without replacement."""
    if k > n_steps:
        raise GenerationError(f'cannot place {k} events in {n_steps} steps')
    slots = list(range(1, n_steps + 1))
    for i in range(k):  # partial Fisher-Yates
        j = i + _randint(rng, 0, n_steps - 1 - i)
        slots[i], slots[j] = slots[j], slots[i]
    return sorted(slots[:k])


def _config_digest(config: GenerationConfig) -> str:
    c = config.constraints
    body = canonical_json({
        'version': 1,
        'master_seed': config.master_seed,
        'scenario_prefix': config.scenario_prefix,
        'step': round(config.step, 9),
        'duration': [config.duration.low, config.duration.high],
        'robot_count': [config.robot_count.low, config.robot_count.high],
        'events_per_robot': [config.events_per_robot.low, config.events_per_robot.high],
        'event_kinds': sorted(config.event_kinds),
        'speed': [config.speed.low, config.speed.high],
        'yaw_rate': [config.yaw_rate.low, config.yaw_rate.high],
        'constraints': {'x_min': c.x_min, 'y_min': c.y_min, 'x_max': c.x_max, 'y_max': c.y_max,
                        'max_linear': c.max_linear, 'max_angular': c.max_angular,
                        'min_separation': c.min_separation, 'max_duration': c.max_duration,
                        'min_step': c.min_step, 'max_step': c.max_step},
    })
    return hashlib.sha256(body.encode()).hexdigest()


def _walk(a, b, path: str, out: dict) -> None:
    if isinstance(a, dict) and isinstance(b, dict):
        for key in sorted(set(a) | set(b)):
            _walk(a.get(key, _MISSING), b.get(key, _MISSING), f'{path}.{key}' if path else key, out)
    elif isinstance(a, list) and isinstance(b, list):
        if _named(a) and _named(b):
            left_by = {item['name']: item for item in a}
            right_by = {item['name']: item for item in b}
            for key in sorted(set(left_by) | set(right_by)):
                _walk(left_by.get(key, _MISSING), right_by.get(key, _MISSING), f'{path}[{key}]', out)
        else:
            for i in range(max(len(a), len(b))):
                _walk(a[i] if i < len(a) else _MISSING,
                      b[i] if i < len(b) else _MISSING, f'{path}[{i}]', out)
    elif a is _MISSING or b is _MISSING:
        out[path] = {'left': None if a is _MISSING else a,
                     'right': None if b is _MISSING else b}
    elif a != b:
        out[path] = {'left': a, 'right': b}


def _named(items) -> bool:
    return all(isinstance(item, dict) and 'name' in item for item in items)
