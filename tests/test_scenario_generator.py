"""Tests for seed-driven batch scenario generation."""
import json

import pytest

from aegisrover.sim.generator import (
    DriveEnvelope, GeneratedScenario, GeneratorConfig, GeneratorError,
    ROBOT_RADIUS, batch_coverage, diff_generated, generate_batch, save_batch,
    validate_spec,
)
from aegisrover.sim.scenarios import ScenarioSpec, ScenarioStore
from aegisrover.storage.audit import AuditLog
from aegisrover.storage.repository import Repository


class Clock:
    def __init__(self, now=1.0):
        self.now = now

    def __call__(self):
        self.now += 0.25
        return self.now


@pytest.fixture()
def repo():
    repository = Repository(':memory:', clock=Clock())
    yield repository
    repository.close()


def make_config(**overrides):
    base = dict(seed=42, count=12, duration_range=(6.0, 12.0), step=0.1,
                arena_range=(5.0, 8.0), robots_range=(1, 3), faults_range=(0, 5),
                min_event_gap=0.5)
    base.update(overrides)
    return GeneratorConfig(**base)


# --------------------------------------------------------------- reproducibility
def test_same_seed_produces_identical_batch():
    first = generate_batch(make_config())
    second = generate_batch(make_config(seed=42))
    assert first.manifest_json() == second.manifest_json()
    assert [s.scenario_id for s in first.specs()] == [s.scenario_id for s in second.specs()]
    assert [s.digest for s in first.specs()] == [s.digest for s in second.specs()]


def test_different_seeds_diverge_but_each_scenario_is_self_deterministic():
    a = generate_batch(make_config(seed=1))
    b = generate_batch(make_config(seed=2))
    assert a.manifest_json() != b.manifest_json()
    # Child seeds derive from (master, index), so index 0 is reproducible per request.
    assert a.scenarios[0].params.child_seed != b.scenarios[0].params.child_seed
    again = generate_batch(make_config(seed=1))
    assert a.scenarios[0].spec.digest == again.scenarios[0].spec.digest


def test_generated_scenarios_replay_deterministically(repo):
    store = ScenarioStore(repo, audit=AuditLog(repo, Clock()), clock=Clock())
    batch = generate_batch(make_config())
    saved = save_batch(store, batch, actor='generator')
    for spec in saved:
        assert store.verify(spec)
        result = store.require_reproducible(spec)
        assert result.steps == round(spec.duration / spec.step)
        assert result.seed == spec.seed


def test_manifest_is_stable_json_with_params_and_signature():
    batch = generate_batch(make_config())
    payload = json.loads(batch.manifest_json())
    assert payload['master_seed'] == 42 and len(payload['scenarios']) == 12
    entry = payload['scenarios'][3]
    assert set(entry) == {'scenario_id', 'digest', 'params', 'fault_signature'}
    assert entry['params']['index'] == 3
    assert entry['params']['master_seed'] == 42
    assert batch.manifest_json() == generate_batch(make_config()).manifest_json()


# --------------------------------------------------------------- ranges
def test_parameters_stay_within_configured_ranges():
    config = make_config()
    batch = generate_batch(config)
    for scn in batch.scenarios:
        p = scn.params
        assert 6.0 <= p.duration <= 12.0
        assert 5.0 <= p.arena <= 8.0
        assert 1 <= p.robot_count <= 3
        assert 0 <= p.fault_count <= 5
        assert len(scn.spec.robots) == p.robot_count
        assert len(scn.spec.events) == p.fault_count
        # times land on the step grid and inside the event window
        for time, *_ in scn.fault_signature:
            assert abs(time - round(time / 0.1) * 0.1) < 1e-9
            assert 0.6 <= time <= p.duration * 0.9 + 1e-9


def test_batch_coverage_reports_combinations():
    batch = generate_batch(make_config(seed=7, count=40, robots_range=(2, 4),
                                       faults_range=(1, 3)))
    coverage = batch_coverage(batch)
    assert coverage['count'] == 40
    robots = {c['robots'] for c in coverage['combinations']}
    faults = {c['faults'] for c in coverage['combinations']}
    assert robots <= {2, 3, 4} and faults <= {1, 2, 3}
    assert sum(c['scenarios'] for c in coverage['combinations']) == 40
    assert coverage['distinct_fault_signatures'] > 1


def test_zero_faults_yields_clean_scenarios():
    batch = generate_batch(make_config(faults_range=(0, 0)))
    assert all(scn.spec.events == () for scn in batch.scenarios)
    assert batch_coverage(batch)['distinct_fault_signatures'] == 1


# --------------------------------------------------------------- physical feasibility
def test_all_twists_respect_drive_envelope_and_min_turn_radius():
    envelope = DriveEnvelope(max_linear=1.2, max_angular=0.8, min_turn_radius=0.5)
    batch = generate_batch(make_config(envelope=envelope, faults_range=(4, 5), count=20))
    for scn in batch.scenarios:
        for event in scn.spec.events:
            if event['kind'] == 'set_twist':
                v, w = event['payload']['linear'], event['payload']['angular']
                assert abs(v) <= 1.2 + 1e-9 and abs(w) <= 0.8 + 1e-9
                if abs(w) > 1e-9:
                    assert abs(v) / abs(w) >= 0.5 - 1e-9


def test_robots_never_overlap_and_teleports_keep_clear_and_inside():
    batch = generate_batch(make_config(arena_range=(4.0, 4.0), robots_range=(3, 3),
                                       faults_range=(5, 5), count=15))
    for scn in batch.scenarios:
        arena = scn.params.arena
        spawn = [(r['pose'][0], r['pose'][1]) for r in scn.spec.robots]
        for i, (x1, y1) in enumerate(spawn):
            for x2, y2 in spawn[i + 1:]:
                assert (x1 - x2) ** 2 + (y1 - y2) ** 2 >= (2 * ROBOT_RADIUS) ** 2 - 1e-9
        for event in scn.spec.events:
            if event['kind'] == 'teleport':
                x, y, _ = event['payload']['pose']
                assert -ROBOT_RADIUS <= x <= arena + ROBOT_RADIUS
                assert -ROBOT_RADIUS <= y <= arena + ROBOT_RADIUS
                target = event['payload']['robot']
                for robot in scn.spec.robots:
                    if robot['name'] == target:
                        continue
                    ox, oy = robot['pose'][0], robot['pose'][1]
                    assert (x - ox) ** 2 + (y - oy) ** 2 >= (2 * ROBOT_RADIUS) ** 2 - 1e-9


def test_events_touching_one_robot_respect_minimum_gap():
    config = make_config(faults_range=(5, 5), duration_range=(6.0, 6.0), count=10)
    for scn in generate_batch(config).scenarios:
        per_robot: dict[str, list] = {}
        for event in scn.spec.events:
            if event['kind'] == 'stop':
                names = event['payload'].get('robots') or [r['name'] for r in scn.spec.robots]
            else:
                names = [event['payload']['robot']]
            for name in names:
                per_robot.setdefault(name, []).append(event['time'])
        for times in per_robot.values():
            for a, b in zip(sorted(times), sorted(times)[1:]):
                assert b - a >= config.min_event_gap - 1e-9


def test_infeasible_requests_are_rejected_upfront():
    # arena too small to fit that many robots
    with pytest.raises(GeneratorError, match='cannot fit'):
        generate_batch(make_config(arena_range=(1.0, 1.0), robots_range=(4, 4)))
    # timeline too short for the fault load at the configured gap
    with pytest.raises(GeneratorError, match='faults cannot fit'):
        generate_batch(make_config(duration_range=(2.0, 2.0), faults_range=(6, 6),
                                   min_event_gap=1.0))
    # envelope whose minimum turn radius is unreachable
    with pytest.raises(GeneratorError, match='unreachable'):
        DriveEnvelope(max_linear=0.1, max_angular=2.0, min_turn_radius=1.0).validate()
    # bad range shape
    with pytest.raises(GeneratorError):
        generate_batch(make_config(duration_range=(9.0, 1.0)))
    with pytest.raises(GeneratorError):
        generate_batch(make_config(count=0))


def test_validate_spec_rejects_impossible_handwritten_specs():
    good = ScenarioSpec(
        scenario_id='hand', revision=0, duration=2.0, step=0.1, seed=1,
        robots=({'name': 'r1', 'pose': [0.0, 0.0, 0.0], 'twist': [0.0, 0.0]},),
        events=({'time': 1.0, 'kind': 'set_twist',
                 'payload': {'robot': 'r1', 'linear': 1.0, 'angular': 0.0}},),
    )
    validate_spec(good, DriveEnvelope(), arena=4.0)  # does not raise

    too_fast = ScenarioSpec(
        scenario_id='hand', revision=0, duration=2.0, step=0.1, seed=1,
        robots=(), events=({'time': 1.0, 'kind': 'set_twist',
                            'payload': {'robot': 'r1', 'linear': 9.0, 'angular': 0.0}},),
    )
    with pytest.raises(GeneratorError, match='unknown robot'):
        validate_spec(too_fast)

    overlap = ScenarioSpec(
        scenario_id='hand', revision=0, duration=2.0, step=0.1, seed=1,
        robots=({'name': 'r1', 'pose': [0.0, 0.0, 0.0], 'twist': [0.0, 0.0]},
                {'name': 'r2', 'pose': [0.1, 0.0, 0.0], 'twist': [0.0, 0.0]}),
        events=(),
    )
    with pytest.raises(GeneratorError, match='overlap'):
        validate_spec(overlap)

    outside = ScenarioSpec(
        scenario_id='hand', revision=0, duration=2.0, step=0.1, seed=1,
        robots=({'name': 'r1', 'pose': [50.0, 0.0, 0.0], 'twist': [0.0, 0.0]},),
        events=(),
    )
    with pytest.raises(GeneratorError, match='outside the arena'):
        validate_spec(outside, arena=4.0)


# --------------------------------------------------------------- comparison
def test_diff_reports_seed_difference_and_identical_structure():
    a = generate_batch(make_config(seed=100, count=3))
    b = generate_batch(make_config(seed=100, count=3))
    same = diff_generated(a.scenarios[0], b.scenarios[0])
    assert same['changed_params'] == {}
    assert same['events_added'] == [] and same['events_removed'] == []
    assert same['digests']['equal']

    other = generate_batch(make_config(seed=101, count=3))
    delta = diff_generated(a.scenarios[0], other.scenarios[0])
    assert delta['changed_params']['seed']
    assert delta['left'].endswith('_0000') and delta['right'].endswith('_0000')


def test_diff_detects_payload_change_between_shared_events():
    batch = generate_batch(make_config(seed=42))
    scn = batch.scenarios[0]
    tweaked_events = list(scn.spec.events)
    if tweaked_events:
        first = dict(tweaked_events[0])
        first['payload'] = dict(first['payload'], linear=0.42)
        tweaked_events[0] = first
    else:
        tweaked_events = [{'time': 1.0, 'kind': 'set_twist',
                           'payload': {'robot': 'r1', 'linear': 0.42, 'angular': 0.0}}]
    rebuilt = ScenarioSpec(**{**scn.spec.to_dict(), 'digest': '',
                              'events': tuple(tweaked_events)})
    rebuilt = ScenarioSpec.from_dict({**rebuilt.to_dict(),
                                      'digest': scn.spec.digest})
    variant = GeneratedScenario(spec=rebuilt, params=scn.params,
                                fault_signature=scn.fault_signature)
    report = diff_generated(scn, variant)
    assert report['digests']['equal']  # digest field copied; structural diff still flags it
    shared = scn.spec.events and tweaked_events and \
        scn.spec.events[0]['kind'] == 'set_twist'
    if shared and scn.spec.events[0]['payload'].get('linear') != 0.42:
        assert report['payload_changes']


def test_diff_lists_added_and_removed_faults():
    a = generate_batch(make_config(seed=5, count=1)).scenarios[0]
    fewer_events = a.spec.events[:-1] if a.spec.events else ()
    rebuilt = ScenarioSpec(**{**a.spec.to_dict(), 'digest': '',
                              'events': tuple(fewer_events)})
    variant = GeneratedScenario(spec=rebuilt, params=a.params,
                                fault_signature=a.fault_signature[:-1])
    report = diff_generated(a, variant)
    assert len(report['events_removed']) == (1 if a.spec.events else 0)
