import copy
import json
from dataclasses import replace

import pytest
from utils.behavior import (
    BehaviorDescriptor,
    EvaluationIdentity,
    behavior_distance,
    descriptor_from_dict,
    descriptor_to_dict,
    parse_behavior,
)


@pytest.fixture
def identity():
    return EvaluationIdentity(100, 'val', 'a' * 64, (2, 7), 0, (('workers', 2),))


@pytest.fixture
def payload(identity):
    return {'size': 100, 'mode': 'val', 'dataset_sha256': 'a' * 64,
                'code_sha256': 'b' * 64, 'indices': [2, 7], 'seed': 0,
                'solver_settings': {'workers': 2}, 'count': 2, 'status': 'ok', 'failures': 0,
                'mean_gap_pct': 2.0,
                'instances': [{'gap_pct': 1, 'status': 'ok'}, {'gap_pct': 3, 'status': 'ok'}]}


def test_roundtrip_order_and_distance(identity, payload):
    result = parse_behavior(payload, expected=identity, code_sha256='b' * 64)
    assert result.gaps == (1, 3)
    assert descriptor_from_dict(json.loads(json.dumps(descriptor_to_dict(result)))) == result
    other = BehaviorDescriptor(identity, (3, 1), 2)
    assert behavior_distance(result, result) == 0
    assert behavior_distance(result, other) == behavior_distance(other, result) > 0
    with pytest.raises(ValueError):
        behavior_distance(result, replace(other, identity=replace(identity, seed=1)))


@pytest.mark.parametrize(('field', 'value'), [
    ('instances', []), ('instances', [{'gap_pct': 2, 'status': 'ok'}]),
    ('failures', 1), ('indices', [7, 2]), ('indices', [2, 2]),
    ('dataset_sha256', 'c' * 64), ('code_sha256', 'c' * 64),
    ('seed', 1), ('size', 200), ('mode', 'test'),
    ('solver_settings', {'workers': 1}), ('count', True),
    ('mean_gap_pct', 3), ('mean_gap_pct', float('nan')), ('status', 'failed'),
])
def test_reject_payload_mismatch(identity, payload, field, value):
    payload[field] = value
    with pytest.raises(ValueError):
        parse_behavior(payload, expected=identity, code_sha256='b' * 64)


@pytest.mark.parametrize('value', [float('nan'), float('inf'), -1, True, '1'])
def test_reject_invalid_gaps(identity, payload, value):
    payload['instances'][0]['gap_pct'] = value
    with pytest.raises(ValueError):
        parse_behavior(payload, expected=identity, code_sha256='b' * 64)


def test_missing_or_failed_instance(identity, payload):
    broken = copy.deepcopy(payload)
    del broken['instances']
    with pytest.raises(ValueError):
        parse_behavior(broken, expected=identity, code_sha256='b' * 64)
    payload['instances'][0]['status'] = 'failed'
    with pytest.raises(ValueError):
        parse_behavior(payload, expected=identity, code_sha256='b' * 64)


@pytest.mark.parametrize('changes', [{'indices': ()}, {'indices': (2, 2)},
    {'indices': (True, 2)}, {'size': 0}, {'seed': True}, {'dataset_sha256': 'bad'},
    {'solver_settings': (('workers', float('nan')),)}])
def test_identity_validation(identity, changes):
    with pytest.raises(ValueError):
        replace(identity, **changes)
