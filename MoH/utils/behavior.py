"""Validated per-instance evaluation behavior shared by training and deployment."""

from __future__ import annotations

import hashlib
import math
from dataclasses import asdict, dataclass


def _number(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError('Expected a finite number')
    return value


def _integer(value, minimum=0):
    if type(value) is not int or value < minimum:
        raise ValueError('Expected an integer in range')
    return value


def _hash(value):
    if not isinstance(value, str) or len(value) != 64 or any(c not in '0123456789abcdef' for c in value):
        raise ValueError('Expected SHA-256 hex digest')
    return value


def code_digest(code: str) -> str:
    return hashlib.sha256(code.encode()).hexdigest()


@dataclass(frozen=True)
class EvaluationIdentity:
    size: int
    mode: str
    dataset_sha256: str
    indices: tuple[int, ...]
    seed: int
    solver_settings: tuple[tuple[str, int | float], ...]

    def __post_init__(self):
        _integer(self.size, 1)
        _integer(self.seed)
        _hash(self.dataset_sha256)
        if self.mode not in {'val', 'test'}:
            raise ValueError('Unsupported evaluation mode')
        indices = tuple(self.indices)
        if not indices or len(set(indices)) != len(indices):
            raise ValueError('Expected nonempty unique indices')
        for index in indices:
            _integer(index)
        settings = tuple(sorted(tuple(pair) for pair in self.solver_settings))
        if not settings or len({key for key, _ in settings}) != len(settings):
            raise ValueError('Expected unique solver settings')
        for key, value in settings:
            if not isinstance(key, str) or not key:
                raise ValueError('Invalid solver setting name')
            _number(value)
        object.__setattr__(self, 'indices', indices)
        object.__setattr__(self, 'solver_settings', settings)


@dataclass(frozen=True)
class BehaviorDescriptor:
    identity: EvaluationIdentity
    gaps: tuple[float, ...]
    mean_gap_pct: float

    def __post_init__(self):
        if not isinstance(self.identity, EvaluationIdentity):
            raise ValueError('Missing evaluation identity')  # noqa: TRY004 -- validation boundary
        gaps = tuple(self.gaps)
        if len(gaps) != len(self.identity.indices):
            raise ValueError('Gap vector length mismatch')
        if any(_number(gap) < 0 for gap in gaps) or _number(self.mean_gap_pct) < 0:
            raise ValueError('Gap must be nonnegative')
        if not math.isclose(math.fsum(gaps) / len(gaps), self.mean_gap_pct,
                            rel_tol=1e-9, abs_tol=1e-9):
            raise ValueError('Mean gap does not match instances')
        object.__setattr__(self, 'gaps', gaps)


def parse_behavior(payload: dict, *, expected: EvaluationIdentity,
                   code_sha256: str) -> BehaviorDescriptor:
    """Validate a fully successful result against parent-owned request identity."""
    try:
        _hash(code_sha256)
        if payload['status'] != 'ok' or _integer(payload['failures']) != 0:
            raise ValueError('Evaluation contains failures')
        if _integer(payload['count'], 1) != len(expected.indices):
            raise ValueError('Evaluation count mismatch')
        actual = EvaluationIdentity(
            payload['size'], payload['mode'], payload['dataset_sha256'],
            tuple(payload['indices']), payload['seed'],
            tuple(payload['solver_settings'].items()),
        )
        if actual != expected or payload['code_sha256'] != code_sha256:
            raise ValueError('Evaluation identity or code mismatch')
        instances = payload['instances']
        if any(item['status'] != 'ok' for item in instances):
            raise ValueError('Evaluation contains failed instances')
        return BehaviorDescriptor(actual, tuple(item['gap_pct'] for item in instances),
                                  payload['mean_gap_pct'])
    except (KeyError, TypeError, AttributeError) as error:
        raise ValueError('Malformed behavior result') from error


def behavior_distance(a: BehaviorDescriptor, b: BehaviorDescriptor) -> float:
    if a.identity != b.identity:
        raise ValueError('Cannot compare different evaluation identities')
    return math.sqrt(math.fsum(
        (math.log1p(x) - math.log1p(y)) ** 2
        for x, y in zip(a.gaps, b.gaps, strict=True)
    ) / len(a.gaps))


def descriptor_to_dict(value: BehaviorDescriptor) -> dict:
    return asdict(value)


def descriptor_from_dict(value: dict) -> BehaviorDescriptor:
    try:
        return BehaviorDescriptor(EvaluationIdentity(**value['identity']),
                                  tuple(value['gaps']), value['mean_gap_pct'])
    except (KeyError, TypeError, AttributeError) as error:
        raise ValueError('Malformed behavior descriptor') from error
