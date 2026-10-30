import copy
import json
from dataclasses import replace

import numpy as np
import pytest
from utils.behavior import BehaviorDescriptor, EvaluationIdentity, code_digest
from utils.behavior_population import BehaviorPopulation
from utils.utils import clean_code

TASK = 'tsp_gls-100'
IDENTITY = EvaluationIdentity(100, 'val', 'a' * 64, (2, 7), 0, (('workers', 2),))
CODES = [f'def heuristic():\n    return {i}' for i in range(5)]


def add(pop, index, gaps):
    return pop.admit(TASK, str(index), CODES[index],
                     BehaviorDescriptor(IDENTITY, gaps, sum(gaps) / len(gaps)))


def test_complement_survives_and_best_remains():
    pop = BehaviorPopulation({TASK: IDENTITY}, 3, .34)
    for i, vector in enumerate([(0, 2), (0, 2.2), (0, 2.4), (3, 0)]):
        add(pop, i, vector)
    assert {e['best_sol'] for e in pop.get_population(TASK)} == {
        clean_code(CODES[i]) for i in (0, 1, 3)}
    assert pop.get_best_solution(TASK)['utility'] == 1
    assert pop.diagnostics(TASK)['mean_pairwise_distance'] > 0


def test_equal_mean_duplicate_and_quality_ablation():
    pop = BehaviorPopulation({TASK: IDENTITY}, 2, .5)
    add(pop, 0, (0, 2))
    add(pop, 1, (2, 0))
    assert pop.get_subtask_size(TASK) == 2
    assert not pop.admit(TASK, 'duplicate', '# comment\n' + CODES[0],
                         BehaviorDescriptor(IDENTITY, (0, 2), 1))
    quality = BehaviorPopulation({TASK: IDENTITY}, 2, 0)
    for i, gaps in enumerate([(0, 2), (0, 2.2), (3, 0)]):
        add(quality, i, gaps)
    assert [e['utility'] for e in quality.get_population(TASK)] == [1, 1.1]


@pytest.mark.parametrize('size,fraction', [(0, .3), (True, .3), (1, -1), (1, 1),
    (1, float('nan')), (1, float('inf')), (1, True)])
def test_invalid_policy(size, fraction):
    with pytest.raises(ValueError):
        BehaviorPopulation({TASK: IDENTITY}, size, fraction)


def test_one_rounding_ties_and_repeatability():
    for size, fraction in [(1, .9), (10, .3)]:
        populations = [BehaviorPopulation({TASK: IDENTITY}, size, fraction) for _ in range(2)]
        for pop in populations:
            for i in range(5):
                add(pop, i, (1, 1))
        assert populations[0].get_population(TASK) == populations[1].get_population(TASK)
        assert populations[0].get_best_solution(TASK)['code_sha256'] == min(
            code_digest(clean_code(code)) for code in CODES)
        assert populations[0].elite_slots == (1 if size == 1 else 7)


def test_identity_and_unmeasured_insertion():
    pop = BehaviorPopulation({TASK: IDENTITY}, 2)
    for identity in [replace(IDENTITY, seed=1), replace(IDENTITY, mode='test')]:
        with pytest.raises(ValueError):
            pop.admit(TASK, '', CODES[0], BehaviorDescriptor(identity, (1, 1), 1))
    with pytest.raises(ValueError):
        pop.save_solution(TASK, '', CODES[0], -100)


def test_roundtrip_and_atomic_load(tmp_path):
    pop = BehaviorPopulation({TASK: IDENTITY}, 3, .34)
    for i, gaps in enumerate([(0, 2), (0, 2.2), (3, 0)]):
        add(pop, i, gaps)
    for all_tasks in [False, True]:
        path = tmp_path / 'population.json'
        restored = BehaviorPopulation({TASK: IDENTITY}, 3, .34)
        if all_tasks:
            pop.save_all_data_to_file(path)
            restored.load_all_data_from_file(path)
        else:
            pop.save_subtask_to_file(TASK, path)
            restored.load_subtask_from_file(TASK, path)
        assert restored.get_population(TASK) == pop.get_population(TASK)
        for obj in [restored, copy.deepcopy(restored)]:
            np.random.seed(7)
            selected = obj.get_random_solution(TASK)
            np.random.seed(7)
            assert selected == pop.get_random_solution(TASK)
        before = copy.deepcopy(restored.get_population(TASK))
        path.write_text(json.dumps([{'best_sol': CODES[0], 'utility': 0}]))
        with pytest.raises(ValueError):
            restored.load_subtask_from_file(TASK, path)
        assert restored.get_population(TASK) == before
