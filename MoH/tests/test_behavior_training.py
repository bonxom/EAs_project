import json
import pickle
from pathlib import Path
from unittest.mock import Mock, patch

import pytest
from deployment_fakes import FakeLLM
from moh import MoH
from omegaconf import OmegaConf
from utils.behavior import code_digest
from utils.run_logger import RunLogger
from utils.utils import clean_code

TASK = 'tsp_gls-100'
CODES = [f'def heuristic_function():\n    return {i}' for i in range(5)]


@pytest.fixture
def training(tmp_path):
    root = tmp_path / 'root'
    data = root / 'problems/tsp_gls/data/val'
    data.mkdir(parents=True)
    with (data / 'val_instance_data_sol_100.pkl').open('wb') as output:
        pickle.dump({key: list(range(8)) for key in
                     ('cost', 'distance_matrix', 'coordinate', 'optimal_tour')}, output)
    cfg = OmegaConf.create({'n_iterations': 1, 'pop_size': 3, 'seed': 0,
        'timeout': 2, 'optimizer_timeout': 10, 'max_eval_calls': 100,
        'population': {'diversity_fraction': .34},
        'problem': {'problem_name': 'tsp_gls', 'problem_size': [100], 'threshold': {100: 9}}})
    with patch.object(MoH, 'init_prompts'), patch.object(MoH, 'init_meta_improver'), \
         patch('moh.RunLogger', return_value=RunLogger(str(tmp_path / 'run'))):
        obj = MoH(cfg, str(root), FakeLLM())
    obj.subtask_form = 'format'
    obj._cur_iter = 0
    obj.vectors = {clean_code(code): vector * 4 for code, vector in
                   zip(CODES, [(0, 2), (0, 2.2), (0, 2.4), (3, 0), (4, 0)], strict=True)}

    def run(code, problem_type, mode, stdout_path, *, result_path):
        identity = obj.subtask_pop.identities[problem_type]
        gaps = obj.vectors[clean_code(code)]
        payload = {'size': identity.size, 'mode': mode, 'dataset_sha256': identity.dataset_sha256,
            'code_sha256': code_digest(code), 'indices': identity.indices, 'seed': identity.seed,
            'solver_settings': dict(identity.solver_settings), 'count': len(gaps),
            'status': 'ok', 'failures': 0, 'mean_gap_pct': sum(gaps) / len(gaps),
            'instances': [{'gap_pct': gap, 'status': 'ok'} for gap in gaps]}
        Path(result_path).write_text(json.dumps(payload))
        return Mock(returncode=0)

    obj._run_code = Mock(side_effect=run)
    return obj


def test_training_admits_all_evaluated_and_verifies_return(training):
    program = f'''def improve_algorithm(population, utility, language_model, function_format, task):
    for code in {CODES[:4]!r}:
        utility(code, 'candidate', task)
    return 'forged', {CODES[0]!r}, -999
'''
    result = training.get_improver(program, 'format', TASK)
    assert not isinstance(result, Exception)
    assert result[2] == 1
    assert training.subtask_pop.check_solution(TASK, clean_code(CODES[3]))
    assert training.subtask_pop.get_subtask_size(TASK) == 3
    assert training._run_code.call_count == 4


def test_unseen_return_evaluated_and_failures_do_not_erase_survivors(training):
    training.evaluate_heuristic(CODES[0], TASK)
    program = f'''def improve_algorithm(*args):
    return 'new', {CODES[3]!r}, -999
'''
    assert training.get_improver(program, '', TASK)[2] == 1.5
    training._run_code.side_effect = OSError('failed launch')
    assert training.evaluate_heuristic(CODES[4], TASK) == 1e6
    assert training.subtask_pop.get_subtask_size(TASK) == 2


def test_duplicate_and_wrong_context(training):
    training.evaluate_heuristic(CODES[0], TASK)
    training.evaluate_heuristic('# comment\n' + CODES[0], TASK)
    assert training.subtask_pop.get_subtask_size(TASK) == 1
    for task, mode in [('tsp_gls-200', 'val'), (TASK, 'test')]:
        before = training._run_code.call_count
        with pytest.raises(ValueError):
            training.evaluate_heuristic(CODES[0], task, mode)
        assert training._run_code.call_count == before


def test_legacy_population_is_remeasured_and_reused(training):
    path = Path(training.root_dir) / f'problems/tsp_gls/solution_{TASK}.json'
    path.write_text(json.dumps([{'idea': 'old', 'best_sol': code, 'utility': -999}
                                for code in CODES[:3]]))
    training.read_base_algorithm(TASK)
    assert training._run_code.call_count == 3
    assert training.subtask_pop.get_best_solution(TASK)['utility'] == 1
    training.read_base_algorithm(TASK)
    assert training._run_code.call_count == 3


def test_matching_snapshot_loads_without_evaluation(training):
    for code in CODES[:3]:
        training.evaluate_heuristic(code, TASK)
    path = Path(training.root_dir) / f'problems/tsp_gls/solution_{TASK}.json'
    training.subtask_pop.save_subtask_to_file(TASK, path)
    training.subtask_pop._replace_population(TASK, [])
    training._run_code.reset_mock()
    training.read_base_algorithm(TASK)
    assert training._run_code.call_count == 0
    assert training._ensure_measured_candidate(TASK, CODES[0], '') is not None
