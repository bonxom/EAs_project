from pathlib import Path
from unittest.mock import Mock, patch
from omegaconf import OmegaConf

from deployment_fakes import HEURISTIC, FakeLLM
from moh import MoH

ROOT_DIR = str(Path(__file__).resolve().parent.parent)


def test_rmoh_flag_disabled_by_default(tmp_path):
    cfg = OmegaConf.create({
        "problem": {
            "problem_name": "tsp_gls",
            "problem_size": [20],
            "threshold": {20: 10.0},
        },
        "n_iterations": 2,
        "pop_size": 3,
        "timeout": 10,
        "seed": 0,
        "rmoh": {
            "enabled": False,
        },
    })
    llm = FakeLLM()
    moh = MoH(cfg, ROOT_DIR, heu_llm=llm, meta_llm=llm)

    assert moh.rmoh_enabled is False
    assert moh.rmoh_pop is None


def test_rmoh_flag_enabled_initializes_meta_population(tmp_path):
    cfg = OmegaConf.create({
        "problem": {
            "problem_name": "tsp_gls",
            "problem_size": [20],
            "threshold": {20: 10.0},
        },
        "n_iterations": 2,
        "pop_size": 3,
        "timeout": 10,
        "seed": 0,
        "rmoh": {
            "enabled": True,
            "alpha": 0.7,
            "c": 0.2,
            "reflection": True,
            "summary_max_words": 50,
        },
    })
    llm = FakeLLM()
    moh = MoH(cfg, ROOT_DIR, heu_llm=llm, meta_llm=llm)

    assert moh.rmoh_enabled is True
    assert moh.rmoh_pop is not None
    assert moh.rmoh_pop.alpha == 0.7
    assert moh.rmoh_pop.c == 0.2


def test_rmoh_trajectory_collection_in_inner_utility(tmp_path):
    cfg = OmegaConf.create({
        "problem": {
            "problem_name": "tsp_gls",
            "problem_size": [20],
            "threshold": {20: 10.0},
        },
        "n_iterations": 2,
        "pop_size": 3,
        "timeout": 10,
        "seed": 0,
        "rmoh": {
            "enabled": True,
            "reflection": False,
        },
    })
    llm = FakeLLM()
    moh = MoH(cfg, ROOT_DIR, heu_llm=llm, meta_llm=llm)

    with patch.object(moh, "evaluate_heuristic", return_value=8.5):
        collector = None
        if moh.rmoh_enabled:
            from utils.diagnostics import TrajectoryCollector
            collector = TrajectoryCollector()

        util_fn = moh._make_subtask_utility("tsp_gls-20", collector=collector)
        score = util_fn(HEURISTIC, "test_idea", "tsp_gls-20")

        assert score == 8.5
        assert collector is not None
        assert len(collector.records) == 1
        assert collector.records[0]["score"] == 8.5
        assert collector.records[0]["is_valid"] is True


def test_rmoh_try_improvement_flow(tmp_path):
    cfg = OmegaConf.create({
        "problem": {
            "problem_name": "tsp_gls",
            "problem_size": [20],
            "threshold": {20: 10.0},
        },
        "n_iterations": 2,
        "pop_size": 3,
        "timeout": 10,
        "seed": 0,
        "rmoh": {
            "enabled": True,
            "alpha": 0.6,
            "c": 0.15,
            "reflection": False,
        },
    })
    llm = FakeLLM()
    moh = MoH(cfg, ROOT_DIR, heu_llm=llm, meta_llm=llm)
    moh._cur_iter = 1
    moh.meta_utility_val = 15.0

    # Populate rmoh_pop with seed
    moh.rmoh_pop.add_individual("opt_seed", "def opt_seed(): pass", "seed", 15.0)

    # Mock candidate improver generator
    candidate_code = (
        "from utils.utils import extract_code\n"
        "def improve_algorithm(population, utility, language_model, function_format, task):\n"
        "    return 'new_idea', 'def h(): return 1', 12.0\n"
    )

    def mock_improver_func(pop, util_fn, lm, prompt, task):
        assert "DIAGNOSTIC FEEDBACK FROM PARENT OPTIMIZER" not in prompt or "Key Weaknesses" in prompt
        return "candidate_idea", candidate_code, 12.0

    with patch.object(moh, "meta_utility", return_value=12.0):
        prev_algo = Mock()
        success, new_code, next_algo = moh.try_improvement(mock_improver_func, prev_algo)

        assert success is True
        assert new_code == candidate_code
        assert moh.meta_utility_val == 12.0
        # Check that offspring was registered to parent in rmoh_pop
        assert len(moh.rmoh_pop) >= 2
        parent = moh.rmoh_pop.get_by_id("opt_seed")
        assert len(parent.offspring_utilities) == 1

