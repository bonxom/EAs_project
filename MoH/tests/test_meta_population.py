import pytest
from utils.diagnostics import TrajectoryDiagnostics
from utils.meta_population import MetaPopulation, OptimizerIndividual


def test_optimizer_individual_meta_uiq_no_offspring():
    diag = TrajectoryDiagnostics(valid_rate=0.8)
    ind = OptimizerIndividual(
        opt_id="opt_1",
        code="def solve(): pass",
        thought="seed",
        raw_utility=15.0,
        diagnostics=diag,
    )
    ind.normalized_utility = 0.7  # manually set for test

    # total_outer_steps = 10, alpha = 0.6, c = 0.15, selection_count = 0
    # expected_q = 0.8 * 0.7 + 0.2 * 0.8 = 0.56 + 0.16 = 0.72
    # exploitation = 0.6 * 0.7 + 0.4 * 0.72 = 0.42 + 0.288 = 0.708
    # exploration = 0.15 * sqrt(ln(11) / 1) = 0.15 * 1.5485 = 0.2323
    score = ind.compute_meta_uiq(total_outer_steps=10, alpha=0.6, c=0.15)
    assert score == pytest.approx(0.708 + 0.15 * (2.397895**0.5), rel=1e-3)


def test_optimizer_individual_meta_uiq_with_offspring():
    diag = TrajectoryDiagnostics(valid_rate=1.0)
    ind = OptimizerIndividual(
        opt_id="opt_parent",
        code="def solve(): pass",
        thought="parent",
        raw_utility=12.0,
        diagnostics=diag,
    )
    ind.normalized_utility = 0.8
    ind.selection_count = 2
    ind.offspring_utilities = [0.85, 0.95]  # mean = 0.90

    # exploitation = 0.6 * 0.8 + 0.4 * 0.9 = 0.48 + 0.36 = 0.84
    # exploration = 0.15 * sqrt(ln(5 + 1) / (2 + 1)) = 0.15 * sqrt(ln(6) / 3)
    score = ind.compute_meta_uiq(total_outer_steps=5, alpha=0.6, c=0.15)
    assert ind.selection_count == 2
    assert score > 0.84


def test_meta_population_normalization():
    pop = MetaPopulation(size=3)
    pop.add_individual("opt_1", "code1", "idea1", raw_utility=20.0)
    pop.add_individual("opt_2", "code2", "idea2", raw_utility=10.0)
    pop.add_individual("opt_3", "code3", "idea3", raw_utility=15.0)

    # raw min = 10.0 (best), raw max = 20.0 (worst)
    # opt_2 should have normalized_utility = 1.0 (best)
    # opt_1 should have normalized_utility = 0.0 (worst)
    # opt_3 should have normalized_utility = 0.5
    ind_map = {ind.opt_id: ind for ind in pop.get_all()}
    assert ind_map["opt_2"].normalized_utility == pytest.approx(1.0)
    assert ind_map["opt_1"].normalized_utility == pytest.approx(0.0)
    assert ind_map["opt_3"].normalized_utility == pytest.approx(0.5)


def test_meta_population_culling_preserves_elitism():
    pop = MetaPopulation(size=2)
    # Add opt_1 (best raw utility: 5.0)
    ind1 = pop.add_individual("opt_1", "code1", "idea1", raw_utility=5.0)
    # Add opt_2 (raw utility: 15.0)
    pop.add_individual("opt_2", "code2", "idea2", raw_utility=15.0)

    # Give opt_1 high selection count so its UCB exploration term drops
    ind1.selection_count = 50
    # Add opt_3 with raw utility 10.0
    pop.add_individual("opt_3", "code3", "idea3", raw_utility=10.0)

    # Population size is now 3, max size is 2 => cull() should trigger
    pop.cull(total_outer_steps=10)

    remaining_ids = [ind.opt_id for ind in pop.get_all()]
    assert len(remaining_ids) == 2
    # Elitist opt_1 (raw utility 5.0) MUST NEVER be culled
    assert "opt_1" in remaining_ids


def test_meta_population_selection_increments_count():
    pop = MetaPopulation(size=5)
    pop.add_individual("opt_1", "code1", "idea1", raw_utility=10.0)
    pop.add_individual("opt_2", "code2", "idea2", raw_utility=12.0)

    parent = pop.select_parent(total_outer_steps=1)
    assert parent.selection_count == 1
    assert parent.opt_id in ["opt_1", "opt_2"]
