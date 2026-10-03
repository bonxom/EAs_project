import random

import pytest
from test_records import scored

from moh.core.models import Heuristic
from moh.core.program_population import PopulationTransaction, ProgramPopulation
from moh.core.programs import GapEvaluation, ProgramRunResult, ScoredProgram


def test_minimize_and_deduplicate():
    population = ProgramPopulation(2).add(scored("b", 2)).add(scored("a", 1))
    assert [x.id for x in population.members] == ["a", "b"]
    assert population.add(scored("c", 3)).members == population.members
    assert population.add(scored("d", 0, "# a")).members == population.members
    assert [x.id for x in population.add(scored("c", 1)).members] == ["a", "c"]


def test_transaction_and_snapshot_cannot_change_parent():
    original = {"tsp4": ProgramPopulation(2).add(scored("a", 2))}
    transaction = PopulationTransaction(original)
    transaction.add("tsp4", scored("b", 1))
    snapshot = original["tsp4"].snapshot("tsp4")
    assert snapshot == {
        "tsp4": [{"id": "a", "best_sol": "# a", "idea": None, "utility": 2}]
    }
    snapshot["tsp4"][0]["utility"] = -100
    assert original["tsp4"].best().utility == 2
    committed = transaction.commit()
    assert committed["tsp4"].best().utility == 1
    committed.clear()
    assert transaction.staged["tsp4"].best().utility == 1


def test_rank_selection_reproducible():
    population = ProgramPopulation(3)
    for i in range(3):
        population = population.add(scored(str(i), i))
    a, b = random.Random(42), random.Random(42)
    assert [population.select(a).id for _ in range(20)] == [
        population.select(b).id for _ in range(20)
    ]


@pytest.mark.parametrize("capacity", [0, -1, True, 1.0])
def test_invalid_capacity(capacity):
    with pytest.raises(ValueError):
        ProgramPopulation(capacity)


def test_failed_member_and_conflicting_identity_rejected():
    failure = ScoredProgram(
        Heuristic("bad", "# bad"),
        GapEvaluation(
            "bad", "tsp4", "validation", "failed", None, (), (), (), "timeout"
        ),
    )
    with pytest.raises(ValueError):
        ProgramPopulation(2).add(failure)
    with pytest.raises(ValueError):
        ProgramPopulation(2).add(scored("a")).add(scored("a", 2, "# changed"))
    with pytest.raises(ValueError):
        ProgramPopulation(2).add(
            ScoredProgram(
                Heuristic("a", "# a"),
                GapEvaluation(
                    "a", "tsp4", "test", "success", 1, (4,), (1,), ((0, 1, 2, 3, 0),)
                ),
            )
        )


def test_constructor_normalizes_members_and_empty_selection():
    population = ProgramPopulation(1, [scored("b", 2), scored("a", 1)])
    assert [x.id for x in population.members] == ["a"]
    assert ProgramPopulation(1).best() is None
    with pytest.raises(ValueError):
        ProgramPopulation(1).select(random.Random(0))


def test_run_result_copies_collections():
    populations = {"tsp4": ProgramPopulation(1).add(scored())}
    result = ProgramRunResult(
        "failed", None, None, [], populations, [], counts=scored().evaluation.counts
    )
    populations.clear()
    assert result.task_populations["tsp4"].best().id == "a"
    assert result.population == result.test_results == ()
    with pytest.raises(TypeError):
        result.task_populations["other"] = ProgramPopulation(1)


def test_selection_uses_specified_rank_weights():
    class RecordingRandom:
        def choices(self, members, *, weights, k):
            assert weights == [1 / 3, 1 / 4, 1 / 5]
            assert k == 1
            return [members[0]]

    population = ProgramPopulation(4, tuple(scored(str(i), i) for i in range(3)))
    assert population.select(RecordingRandom()).id == "0"


def test_transaction_mapping_copied_and_missing_task_rejected():
    original = {"tsp4": ProgramPopulation(2).add(scored())}
    transaction = PopulationTransaction(original)
    original.clear()
    assert transaction.staged["tsp4"].best().id == "a"
    with pytest.raises(KeyError):
        transaction.add("missing", scored())


def test_success_run_tracks_active_separately_from_best():
    winner, active = scored("best", 0), scored("active", 2)
    result = ProgramRunResult(
        "success", winner, active, (winner,), {}, (), winner.evaluation.counts
    )
    assert result.winner.utility == 0
    assert result.active.utility == 2
    with pytest.raises(ValueError):
        ProgramRunResult("success", None, active, (), {}, (), active.evaluation.counts)


@pytest.mark.parametrize(
    "winner,active,members",
    [
        (scored(), None, (scored(),)),
        (None, scored(), ()),
        (None, None, (scored(),)),
    ],
)
def test_failed_run_cannot_carry_successful_optimizer_state(winner, active, members):
    with pytest.raises(ValueError):
        ProgramRunResult(
            "failed", winner, active, members, {}, (), scored().evaluation.counts
        )
