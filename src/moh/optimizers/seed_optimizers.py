from moh.core.models import OptimizerCandidate
from moh.core.specs import OptimizerSpec


def initial_optimizer_candidates(inner_capacity):
    return tuple(
        OptimizerCandidate(
            f"o{index:06d}",
            OptimizerSpec(
                parent_selection="best",
                generation_operator=operator,
                use_reflection=False,
                survivor_selection="elitist",
                population_size=inner_capacity,
            ),
        )
        for index, operator in enumerate(("mutate", "crossover"), 1)
    )
