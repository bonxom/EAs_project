import json

from moh.logging import to_json


def optimizer_prompt(population):
    records = [
        {
            "id": x.candidate.id,
            "spec": x.candidate.spec.model_dump(),
            "status": x.evaluation.status,
            "utility": x.evaluation.utility,
            "counts": to_json(x.evaluation.counts),
        }
        for x in population
    ]
    return (
        "KIND: optimizer_spec\nReturn only one JSON object with exactly these fields: "
        "parent_selection (random, best, tournament), generation_operator (mutate, crossover), "
        "use_reflection (boolean), survivor_selection (elitist, diversity), population_size (integer 2 through 10). "
        "Higher utility is better; counts describe actual work, not equal budgets.\n"
        + json.dumps(records, sort_keys=True, allow_nan=False)
    )
