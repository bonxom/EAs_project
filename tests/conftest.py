import pytest


@pytest.fixture
def base_spec():
    return {
        "parent_selection": "best",
        "generation_operator": "mutate",
        "use_reflection": False,
        "survivor_selection": "elitist",
        "population_size": 3,
    }
