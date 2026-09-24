import itertools

import pytest
from pydantic import ValidationError

from moh.core.specs import OptimizerSpec


@pytest.mark.parametrize(
    "field,bad",
    [("population_size", x) for x in [True, "3", 1, 11]]
    + [("use_reflection", 1), ("parent_selection", "other"), ("reflection", False)],
)
def test_reject_invalid(base_spec, field, bad):
    with pytest.raises(ValidationError):
        OptimizerSpec.model_validate({**base_spec, field: bad})


def test_missing_and_all_options(base_spec):
    for field in base_spec:
        with pytest.raises(ValidationError):
            OptimizerSpec.model_validate(
                {k: v for k, v in base_spec.items() if k != field}
            )
    for p, g, r, s in itertools.product(
        ["best", "random", "tournament"],
        ["mutate", "crossover"],
        [True, False],
        ["elitist", "diversity"],
    ):
        assert (
            OptimizerSpec(
                parent_selection=p,
                generation_operator=g,
                use_reflection=r,
                survivor_selection=s,
                population_size=2,
            ).population_size
            == 2
        )
