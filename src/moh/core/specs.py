from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, StrictBool, StrictInt


class OptimizerSpec(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid", frozen=True)
    parent_selection: Literal["random", "best", "tournament"]
    generation_operator: Literal["mutate", "crossover"]
    use_reflection: StrictBool
    survivor_selection: Literal["elitist", "diversity"]
    population_size: Annotated[StrictInt, Field(ge=2, le=10)]
