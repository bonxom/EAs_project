"""Validate all configuration before any run artifacts, workers, or API calls."""

from pathlib import Path
from typing import Annotated, Literal

import yaml
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictFloat,
    StrictInt,
    field_validator,
    model_validator,
)

from moh.evaluation import normalized_weights

PositiveFloat = Annotated[StrictFloat, Field(gt=0, allow_inf_nan=False)]
PositiveInt = Annotated[StrictInt, Field(gt=0)]
NonnegativeInt = Annotated[StrictInt, Field(ge=0)]


class StrictConfig(BaseModel):
    model_config = ConfigDict(
        strict=True, extra="forbid", frozen=True, validate_default=True
    )


class OuterConfig(StrictConfig):
    iterations: NonnegativeInt = 2
    population_size: Annotated[StrictInt, Field(ge=2)] = 2


class InnerConfig(StrictConfig):
    iterations: NonnegativeInt = 2
    population_size: Annotated[StrictInt, Field(ge=2, le=10)] = 3


class TasksConfig(StrictConfig):
    sizes: list[StrictInt] = Field(default_factory=lambda: [10, 20])
    weights: list[PositiveFloat] | None = None

    @model_validator(mode="after")
    def validate_tasks(self):
        if (
            not self.sizes
            or len(set(self.sizes)) != len(self.sizes)
            or any(x not in (10, 20, 50) for x in self.sizes)
        ):
            raise ValueError("tasks must be distinct supported sizes 10, 20, 50")
        weights = self.weights if self.weights is not None else [1.0] * len(self.sizes)
        normalized_weights(weights, len(self.sizes))
        object.__setattr__(self, "weights", weights)
        return self


class LLMConfig(StrictConfig):
    provider: Literal["fake", "openai"] = "fake"
    model: str | None = None
    timeout_seconds: PositiveFloat = 30.0

    @model_validator(mode="after")
    def validate_model(self):
        if self.provider == "openai" and (not self.model or not self.model.strip()):
            raise ValueError("real provider requires an explicit model")
        return self


class ExecutionConfig(StrictConfig):
    timeout_seconds: PositiveFloat = 5.0
    source_bytes: PositiveInt = 65536
    result_bytes: PositiveInt = 1048576
    output_bytes: PositiveInt = 65536


class ExperimentConfig(StrictConfig):
    seed: NonnegativeInt = 42
    output_dir: Path = Path("outputs")
    outer: OuterConfig = Field(default_factory=OuterConfig)
    inner: InnerConfig = Field(default_factory=InnerConfig)
    tasks: TasksConfig = Field(default_factory=TasksConfig)
    instances_per_task: PositiveInt = 3
    llm: LLMConfig = Field(default_factory=LLMConfig)
    execution: ExecutionConfig = Field(default_factory=ExecutionConfig)

    @field_validator("output_dir", mode="before")
    @classmethod
    def parse_path(cls, value):
        if isinstance(value, str):
            return Path(value)
        return value


class UniqueKeyLoader(yaml.SafeLoader):
    pass


def _mapping(loader, node, deep=False):
    mapping = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        if not isinstance(key, str) or key in mapping:
            raise ValueError("YAML mapping keys must be unique strings")
        mapping[key] = loader.construct_object(value_node, deep=deep)
    return mapping


UniqueKeyLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _mapping
)


def load_config(path):
    data = yaml.load(Path(path).read_text(encoding="utf-8"), Loader=UniqueKeyLoader)
    return ExperimentConfig.model_validate({} if data is None else data)
