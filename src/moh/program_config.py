"""Strict configuration for reproducible program-based MoH experiments."""

import math
from pathlib import Path
from typing import Annotated, Literal

import yaml
from pydantic import (
    BeforeValidator,
    Field,
    StrictFloat,
    StrictInt,
    field_validator,
    model_validator,
)

from moh.config import LLMConfig, StrictConfig, UniqueKeyLoader

# Keep worker integers and deadlines within portable operating-system bounds.
PositiveInt = Annotated[StrictInt, Field(gt=0, le=2**31 - 1)]
NonnegativeInt = Annotated[StrictInt, Field(ge=0, le=2**31 - 1)]


def _timeout(value):
    if type(value) not in (int, float):
        raise ValueError("timeout must be a finite positive number")
    try:
        number = float(value)
    except OverflowError as exc:
        raise ValueError("timeout must be at most 86400 seconds") from exc
    if not math.isfinite(number) or not 0 < number <= 86400:
        raise ValueError("timeout must be finite and between zero and 86400 seconds")
    return number


Timeout = Annotated[StrictFloat, BeforeValidator(_timeout)]
Weight = Annotated[StrictFloat, Field(gt=0, allow_inf_nan=False)]


class ProgramLLMConfig(LLMConfig):
    timeout_seconds: Timeout = 30.0


class DatasetConfig(StrictConfig):
    path: Path
    validation_indices: list[NonnegativeInt]
    test_indices: list[NonnegativeInt]

    @field_validator("path", mode="before")
    @classmethod
    def parse_path(cls, value):
        return Path(value) if isinstance(value, str) else value

    @model_validator(mode="after")
    def validate_splits(self):
        for indices in (self.validation_indices, self.test_indices):
            if not indices or len(set(indices)) != len(indices):
                raise ValueError("dataset indices must be nonempty and unique")
        if set(self.validation_indices) & set(self.test_indices):
            raise ValueError("validation and test splits must be disjoint")
        return self


class ProgramTasksConfig(StrictConfig):
    source: Literal["synthetic", "npz"] = "synthetic"
    sizes: list[PositiveInt] = Field(default_factory=lambda: [4, 6])
    validation_count: PositiveInt = 2
    test_count: PositiveInt = 1
    datasets: list[DatasetConfig] = Field(default_factory=list)

    @model_validator(mode="before")
    @classmethod
    def dataset_sizes(cls, values):
        if (
            isinstance(values, dict)
            and values.get("source") == "npz"
            and "sizes" not in values
        ):
            return {**values, "sizes": []}
        return values

    @model_validator(mode="after")
    def validate_source(self):
        if self.source == "synthetic":
            if (
                not self.sizes
                or len(set(self.sizes)) != len(self.sizes)
                or any(not 4 <= size <= 12 for size in self.sizes)
            ):
                raise ValueError(
                    "synthetic sizes must be distinct integers between 4 and 12"
                )
            if self.datasets:
                raise ValueError("synthetic tasks cannot specify datasets")
        else:
            if not self.datasets or self.sizes:
                raise ValueError(
                    "NPZ tasks require datasets; sizes are read from the data"
                )
            paths = [dataset.path.resolve() for dataset in self.datasets]
            if len(set(paths)) != len(paths):
                raise ValueError("dataset paths must be distinct")
        return self


class ProgramSolverConfig(StrictConfig):
    iterations: NonnegativeInt = 3
    perturbation_moves: PositiveInt = 1
    edges_per_move: PositiveInt = 5
    neighborhood_size: PositiveInt = 100
    reset_interval: PositiveInt = 50


class ProgramExecutionConfig(StrictConfig):
    run_timeout_seconds: Timeout = 120.0
    optimizer_timeout_seconds: Timeout = 30.0
    heuristic_timeout_seconds: Timeout = 5.0
    source_bytes: PositiveInt = 65536
    request_bytes: PositiveInt = 1048576
    result_bytes: PositiveInt = 1048576
    output_bytes: PositiveInt = 65536
    max_callbacks: PositiveInt = 100
    batch_size: PositiveInt = 2


class ProgramBudgetConfig(StrictConfig):
    # Zero deliberately permits reproducible exhausted-budget runs; None is unlimited.
    max_llm_calls: NonnegativeInt | None = 100
    max_heuristic_evaluations: NonnegativeInt | None = 100


class ProgramConfig(StrictConfig):
    seed: Annotated[StrictInt, Field(ge=0, le=2**32 - 1)] = 42
    output_dir: Path = Path("outputs")
    outer_iterations: NonnegativeInt = 2
    population_size: PositiveInt = 3
    seed_attempts: NonnegativeInt = 1
    tasks: ProgramTasksConfig = Field(default_factory=ProgramTasksConfig)
    weights: list[Weight] | None = None
    heuristic_llm: ProgramLLMConfig = Field(default_factory=ProgramLLMConfig)
    meta_llm: ProgramLLMConfig = Field(default_factory=ProgramLLMConfig)
    solver: ProgramSolverConfig = Field(default_factory=ProgramSolverConfig)
    execution: ProgramExecutionConfig = Field(default_factory=ProgramExecutionConfig)
    budgets: ProgramBudgetConfig = Field(default_factory=ProgramBudgetConfig)

    @field_validator("output_dir", mode="before")
    @classmethod
    def parse_path(cls, value):
        return Path(value) if isinstance(value, str) else value

    @model_validator(mode="after")
    def normalize_weights(self):
        count = (
            len(self.tasks.sizes)
            if self.tasks.source == "synthetic"
            else len(self.tasks.datasets)
        )
        if self.weights is None:
            # Dataset dimensions are validated/read before artifacts; resolve size weights there.
            if self.tasks.source == "npz":
                return self
            values = self.tasks.sizes
        else:
            values = self.weights
        if len(values) != count:
            raise ValueError("weights must match the task count")
        total = sum(values)
        if math.isfinite(total):
            normalized = [value / total for value in values]
        else:
            scale = max(values)
            scaled = [value / scale for value in values]
            total = math.fsum(scaled)
            normalized = [value / total for value in scaled]
        if any(value <= 0 for value in normalized):
            raise ValueError("normalized weights must remain positive")
        object.__setattr__(self, "weights", normalized)
        return self


def load_program_config(path: Path) -> ProgramConfig:
    data = yaml.load(Path(path).read_text(encoding="utf-8"), Loader=UniqueKeyLoader)
    return ProgramConfig.model_validate({} if data is None else data)
