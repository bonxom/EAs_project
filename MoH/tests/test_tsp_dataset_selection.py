import pickle

import pytest
from problems.tsp_gls.dataset_selection import (
    load_eval_dataset,
    select_instance_indices,
)


def dataset(n):
    return {key: list(range(n)) for key in ("cost", "distance_matrix", "coordinate", "optimal_tour")}


def test_validation_samples_eight_unique_instances_reproducibly():
    data = dataset(128)
    indices = select_instance_indices(data, "val", seed=42)
    assert len(indices) == len(set(indices)) == 8
    assert indices == select_instance_indices(data, "val", seed=42)
    assert indices != select_instance_indices(data, "val", seed=43)
    assert indices != list(range(8))


def test_test_count_is_explicit():
    data = dataset(128)
    assert len(select_instance_indices(data, "test")) == 100
    assert len(select_instance_indices(data, "test", count=128)) == 128


@pytest.mark.parametrize(
    ("mode", "available", "requested"), [("val", 7, None), ("test", 127, 128)]
)
def test_insufficient_data_fails_instead_of_reusing_instances(
    mode, available, requested
):
    with pytest.raises(ValueError, match="requires"):
        select_instance_indices(dataset(available), mode, count=requested)


def test_rejects_inconsistent_dataset_fields():
    data = dataset(128)
    data["optimal_tour"] = [0]
    with pytest.raises(ValueError, match="length"):
        select_instance_indices(data, "test")


def test_rejects_unknown_mode():
    with pytest.raises(ValueError, match="mode"):
        select_instance_indices(dataset(128), "train")


@pytest.mark.parametrize("mode", ["val", "test"])
def test_loads_data_from_requested_split(tmp_path, mode):
    directory = tmp_path / "problems" / "tsp_gls" / "data" / mode
    directory.mkdir(parents=True)
    path = directory / f"{mode}_instance_data_sol_100.pkl"
    with path.open("wb") as output:
        pickle.dump(dataset(128), output)
    requested = 128 if mode == "test" else None
    data, indices, source = load_eval_dataset(
        tmp_path, 100, mode, seed=42, count=requested
    )
    assert source == path
    assert len(indices) == (8 if mode == "val" else 128)
    assert len(data["cost"]) == 128
