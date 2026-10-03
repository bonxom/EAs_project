import pickle
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from test_gls_tour import square_distances

from moh.problems.tsp_gls.dataset import load_npz_task, synthetic_task
from moh.problems.tsp_gls.task import GLSInstance, GLSTask


def arrays():
    return {
        "coordinates": np.zeros((3, 4, 2)),
        "distance_matrix": np.repeat(square_distances()[None], 3, axis=0),
        "cost": np.full(3, 4.0),
        "optimal_tour": np.tile([2, 3, 0, 1], (3, 1)),
    }


def test_synthetic_repeat_and_disjoint_splits():
    left = synthetic_task(6, 2, 2, 42)
    right = synthetic_task(6, 2, 2, 42)
    assert left.id == "tsp_gls6"
    assert {x.id for x in left.validation}.isdisjoint(x.id for x in left.test)
    for a, b in zip(
        (*left.validation, *left.test), (*right.validation, *right.test), strict=True
    ):
        np.testing.assert_array_equal(a.distances, b.distances)
        assert a.optimal_cost == b.optimal_cost
        assert not a.distances.flags.writeable
        with pytest.raises(ValueError):
            a.distances.setflags(write=True)


def test_npz_authoritative_matrix_and_provenance(tmp_path):
    path = tmp_path / "data.npz"
    np.savez_compressed(path, **arrays())
    task = load_npz_task(path, (0, 1), (2,))
    assert task.validation[0].optimal_tour == (0, 1, 2, 3, 0)
    assert len(task.provenance["sha256"]) == 64
    assert task.size == 4


@pytest.mark.parametrize(
    "validation,test",
    [
        ((0, 1), (1, 2)),
        ((0, 0), (1,)),
        ((-1,), (0,)),
        ((0.0,), (1,)),
        ((False,), (1,)),
        ((), (1,)),
    ],
)
def test_reject_bad_split_before_reading_file(tmp_path, validation, test):
    with pytest.raises(ValueError, match="split"):
        load_npz_task(tmp_path / "absent.npz", validation, test)


def test_reject_out_of_range_split(tmp_path):
    path = tmp_path / "data.npz"
    np.savez(path, **arrays())
    with pytest.raises(ValueError, match="split"):
        load_npz_task(path, (0,), (3,))


@pytest.mark.parametrize(
    "error",
    [
        "cost",
        "boolean_tour",
        "float_tour",
        "repeated_tour",
        "negative",
        "nan",
        "asymmetric",
        "diagonal",
        "object",
        "coordinates",
        "shape",
    ],
)
def test_reject_bad_npz(tmp_path, error):
    data = arrays()
    if error == "cost":
        data["cost"][0] = 5
    elif error == "boolean_tour":
        data["optimal_tour"] = data["optimal_tour"].astype(bool)
    elif error == "float_tour":
        data["optimal_tour"] = data["optimal_tour"].astype(float)
    elif error == "repeated_tour":
        data["optimal_tour"][0] = [0, 1, 1, 3]
    elif error == "object":
        data["coordinates"] = data["coordinates"].astype(object)
    elif error == "coordinates":
        data["coordinates"][0, 0, 0] = np.inf
    elif error == "shape":
        data["cost"] = np.ones((3, 1))
    else:
        data["distance_matrix"][0, 0, 1] = {
            "negative": -1,
            "nan": np.nan,
            "asymmetric": 2,
            "diagonal": 1,
        }[error]
        if error == "diagonal":
            data["distance_matrix"][0, 0, 0] = 1
    path = tmp_path / "data.npz"
    np.savez(path, **data)
    with pytest.raises(ValueError):
        load_npz_task(path, (0,), (1,))


def test_instance_defensive_copy_and_task_validation():
    matrix = square_distances()
    instance = GLSInstance("one", np.zeros((4, 2)), matrix, 4.0, (0, 1, 2, 3, 0))
    matrix[0, 1] = 99
    assert instance.distances[0, 1] == 1
    with pytest.raises(ValueError):
        GLSTask("task", 4, (instance,), (instance,), {})


def test_trusted_local_pickle_conversion(tmp_path):
    source = tmp_path / "fixture.pkl"
    target = tmp_path / "converted.npz"
    data = arrays()
    data["coordinate"] = data.pop("coordinates")
    source.write_bytes(pickle.dumps(data))
    command = [
        sys.executable,
        str(Path("tools/convert_moh_dataset.py")),
        "--trusted-pickle",
        str(source),
        "--output",
        str(target),
    ]
    result = subprocess.run(
        command, capture_output=True, text=True, check=False, timeout=10
    )
    assert result.returncode == 0, result.stderr
    assert load_npz_task(target, (0,), (1,)).size == 4
    again = subprocess.run(
        command, capture_output=True, text=True, check=False, timeout=10
    )
    assert again.returncode != 0


@pytest.mark.parametrize("dtype", [np.uint8, np.int8])
def test_npz_accepts_valid_narrow_integer_distances(tmp_path, dtype):
    data = arrays()
    distances = np.full((3, 4, 4), 100, dtype=dtype)
    for matrix in distances:
        np.fill_diagonal(matrix, 0)
    data["distance_matrix"] = distances
    data["cost"] = np.full(3, 400)
    path = tmp_path / "integer.npz"
    np.savez(path, **data)
    task = load_npz_task(path, (0,), (1,))
    assert task.validation[0].optimal_cost == 400.0
    assert task.validation[0].distances.dtype == np.float64
