import os
import subprocess
import sys

import pytest

from moh.core.seeds import derive_seed


def test_stable_unambiguous():
    assert derive_seed(42, "task", 10) == derive_seed(42, "task", 10)
    assert derive_seed(42, "ab", "c") != derive_seed(42, "a", "bc")
    assert derive_seed(42, "1") != derive_seed(42, 1)
    results = [
        subprocess.check_output(
            [
                sys.executable,
                "-c",
                'from moh.core.seeds import derive_seed; print(derive_seed(42,"task",10))',
            ],
            env={**os.environ, "PYTHONHASHSEED": value},
        )
        for value in ["1", "999"]
    ]
    assert results[0] == results[1]


@pytest.mark.parametrize("args", [(True,), (-1,), (42, None), (42, True)])
def test_invalid(args):
    with pytest.raises(ValueError):
        derive_seed(*args)
