"""Only this disposable process compiles or executes heuristic programs."""

import json
import os
import random
import sys

import numpy as np


class InvalidCandidate(Exception):
    pass


def main():
    result_fd = int(sys.argv[1])
    request = json.loads(sys.stdin.buffer.read())
    result = {
        "id": request["id"],
        "status": "failed",
        "tour": None,
        "error": "exception",
    }
    try:
        random.seed(request["seed"])
        np.random.seed(request["seed"])
        coordinates = np.array(request["coordinates"], dtype=float)
        namespace = {"random": random, "np": np}
        exec(compile(request["source"], "<heuristic>", "exec"), namespace)  # noqa: S102 — child only
        choose = namespace.get("select_next_node")
        if not callable(choose):
            raise InvalidCandidate("missing_function")
        tour, remaining = [0], set(range(1, len(coordinates)))
        while remaining:
            node = choose(tour[-1], sorted(remaining), coordinates.copy())
            if (
                isinstance(node, (bool, np.bool_))
                or not isinstance(node, (int, np.integer))
                or int(node) not in remaining
            ):
                raise InvalidCandidate("invalid_return")
            remaining.remove(int(node))
            tour.append(int(node))
        tour.append(0)
        result.update(status="success", tour=tour, error=None)
    except SyntaxError:
        result["error"] = "syntax"
    except InvalidCandidate as exc:
        result["error"] = str(exc)
    except BaseException:  # noqa: BLE001 — candidate SystemExit is a failed evaluation
        result["error"] = "exception"
    data = json.dumps(result, allow_nan=False).encode()
    while data:
        written = os.write(result_fd, data)
        data = data[written:]
    os.close(result_fd)


if __name__ == "__main__":
    main()
