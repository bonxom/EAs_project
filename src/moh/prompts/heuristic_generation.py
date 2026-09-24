import json


def heuristic_prompt(kind, parents, idea=None):
    if kind not in ("mutate", "crossover", "reflection"):
        raise ValueError("invalid heuristic prompt kind")
    data = [
        {
            "id": p.heuristic.id,
            "source_code": p.heuristic.source_code,
            "utility": p.evaluation.utility,
            "status": p.evaluation.status,
        }
        for p in parents
    ]
    rules = (
        "Return only a short improvement idea."
        if kind == "reflection"
        else "Return only Python code defining select_next_node(current_node, unvisited, coordinates) -> int. "
        "Return one city from unvisited; ties use smallest index. No imports, file IO, network IO, or global mutation. "
        "coordinates is an Nx2 NumPy array. Do not modify inputs."
    )
    return f"KIND: {kind}\n{rules}\n" + json.dumps(
        {"parents": data, "idea": idea}, sort_keys=True, allow_nan=False
    )
