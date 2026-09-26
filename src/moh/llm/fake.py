import json
from collections import defaultdict

from moh.core.seeds import derive_seed
from moh.llm.base import GenerationError
from moh.problems.baselines import NEAREST_NEIGHBOR_SOURCE


class FakeLLM:
    def __init__(self, seed, responses=None):
        derive_seed(seed)
        self.seed = seed
        self.responses = dict(responses or {})
        self.cursors = defaultdict(int)

    def generate(self, prompt):
        first = prompt.split("\n", 1)[0]
        if not first.startswith("KIND: "):
            raise GenerationError("missing prompt kind")
        kind = first[6:]
        if kind not in ("mutate", "crossover", "reflection", "optimizer_spec", "optimizer_program"):
            raise GenerationError("unknown prompt kind")
        cursor = self.cursors[kind]
        self.cursors[kind] += 1
        if kind in self.responses:
            values = self.responses[kind]
            if cursor >= len(values):
                raise GenerationError("scripted responses exhausted")
            value = values[cursor]
            if isinstance(value, GenerationError):
                raise value
            return value
        index = (derive_seed(self.seed, kind) + cursor) % 3
        if kind == "reflection":
            return (
                "Prefer short edges.",
                "Explore distant cities early.",
                "Use stable city ordering.",
            )[index]
        if kind == "optimizer_spec":
            return json.dumps(
                {
                    "parent_selection": ("best", "random", "tournament")[index],
                    "generation_operator": ("mutate", "crossover", "mutate")[index],
                    "use_reflection": index == 2,
                    "survivor_selection": "diversity" if index else "elitist",
                    "population_size": 3,
                },
                sort_keys=True,
            )
        if kind == "optimizer_program":
            return (
                "def improve_algorithm(api):\n    return None\n",
                "def improve_algorithm(api):\n    # deterministic option 2\n    return None\n",
                "def improve_algorithm(api):\n    # deterministic option 3\n    return None\n",
            )[index]
        return (
            NEAREST_NEIGHBOR_SOURCE,
            "def select_next_node(current_node, unvisited, coordinates):\n    return max(unvisited, key=lambda j: sum((coordinates[current_node]-coordinates[j])**2))\n",
            "def select_next_node(current_node, unvisited, coordinates):\n    return min(unvisited)\n",
        )[index]
