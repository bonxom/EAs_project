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
        if kind not in ("mutate", "crossover", "reflection", "optimizer_spec"):
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
        return (
            NEAREST_NEIGHBOR_SOURCE,
            "def select_next_node(current_node, unvisited, coordinates):\n    return max(unvisited, key=lambda j: sum((coordinates[current_node]-coordinates[j])**2))\n",
            "def select_next_node(current_node, unvisited, coordinates):\n    return min(unvisited)\n",
        )[index]

    def generate_request(self, request):
        from moh.llm.base import validate_response
        from moh.optimizers.seeds import (
            BASIC_SOURCE,
            BEST_PARENT_SOURCE,
            MULTI_TEMPERATURE_SOURCE,
        )

        first = request.message.split('\n', 1)[0]
        kind = first.removeprefix('KIND: ')
        if not first.startswith('KIND: ') or kind not in (
            'directions', 'heuristic_program', 'optimizer_program'
        ):
            raise GenerationError('unknown prompt kind')
        cursor = self.cursors[kind]
        self.cursors[kind] += 1
        if kind in self.responses:
            values = self.responses[kind]
            if cursor >= len(values):
                raise GenerationError('scripted responses exhausted')
            value = values[cursor]
            if isinstance(value, GenerationError):
                raise value
            return validate_response(value)
        index = (derive_seed(self.seed, kind) + cursor) % 3
        if kind == 'directions':
            content = json.dumps({'insights': [
                'Prefer frequently unused edges.', 'Penalize long tour edges.',
                'Balance distances and accumulated penalties.',
            ]})
            return f'```json\n{content}\n```'
        if kind == 'optimizer_program':
            source = (BASIC_SOURCE, MULTI_TEMPERATURE_SOURCE, BEST_PARENT_SOURCE)[index]
        else:
            # Independent baseline expressions avoid importing executable seed modules.
            factor = (0.1, 0.2, 0.4)[index]
            source = ('import numpy as np\n'
                      'def update_edge_distance(edge_distance, local_opt_tour, edge_n_used):\n'
                      f'    result = edge_distance + {factor} * edge_distance.mean() * edge_n_used\n'
                      '    np.fill_diagonal(result, 0.0)\n'
                      '    return result\n')
        return f'# {{offline variant {index}}}\n```python\n{source}\n```'
