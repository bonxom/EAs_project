import hashlib

from utils.behavior import BehaviorDescriptor, EvaluationIdentity
from utils.heuristic_evaluator import EvalResult
from utils.population import Pop

HEURISTIC = (
    "def update_edge_distance(edge_distance, local_opt_tour, edge_n_used):\n"
    "    return edge_distance\n"
)

OPTIMIZER = """from utils.utils import extract_code
def improve_algorithm(population, utility, language_model, function_format, task):
    responses = language_model.prompt_batch("expert", [function_format])
    for code in extract_code(responses):
        if code:
            utility(code, "generated", task)
    best = population.get_best_solution(task)
    return best["idea"], best["best_sol"], best["utility"]
"""


class FakeLLM:
    batch_size = 1
    model = "fake"
    temperature = 0.0

    def __init__(self, responses=None):
        self.responses = list(responses or [HEURISTIC])
        self.messages = []

    def prompt(self, expertise, message, temperature=None):
        self.messages.append(message)
        return self.responses.pop(0) if self.responses else HEURISTIC

    def prompt_batch(self, expertise, message_batch, temperature=None):
        return [
            self.prompt(expertise, message, temperature) for message in message_batch
        ]


class FakeEvaluator:
    def __init__(self, scores=None, failed_codes=(), vectors=None):
        self.vectors = vectors or {}
        self.scores = scores or {}
        self.failed_codes = set(failed_codes)
        self.calls = []
        self.seed, self.val_count, self.test_count = 0, 8, 100
        self.solver_settings = {
            "workers": 8,
            "instance_timeout": 90,
            "time_limit": 20,
            "iteration_limit": 1000,
            "perturbation_moves": 1,
        }

    def preflight(self, sizes, include_test):
        modes = ("val", "test") if include_test else ("val",)
        return {
            size: {
                mode: {
                    "dataset_path": f"fake/{mode}/{size}",
                    "dataset_sha256": hashlib.sha256(
                        f"{mode}:{size}".encode()
                    ).hexdigest(),
                    "indices": tuple(
                        range(
                            self.val_count if mode == "val" else self.test_count
                        )
                    ),
                }
                for mode in modes
            }
            for size in sizes
        }

    def evaluate(self, code, size, mode):
        self.calls.append((code, size, mode))
        failed = code in self.failed_codes
        manifest = self.preflight([size], include_test=True)[size][mode]
        gaps = tuple(self.vectors.get(code, [self.scores.get(code, 1.0)] * len(manifest["indices"])))
        score = sum(gaps) / len(gaps)
        behavior = None if failed else BehaviorDescriptor(
            EvaluationIdentity(size, mode, manifest["dataset_sha256"], manifest["indices"],
                               self.seed, tuple(self.solver_settings.items())), gaps, score)
        return EvalResult(
            size=size,
            mode=mode,
            mean_gap_pct=None if failed else score,
            behavior=behavior,
            instance_gaps=() if failed else gaps,
            status="failed" if failed else "ok",
            failures=0,
            indices=manifest["indices"],
            code_sha256=hashlib.sha256(code.encode()).hexdigest(),
            dataset_sha256=manifest["dataset_sha256"],
            elapsed_seconds=0.0,
            stdout_path="fake",
            error="fake failure" if failed else None,
        )


def training_population(sizes=(20, 50, 100, 200)):
    pop = Pop([f"tsp_gls-{size}" for size in sizes], 10)
    for size in sizes:
        pop.save_solution(f"tsp_gls-{size}", f"elite-{size}", HEURISTIC, 0.01)
    return pop
