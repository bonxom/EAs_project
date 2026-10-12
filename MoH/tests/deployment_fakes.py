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


def training_population(sizes=(20, 50, 100, 200)):
    pop = Pop([f"tsp_gls-{size}" for size in sizes], 10)
    for size in sizes:
        pop.save_solution(f"tsp_gls-{size}", f"elite-{size}", HEURISTIC, 0.01)
    return pop
