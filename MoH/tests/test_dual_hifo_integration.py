"""Exercise the MoH hook boundaries without the external API or TSP dataset."""

import json
from types import SimpleNamespace

from utils.dual_hifo import DualHiFoMoHMixin, HiFoConfig


class FakeLLM:
    batch_size = 2
    model = "fake"
    temperature = 0.7

    def __init__(self):
        self.messages = []

    def prompt(self, expertise, message, temperature=None):
        self.messages.append(message)
        return "response"

    def prompt_batch(self, expertise, message_batch, temperature=None):
        self.messages.extend(message_batch)
        return ["response"] * len(message_batch)


SEED = "def improve_algorithm(population, utility, language_model, function_format, task):\n    return ('a', 'b', 1)"
NEW = "def improve_algorithm(population, utility, language_model, function_format, task):\n    return ('a', 'c', 0.5)"


class FakeMoH:
    def __init__(self, output_dir):
        self.subtask_list = ["tsp_gls-20", "tsp_gls-50"]
        self.heu_llm = FakeLLM()
        self.meta_llm = FakeLLM()
        self.run_logger = SimpleNamespace(output_dir=str(output_dir))
        self.improver_str = SEED
        self.meta_utility_val = 10.0
        self._cur_iter = 0

    def get_seed(self):
        return "seed-result"

    def evaluate_heuristic(self, algorithm_str, problem_type, mode="val"):
        return 3.0 if problem_type.endswith("20") else 5.0

    def _make_subtask_utility(self, subtask):
        def utility(algorithm_str, idea=None, problem_type=None):
            problem_type = problem_type or subtask
            return self.evaluate_heuristic(algorithm_str, problem_type, mode="val")
        return utility

    def get_improver(self, improve_str, subtask_str, subtask):
        self.heu_llm.prompt("writer", "generate heuristic")
        return self._make_subtask_utility(subtask)("def x(): return 1")

    def try_improvement(self, improve_algorithm_func, previous_algo):
        self.meta_llm.prompt("writer", "generate optimizer")
        self.improver_str = NEW
        self.meta_utility_val = 5.0
        return True, NEW, previous_algo


class Harness(DualHiFoMoHMixin, FakeMoH):
    def __init__(self, output_dir, mode):
        super().__init__(output_dir)
        self.init_dual_hifo(HiFoConfig(mode=mode))


def test_full_moh_does_not_modify_prompts(tmp_path):
    h = Harness(tmp_path, "full_moh")
    assert h.get_seed() == "seed-result"
    h.get_improver(SEED, "desc", "tsp_gls-20")
    h.try_improvement(None, None)
    assert h.heu_llm.messages == ["generate heuristic"]
    assert h.meta_llm.messages == ["generate optimizer"]
    assert not list(tmp_path.glob("dual_hifo_events.jsonl"))


def test_inner_outer_hooks_do_not_replace_original_clients(tmp_path):
    h = Harness(tmp_path, "dual_hifo_independent")
    original_heu, original_meta = h.heu_llm, h.meta_llm
    h.get_seed()
    h.get_improver(SEED, "desc", "tsp_gls-20")
    h.try_improvement(None, None)
    assert h.heu_llm is original_heu
    assert h.meta_llm is original_meta
    assert "Level: INNER" in original_heu.messages[0]
    assert "Level: OUTER" in original_meta.messages[0]
    assert "Completed inner-search evidence" not in original_meta.messages[0]
    assert h.inner_hifo["tsp_gls-50"].step == 0
    events = [json.loads(line) for line in (tmp_path / "dual_hifo_events.jsonl").read_text().splitlines()]
    assert {e["level"] for e in events} == {"inner", "outer"}
    assert any(e.get("accepted") is True for e in events)


def test_cross_level_outer_prompt_includes_completed_inner_trace(tmp_path):
    h = Harness(tmp_path, "dual_hifo_cross_level")
    h.get_seed()
    h.get_improver(SEED, "desc", "tsp_gls-20")
    h.try_improvement(None, None)
    assert "Completed inner-search evidence" in h.meta_llm.messages[0]
    assert "tsp_gls-20" in h.meta_llm.messages[0]
    assert "Directive:" in h.meta_llm.messages[0]


def test_cross_level_outer_prompt_includes_distilled_mechanisms(tmp_path):
    h = Harness(tmp_path, "dual_hifo_cross_level")
    h.get_seed()
    # Simulate an inner subtask improving with a loop mechanism
    heuristic_code = "def solve(inst):\n    for i in range(10):\n        pass\n    return 0"
    h.inner_hifo["tsp_gls-20"].observe(heuristic_code, 2.0)
    h.try_improvement(None, None)
    msg = h.meta_llm.messages[0]
    assert "Validated heuristic mechanisms across subtasks:" in msg
    assert "Iteratively evaluate a bounded collection of candidates" in msg
    assert "active in: tsp_gls-20" in msg
