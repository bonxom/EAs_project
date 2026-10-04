import unittest
from types import SimpleNamespace
from unittest.mock import Mock

from moh import MoH
from utils.generated_improver import GeneratedImprover
from utils.population import Pop


class FakeLLM:
    batch_size = 1

    def __init__(self):
        self.calls = []

    def prompt(self, *args, **kwargs):
        self.calls.append((args, kwargs))
        return "fake"

    def prompt_batch(self, *args, **kwargs):
        self.calls.append((args, kwargs))
        return ["fake"]


class ImproverValidationTests(unittest.TestCase):
    def setUp(self):
        self.optimizer = MoH.__new__(MoH)
        self.optimizer.improver_pop = Pop(["meta-optimizer"], 10)
        self.optimizer.subtask_pop = Pop(["tsp_gls-100"], 10)
        self.optimizer.improver_str = "previous code"
        self.optimizer.meta_utility_val = 5.0
        self.optimizer._total_eval_calls = 0
        self.optimizer.meta_llm = FakeLLM()
        self.optimizer.cfg = SimpleNamespace(optimizer_timeout=10, timeout=10)
        self.optimizer.meta_prompts = "prompt"
        self.optimizer.run_logger = Mock()
        self.previous = Mock()

    def test_invalid_candidates_preserve_previous_optimizer(self):
        candidates = [
            'def improve_algorithm():\n    return f"unterminated\n',
            'raise RuntimeError("load failed")',
            'other_function = lambda: 1',
            None,
        ]
        for code in candidates:
            with self.subTest(code=code):
                generator = Mock(return_value=("idea", code, 1.0))
                success, _, active = self.optimizer.try_improvement(generator, self.previous)
                self.assertFalse(success)
                self.assertIs(active, self.previous)
                self.assertEqual(self.optimizer.improver_str, "previous code")
                self.assertEqual(self.optimizer.meta_utility_val, 5.0)
                self.assertEqual(self.optimizer.improver_pop.get_subtask_size("meta-optimizer"), 0)

    def test_valid_candidate_is_loaded_and_saved(self):
        code = 'def improve_algorithm(*args):\n    return "loaded"\n'
        generator = Mock(return_value=("idea", code, 1.0))
        success, returned_code, active = self.optimizer.try_improvement(generator, self.previous)
        self.assertTrue(success)
        self.assertEqual(returned_code, code)
        self.assertEqual(active(self.optimizer.improver_pop, Mock(), FakeLLM(), "format", "task"), "loaded")
        self.assertEqual(self.optimizer.improver_pop.get_best_solution("meta-optimizer")["best_sol"], code.strip())

    def test_meta_rejects_syntax_error_before_llm_call(self):
        score = self.optimizer.meta_utility('def broken(:', "idea")
        self.assertEqual(score, 1e6)
        self.assertEqual(self.optimizer.meta_llm.calls, [])

    def test_unchanged_optimizer_is_not_accepted(self):
        code = 'def improve_algorithm(*args):\n    return "old"\n'
        self.optimizer.improver_str = code
        generator = Mock(return_value=("new idea", "# new comment\n" + code, 1.0))
        success, _, active = self.optimizer.try_improvement(generator, self.previous)
        self.assertFalse(success)
        self.assertIs(active, self.previous)
        self.assertEqual(self.optimizer.meta_utility_val, 5.0)

    def test_non_improving_and_invalid_scores_are_not_accepted(self):
        code = 'def improve_algorithm(*args):\n    return "new"\n'
        for score in (5.0, 6.0, float("nan"), float("inf"), None, -1.0):
            with self.subTest(score=score):
                generator = Mock(return_value=("idea", code, score))
                success, _, active = self.optimizer.try_improvement(generator, self.previous)
                self.assertFalse(success)
                self.assertIs(active, self.previous)
                self.assertEqual(self.optimizer.meta_utility_val, 5.0)

    def test_zero_gap_is_valid(self):
        code = 'def improve_algorithm(*args):\n    return "perfect"\n'
        generator = Mock(return_value=("idea", code, 0.0))
        success, _, _ = self.optimizer.try_improvement(generator, self.previous)
        self.assertTrue(success)
        self.assertEqual(self.optimizer.meta_utility_val, 0.0)

    def test_raw_llm_candidate_reaches_outer_utility(self):
        candidate_code = 'def improve_algorithm(*args):\n    return "candidate"\n'
        self.optimizer.meta_llm.prompt_batch = Mock(return_value=[candidate_code])
        self.optimizer.meta_utility = Mock(return_value=1.0)
        generator_code = '''from utils.utils import extract_code
def improve_algorithm(population, utility, language_model, function_format, task):
    elite = population.get_solution_by_index(task, 0)
    best_code, best_score = elite["best_sol"], elite["utility"]
    responses = language_model.prompt_batch("expert", [function_format])
    for code in extract_code(responses):
        if not code:
            continue
        score = utility(code, "idea", task)
        if score < best_score:
            best_code, best_score = code, score
    return "idea", best_code, best_score
'''
        self.optimizer.improver_str = generator_code
        self.optimizer.improver_pop.save_solution("meta-optimizer", "old", generator_code, 5.0)
        generator = GeneratedImprover(generator_code, timeout=10)
        success, returned_code, _ = self.optimizer.try_improvement(generator, self.previous)
        self.assertTrue(success)
        self.assertEqual(returned_code, candidate_code.strip())
        self.optimizer.meta_utility.assert_called_once_with(candidate_code.strip(), "idea", "meta-optimizer")

    def test_inner_optimizer_uses_parent_evaluator(self):
        self.optimizer.heu_llm = FakeLLM()

        def evaluate(*args, **kwargs):
            self.optimizer._total_eval_calls += 1
            return 0.5

        self.optimizer.evaluate_heuristic = Mock(side_effect=evaluate)
        code = '''def improve_algorithm(population, utility, language_model, function_format, task):
    code = "def heuristic():\\n    return 1"
    score = utility(code, "new", task)
    return "new", code, score
'''
        result = self.optimizer.get_improver(code, "format", "tsp_gls-100")
        self.assertEqual(result, ("new", "def heuristic():\n    return 1", 0.5))
        self.assertEqual(self.optimizer._total_eval_calls, 1)

    def test_outer_loop_continues_after_rejection(self):
        self.optimizer.get_seed = Mock()
        self.optimizer.iteration = 2
        self.optimizer.max_eval_calls = None
        self.optimizer.try_improvement = Mock(return_value=(False, None, self.previous))
        self.optimizer.run_meta_optimizer()
        self.assertEqual(self.optimizer.try_improvement.call_count, 2)
        initial_improver = self.optimizer.try_improvement.call_args_list[0].args[0]
        self.assertIs(self.optimizer.try_improvement.call_args.args[0], initial_improver)
        self.assertEqual(self.optimizer.run_logger.save_iteration.call_count, 2)


if __name__ == "__main__":
    unittest.main()
