import unittest
from unittest.mock import Mock

from moh import MoH
from utils.population import Pop


class ImproverValidationTests(unittest.TestCase):
    def setUp(self):
        self.optimizer = MoH.__new__(MoH)
        self.optimizer.improver_pop = Pop(["meta-optimizer"], 10)
        self.optimizer.subtask_pop = Pop(["tsp_gls-100"], 10)
        self.optimizer.improver_str = "previous code"
        self.optimizer.meta_utility_val = 5.0
        self.optimizer.meta_llm = Mock()
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
        self.assertEqual(active(), "loaded")
        self.assertEqual(self.optimizer.improver_pop.get_best_solution("meta-optimizer")["best_sol"], code.strip())

    def test_meta_rejects_syntax_error_before_llm_call(self):
        score = self.optimizer.meta_utility('def broken(:', "idea")
        self.assertEqual(score, 1e6)
        self.optimizer.meta_llm.prompt.assert_not_called()

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
