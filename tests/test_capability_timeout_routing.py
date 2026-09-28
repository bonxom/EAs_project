import json
import time
from dataclasses import replace
from pathlib import Path

from moh.experiments.runner import load_experiment_config, run_experiment
from moh.llm.base import GenerationError
from moh.optimizers.capabilities import CapabilityLimits, OptimizerCapabilityController
from moh.optimizers.programs import parse_optimizer_program
from moh.optimizers.runner import OptimizerProgramRunner, ProgramLimits


class DelayedFakeTransport:
    def __init__(self, responses, delay_seconds=0.3):
        self.responses = responses
        self.delay_seconds = delay_seconds
        self.idx = 0
        self.chat = self
        self.completions = self

    def create(self, **kwargs):
        if self.delay_seconds > 0:
            time.sleep(self.delay_seconds)
        resp_text = self.responses[self.idx % len(self.responses)]
        self.idx += 1

        class Message:
            content = resp_text

        class Choice:
            message = Message()

        class Usage:
            prompt_tokens = 100
            completion_tokens = 50
            total_tokens = 150

        class Response:
            def __init__(self):
                self.choices = [Choice()]
                self.usage = Usage()
                self.model = "ag/gemini-3.6-flash-low"
                self.status = "completed"

        return Response()


def test_infinite_loop_without_capabilities_dies():
    code = """
def improve_algorithm(api):
    while True:
        pass
"""
    prog = parse_optimizer_program(code, "opt_loop_1")
    runner = OptimizerProgramRunner()
    res = runner.run(prog, lambda msg: {}, limits=ProgramLimits(timeout_seconds=0.3))
    assert res["status"] == "failed"
    assert res["code"] == "optimizer_execution_timeout"


def test_infinite_loop_after_capability_dies():
    class SlowLLM:
        def generate(self, prompt: str) -> str:
            time.sleep(0.3)
            return "Reflection notes"

    def dummy_eval(src: str) -> float:
        return 1.0

    code = """
def improve_algorithm(api):
    ref = api.generate('KIND: reflection\\nAnalyze')
    while True:
        pass
"""
    prog = parse_optimizer_program(code, "opt_loop_2")
    controller = OptimizerCapabilityController(SlowLLM(), dummy_eval, CapabilityLimits(5, 5))
    runner = OptimizerProgramRunner()
    res = runner.run(prog, controller.handle, limits=ProgramLimits(timeout_seconds=0.3))
    assert res["status"] == "failed"
    assert res["code"] == "optimizer_execution_timeout"


def test_multiple_slow_valid_capabilities_succeed():
    class SlowLLM:
        def generate(self, prompt: str) -> str:
            time.sleep(0.3)
            if "reflection" in prompt:
                return "Reflection text"
            return "def select_next_node(current_node, unvisited, coordinates):\n    return unvisited[0]"

    def slow_eval(src: str) -> float:
        time.sleep(0.3)
        return 99.5

    code = """
def improve_algorithm(api):
    ref = api.generate('KIND: reflection\\nAnalyze')
    c1 = api.generate('KIND: mutate\\nOption 1')
    return api.evaluate(c1)
"""
    prog = parse_optimizer_program(code, "opt_slow_valid")
    controller = OptimizerCapabilityController(SlowLLM(), slow_eval, CapabilityLimits(5, 5))
    runner = OptimizerProgramRunner()

    # Total capability wait = 0.3 * 3 = 0.9s, exceeding timeout_seconds=0.6
    t_start = time.monotonic()
    res = runner.run(prog, controller.handle, limits=ProgramLimits(timeout_seconds=0.6))
    t_elapsed = time.monotonic() - t_start

    assert t_elapsed > 0.8
    assert res["status"] == "success"
    assert res["result"] == 99.5


def test_five_capability_boundary():
    class MultiSlowLLM:
        def generate(self, prompt: str) -> str:
            time.sleep(0.2)
            if "reflection" in prompt:
                return "Reflection"
            return "def select_next_node(current_node, unvisited, coordinates):\n    return unvisited[0]"

    def multi_eval(src: str) -> float:
        time.sleep(0.2)
        return 50.0

    code = """
def improve_algorithm(api):
    api.generate('KIND: reflection\\n1')
    c1 = api.generate('KIND: mutate\\n2')
    s1 = api.evaluate(c1)
    c2 = api.generate('KIND: mutate\\n3')
    s2 = api.evaluate(c2)
    return max(s1, s2)
"""
    prog = parse_optimizer_program(code, "opt_five_cap")
    controller = OptimizerCapabilityController(MultiSlowLLM(), multi_eval, CapabilityLimits(5, 5))
    runner = OptimizerProgramRunner()

    # Total capability wait = 5 * 0.2 = 1.0s, exceeding timeout_seconds=0.6
    t_start = time.monotonic()
    res = runner.run(prog, controller.handle, limits=ProgramLimits(timeout_seconds=0.6))
    t_elapsed = time.monotonic() - t_start

    assert t_elapsed > 0.9
    assert res["status"] == "success"
    assert res["result"] == 50.0
    assert controller.usage.generate_requests == 3
    assert controller.usage.evaluate_requests == 2


def test_provider_failure_not_classified_as_optimizer_timeout():
    class FailingLLM:
        def generate(self, prompt: str) -> str:
            time.sleep(0.2)
            raise GenerationError("provider_generation_failed")

    def dummy_eval(src: str) -> float:
        return 1.0

    code = """
def improve_algorithm(api):
    return api.generate('KIND: reflection\\nAnalyze')
"""
    prog = parse_optimizer_program(code, "opt_provider_fail")
    controller = OptimizerCapabilityController(FailingLLM(), dummy_eval, CapabilityLimits(5, 5))
    runner = OptimizerProgramRunner()

    res = runner.run(prog, controller.handle, limits=ProgramLimits(timeout_seconds=0.6))
    assert res["status"] == "failed"
    assert res["code"] == "capability_error"
    assert "provider_generation_failed" in res["message"]


def test_global_wall_timeout():
    class VerySlowLLM:
        def generate(self, prompt: str) -> str:
            time.sleep(0.5)
            return "Text"

    def dummy_eval(src: str) -> float:
        return 1.0

    code = """
def improve_algorithm(api):
    return api.generate('KIND: reflection\\nAnalyze')
"""
    prog = parse_optimizer_program(code, "opt_wall_timeout")
    controller = OptimizerCapabilityController(VerySlowLLM(), dummy_eval, CapabilityLimits(5, 5))
    runner = OptimizerProgramRunner()

    # Max wall seconds = 0.3s, while capability takes 0.5s
    res = runner.run(prog, controller.handle, limits=ProgramLimits(timeout_seconds=2.0, max_wall_seconds=0.3))
    assert res["status"] == "failed"
    assert res["code"] == "optimizer_wall_timeout"


def test_single_replicate_offline_control_with_delays(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENAI_COMPAT_API_KEY", "sk-fake-test-key")
    monkeypatch.setenv("OPENAI_COMPAT_BASE_URL", "http://172.25.16.1:20128/v1")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    raw_cfg = load_experiment_config("configs/full_moh_sanity.yaml")
    # Override timeout_seconds to 0.6s and set output_dir to tmp_path
    cfg = replace(
        raw_cfg,
        replicate_index=0,
        output_dir=tmp_path / "delayed_replicate_test",
        algorithm=replace(raw_cfg.algorithm, program_limits=replace(raw_cfg.algorithm.program_limits, timeout_seconds=0.6)),
    )

    def delayed_transport_factory():
        return DelayedFakeTransport(
            responses=[
                "Reflection analysis text.",
                "def select_next_node(current_node, unvisited, coordinates):\n    return unvisited[0]\n",
                "Option 2 reflection.",
                "def select_next_node(current_node, unvisited, coordinates):\n    return min(unvisited)\n",
            ],
            delay_seconds=0.3, # Delays per LLM call (0.3s) accumulate, exceeding timeout_seconds=0.6
        )

    manifest = run_experiment(
        cfg,
        allow_real_api=True,
        transport_factory=delayed_transport_factory,
    )

    assert manifest["status"] == "COMPLETED_VALID"
    assert manifest["best_utility"] is not None
    assert manifest["work_counts"]["inner_generate_requests"] > 0
    assert manifest["work_counts"]["inner_evaluate_requests"] > 0

    manifest_data = json.loads(Path(manifest["manifest_path"]).read_text())
    assert manifest_data["token_counts"]["combined_total_tokens"] > 0
