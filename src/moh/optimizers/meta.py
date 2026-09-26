from pydantic import ValidationError

from moh.core.models import OptimizerCandidate
from moh.core.specs import OptimizerSpec
from moh.execution.protocol import strict_json
from moh.llm.base import GenerationError
from moh.optimizers.programs import parse_optimizer_program
from moh.prompts.optimizer_generation import optimizer_prompt
from moh.prompts.optimizer_program_generation import optimizer_program_prompt


class MetaOptimizer:
    def propose(self, population, llm, candidate_id):
        response = llm.generate(optimizer_prompt(population))
        try:
            spec = OptimizerSpec.model_validate(strict_json(response))
        except (ValueError, ValidationError) as exc:
            raise GenerationError("invalid_optimizer_spec") from exc
        return OptimizerCandidate(candidate_id, spec)


class ProgramMetaOptimizer:
    def propose(self, population, llm, candidate_id, idea=None):
        prompt = optimizer_program_prompt(population, idea=idea)
        response = llm.generate(prompt)
        return parse_optimizer_program(response, candidate_id, idea=idea)
