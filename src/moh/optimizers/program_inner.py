"""Worker optimizer search with authoritative parent fitness and atomic commit."""
import hashlib
import json
import math
from dataclasses import asdict, dataclass

from moh.core.models import Heuristic, WorkCounts
from moh.core.program_population import ProgramPopulation
from moh.core.programs import ScoredProgram
from moh.core.seeds import derive_seed
from moh.execution.optimizer_runner import OptimizerRequest
from moh.execution.process import CandidateFailure, Deadline
from moh.execution.program_callbacks import InvocationDispatcher
from moh.llm.base import GenerationError
from moh.optimizers.helpers import extract_code, extract_idea
from moh.problems.tsp_gls.baselines import (
    IDENTITY_SOURCE,
    PENALTY_SOURCE,
    UPSTREAM_BASELINE_SOURCE,
)
from moh.prompts.gls_heuristic import (
    HEURISTIC_FORMAT,
    seed_direction_prompt,
    seed_heuristic_prompt,
)


@dataclass(frozen=True)
class InnerProgramResult:
    status: str
    selected: ScoredProgram | None
    population: ProgramPopulation
    evaluated: tuple[ScoredProgram, ...]
    error: str | None
    counts: WorkCounts


class _Assessment:
    def __init__(self, task, population, runner, budget, emit, root_seed, invocation_id, scope, optimizer_id=None):
        self.task, self.population, self.runner = task, population, runner
        self.budget, self.emit, self.root_seed = budget, emit, root_seed
        self.invocation_id, self.scope = invocation_id, scope
        self.optimizer_id = optimizer_id
        self.known = {item.source_code: item for item in population.members}
        self.cache = {source: item.utility for source, item in self.known.items()}
        self.evaluated = []
        self.sequence = 0

    def evaluate(self, source, idea, deadline):
        deadline.check()
        self.scope.check()
        if source in self.cache:
            self.emit('cache_hit', {'invocation_id': self.invocation_id,
                                   'task_id': self.task.id,
                                   'source_hash': hashlib.sha256(source.encode()).hexdigest()})
            return self.cache[source]
        self.budget.charge_evaluation()
        self.sequence += 1
        # Hash the trusted invocation label to keep safe IDs within 128 bytes.
        prefix = hashlib.sha256(self.invocation_id.encode()).hexdigest()[:16]
        candidate = Heuristic(f'h-{prefix}-{self.sequence:06d}', source, idea)
        payload = {'invocation_id': self.invocation_id, 'task_id': self.task.id,
                   'id': candidate.id, 'source_code': source, 'idea': idea,
                   'optimizer_id': self.optimizer_id,
                   'source_hash': hashlib.sha256(source.encode()).hexdigest(),
                   'kind': 'heuristic'}
        self.emit('heuristic_generated', payload)
        self.emit('heuristic_worker_started', {key: value for key, value in payload.items()
                                     if key not in ('source_code', 'idea')})
        evaluation = self.runner.evaluate(candidate, self.task, split='validation',
                                          root_seed=self.root_seed,
                                          deadline=deadline, scope=self.scope)
        self.budget.charge_instances(evaluation.counts.instance_attempts)
        self.emit('heuristic_evaluated', {**payload, 'evaluation': asdict(evaluation)})
        value = 1e6
        if evaluation.status == 'success':
            item = ScoredProgram(candidate, evaluation)
            self.known[source] = item
            self.evaluated.append(item)
            self.population = self.population.add(item)
            value = item.utility
        self.cache[source] = value
        return value


class ProgramInner:
    def __init__(self, optimizer_runner, gls_runner, budget, emit):
        self.optimizer_runner, self.gls_runner = optimizer_runner, gls_runner
        self.budget, self.emit = budget, emit

    def search(self, program, task, population, llm, *, root_seed, invocation_id,
               deadline, scope):
        seed = derive_seed(root_seed, 'inner', task.id, invocation_id)
        before = self.budget.counts
        invocation = Deadline.after(self.optimizer_runner.limits.timeout_seconds,
                                    parent=deadline)
        with self.optimizer_runner.supervisor.scope(invocation, parent_scope=scope) as child:
            assessment = _Assessment(task, population, self.gls_runner, self.budget,
                                     self.emit, root_seed, invocation_id, child, program.id)
            dispatcher = InvocationDispatcher(task.id, llm, self.budget, assessment.evaluate,
                                              max_batch_size=self.optimizer_runner.limits.batch_size)
            request = OptimizerRequest(program, population.snapshot(task.id), task.id,
                                       HEURISTIC_FORMAT, seed, llm.batch_size)
            self.emit('optimizer_worker_started', {'invocation_id': invocation_id,
                                        'id': program.id, 'task_id': task.id,
                                        'level': 'inner',
                                        'source_hash': hashlib.sha256(
                                            program.source_code.encode()).hexdigest()})
            result = self.optimizer_runner.run(request, dispatcher.dispatch,
                                               deadline=invocation, scope=child)
            selected = assessment.known.get(result.source_code) if result.status == 'success' else None
            verified = selected is not None and math.isclose(
                selected.utility, result.claimed_utility, rel_tol=1e-10, abs_tol=1e-10)
            error = None if verified else (result.error or 'unverified_result')
            self.emit('population_updated' if verified else 'population_rollback',
                      {'invocation_id': invocation_id, 'task_id': task.id,
                       'optimizer_id': program.id, 'error': error, 'level': 'inner',
                       'ids': [item.id for item in (assessment.population if verified
                                                  else population).members]})
            return InnerProgramResult('success' if verified else 'failed',
                                      selected if verified else None,
                                      assessment.population if verified else population,
                                      tuple(assessment.evaluated), error, self.budget.delta(before))


def initialize_task_population(task, gls_runner, llm, budget, *, capacity,
                               seed_attempts, threshold, root_seed, deadline, scope, emit):
    if type(seed_attempts) is not int or seed_attempts < 0:
        raise ValueError('seed_attempts must be nonnegative')
    if threshold is not None and (type(threshold) not in (int, float)
                                  or not math.isfinite(threshold) or threshold < 0):
        raise ValueError('threshold must be finite and nonnegative')
    derive_seed(root_seed)
    assessment = _Assessment(task, ProgramPopulation(capacity), gls_runner, budget, emit,
                             root_seed, f'initialize-{task.id}', scope)
    for idea, source in [('identity', IDENTITY_SOURCE), ('penalty', PENALTY_SOURCE),
                         ('upstream', UPSTREAM_BASELINE_SOURCE)]:
        try:
            assessment.evaluate(source, idea, deadline)
        except CandidateFailure:
            return assessment.population
    baseline_population = assessment.population
    dispatcher = InvocationDispatcher(task.id, llm, budget, assessment.evaluate,
                                      max_batch_size=llm.batch_size)
    for _ in range(seed_attempts):
        try:
            request = seed_direction_prompt(task.id, [item.idea for item in assessment.evaluated])
            response = dispatcher.dispatch('llm_prompt', {
                'expertise': request.expertise, 'message': request.message,
                'temperature': request.temperature}, deadline)
            try:
                directions = json.loads(extract_code(response))
            except (ValueError, RecursionError) as exc:
                raise GenerationError('invalid_directions') from exc
            insights = directions.get('insights') if isinstance(directions, dict) else None
            if (not isinstance(insights, list) or not insights
                    or not isinstance(insights[0], str) or not insights[0].strip()):
                raise GenerationError('invalid_directions')
            try:
                request = seed_heuristic_prompt(task.id, insights[0])
            except (ValueError, UnicodeError) as exc:
                raise GenerationError('invalid_directions') from exc
            response = dispatcher.dispatch('llm_prompt', {
                'expertise': request.expertise, 'message': request.message,
                'temperature': request.temperature}, deadline)
            source, idea = extract_code(response), extract_idea(response)
            assessment.evaluate(source, idea, deadline)
        except CandidateFailure as exc:
            if exc.code in ('llm_budget', 'evaluation_budget', 'timeout'):
                break
        except (GenerationError, json.JSONDecodeError, KeyError):
            continue
    population = baseline_population
    for item in assessment.evaluated:
        if threshold is None or item.utility <= threshold:
            population = population.add(item)
    return population
