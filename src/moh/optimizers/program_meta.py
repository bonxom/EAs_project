"""Self-improvement of optimizer programs through bounded nested worker search."""
import hashlib
import math
from dataclasses import asdict

from moh.core.program_population import ProgramPopulation
from moh.core.programs import (
    OptimizerProgram,
    ProgramEvaluation,
    ProgramRunResult,
    ScoredProgram,
    TaskOutcome,
)
from moh.core.seeds import derive_seed
from moh.execution.optimizer_protocol import number
from moh.execution.optimizer_runner import OptimizerRequest
from moh.execution.process import CandidateFailure, Deadline
from moh.execution.program_callbacks import InvocationDispatcher
from moh.optimizers.seeds import BASIC_SOURCE
from moh.problems.tsp_gls.evaluation import weighted_gap
from moh.prompts.program_optimizer import OPTIMIZER_FORMAT


def _hash(source):
    return hashlib.sha256(source.encode('utf-8')).hexdigest()


def _population_ids(populations):
    return {task: [item.id for item in population.members]
            for task, population in populations.items()}


def _program_payload(program, invocation_id):
    return {'id': program.id, 'source_code': program.source_code,
            'source_hash': _hash(program.source_code), 'idea': program.idea,
            'invocation_id': invocation_id}


class ProgramEvaluator:
    def __init__(self, tasks, weights, inner, llm_factory, budget, emit):
        self.tasks = tuple(tasks)
        if not self.tasks or len({task.id for task in self.tasks}) != len(self.tasks):
            raise ValueError('evaluation requires nonempty tasks with distinct IDs')
        self.weights = tuple(weights) if weights is not None else tuple(
            float(task.size) for task in self.tasks)
        if len(self.weights) != len(self.tasks) or any(
                not number(weight) or weight <= 0 for weight in self.weights):
            raise ValueError('weights must match tasks and be finite and positive')
        self.inner, self.llm_factory = inner, llm_factory
        self.budget, self.emit = budget, emit

    def evaluate(self, program, populations, *, root_seed, invocation_id, deadline, scope):
        derive_seed(root_seed)
        if set(populations) != {task.id for task in self.tasks}:
            raise ValueError('populations must match evaluation tasks')
        before = self.budget.counts
        staged = dict(populations)
        outcomes = []
        error = None
        payload = _program_payload(program, invocation_id)
        self.emit('optimizer_generated', payload)
        for task in self.tasks:
            try:
                deadline.check()
                scope.check()
                llm = self.llm_factory(('inner', invocation_id, task.id))
                result = self.inner.search(
                    program, task, staged[task.id], llm, root_seed=root_seed,
                    invocation_id=f'{invocation_id}-{task.id}', deadline=deadline, scope=scope)
            except CandidateFailure as exc:
                error = exc.code
                outcomes.append(TaskOutcome(task.id, None))
                break
            if result.status != 'success':
                error = result.error or 'inner_failed'
                outcomes.append(TaskOutcome(task.id, None))
                break
            outcomes.append(TaskOutcome(task.id, result.selected))
            staged[task.id] = result.population
        evaluation = ProgramEvaluation(
            program.id, 'failed' if error else 'success',
            None if error else weighted_gap(tuple(outcomes), self.weights), tuple(outcomes),
            error, self.budget.delta(before))
        committed = dict(populations) if error else staged
        self.emit('optimizer_evaluated', {
            **payload, 'evaluation': asdict(evaluation), 'status': evaluation.status,
            'counts': asdict(evaluation.counts),
            'population_before': _population_ids(populations),
            'population_after': _population_ids(committed)})
        return evaluation, committed


class ProgramMeta:
    def __init__(self, optimizer_runner, evaluator, budget, emit):
        self.optimizer_runner, self.evaluator = optimizer_runner, evaluator
        self.budget, self.emit = budget, emit

    def search(self, seeds, populations, meta_llm, *, iterations, capacity,
               root_seed, deadline, scope):
        if type(iterations) is not int or iterations < 0:
            raise ValueError('iterations must be a nonnegative integer')
        derive_seed(root_seed)
        population = ProgramPopulation(capacity)
        task_populations = dict(populations)
        seed_cache = {}
        successful_seeds = []
        for index, program in enumerate(seeds):
            if program.source_code in seed_cache:
                self.emit('cache_hit', {'kind': 'optimizer', 'candidate_id': program.id,
                                       'source_hash': _hash(program.source_code),
                                       'invocation_id': f'seed-{index}'})
                continue
            evaluation, staged = self.evaluator.evaluate(
                program, task_populations, root_seed=root_seed,
                invocation_id=f'seed-{index}', deadline=deadline, scope=scope)
            seed_cache[program.source_code] = evaluation
            if evaluation.status == 'success':
                item = ScoredProgram(program, evaluation)
                successful_seeds.append(item)
                population = population.add(item)
                task_populations = staged
        if not population.members:
            self.emit('search_checkpoint', {
                'label': 'seeds', 'populations': {**task_populations, 'meta-optimizer': population},
                'active': None, 'winner': None})
            return ProgramRunResult('failed', None, None, (), task_populations, (),
                                    self.budget.counts)
        active = next((item for item in successful_seeds if item.source_code == BASIC_SOURCE),
                      population.best())
        self.emit('search_checkpoint', {
            'label': 'seeds', 'populations': {**task_populations, 'meta-optimizer': population},
            'active': active, 'winner': population.best()})
        for round_index in range(iterations):
            try:
                deadline.check()
                scope.check()
            except CandidateFailure:
                break
            if (self.budget.counts.llm_calls >= self.budget.max_llm_calls
                    or self.budget.counts.heuristic_evaluations >= self.budget.max_heuristic_evaluations):
                break
            before = self.budget.counts
            original_population, original_tasks = population, task_populations
            invocation_id = f'meta-round-{round_index}'
            self.emit('meta_round_started', {'round': round_index,
                                             'invocation_id': invocation_id,
                                             'active_id': active.id,
                                             'source_hash': _hash(active.source_code)})
            staged_population, staged_tasks = population, dict(task_populations)
            known = {item.source_code: item for item in (*population.members, active)}
            cache = {source: item.utility for source, item in known.items()}
            sequence = 0
            error = None
            selected = None
            invocation = Deadline.after(self.optimizer_runner.limits.timeout_seconds,
                                        parent=deadline)
            try:
                with self.optimizer_runner.supervisor.scope(invocation, parent_scope=scope) as child:
                    def evaluate(source, idea, callback_deadline, *, cache=cache, known=known,
                                 invocation_id=invocation_id, round_index=round_index, child=child):
                        nonlocal sequence, staged_population, staged_tasks
                        callback_deadline.check()
                        child.check()
                        if source in cache:
                            self.emit('cache_hit', {'kind': 'optimizer',
                                                   'invocation_id': invocation_id,
                                                   'source_hash': _hash(source)})
                            return cache[source]
                        sequence += 1
                        program = OptimizerProgram(f'o-{round_index:06d}-{sequence:06d}', source, idea)
                        evaluation, next_tasks = self.evaluator.evaluate(
                            program, staged_tasks, root_seed=root_seed,
                            invocation_id=f'{invocation_id}-candidate-{sequence}',
                            deadline=callback_deadline, scope=child)
                        value = 1e6
                        if evaluation.status == 'success':
                            item = ScoredProgram(program, evaluation)
                            known[source] = item
                            staged_population = staged_population.add(item)
                            staged_tasks = next_tasks
                            value = item.utility
                        cache[source] = value
                        return value

                    dispatcher = InvocationDispatcher('meta-optimizer', meta_llm,
                                                      self.budget, evaluate,
                                                      max_batch_size=self.optimizer_runner.limits.batch_size)
                    request = OptimizerRequest(
                        active.candidate, population.snapshot('meta-optimizer'), 'meta-optimizer',
                        OPTIMIZER_FORMAT, derive_seed(root_seed, 'outer', round_index),
                        meta_llm.batch_size)
                    self.emit('optimizer_worker_started', {
                        'id': active.id, 'invocation_id': invocation_id, 'level': 'outer',
                        'task_id': 'meta-optimizer', 'source_hash': _hash(active.source_code)})
                    worker_result = self.optimizer_runner.run(
                        request, dispatcher.dispatch, deadline=invocation, scope=child)
                    if worker_result.status == 'success':
                        selected = known.get(worker_result.source_code)
                        if selected is None or not math.isclose(
                                selected.utility, worker_result.claimed_utility,
                                rel_tol=1e-10, abs_tol=1e-10):
                            error = 'unverified_result'
                    else:
                        error = worker_result.error
            except CandidateFailure as exc:
                error = exc.code
            if error is None and selected is not None:
                population, task_populations = staged_population, staged_tasks
                active = selected
                self.emit('population_updated', {'level': 'outer', 'invocation_id': invocation_id,
                                                 'ids': [item.id for item in population.members]})
            else:
                error = error or 'unverified_result'
                population, task_populations = original_population, original_tasks
                active = population.best()
                self.emit('population_rollback', {'level': 'outer',
                                                  'invocation_id': invocation_id, 'error': error,
                                                  'ids': [item.id for item in population.members]})
            self.emit('active_optimizer_changed', {
                'round': round_index, 'invocation_id': invocation_id, 'optimizer_id': active.id,
                'source_hash': _hash(active.source_code), 'status': 'failed' if error else 'success'})
            self.emit('meta_round_finished', {
                'round': round_index, 'invocation_id': invocation_id,
                'active_id': active.id, 'source_hash': _hash(active.source_code),
                'status': 'failed' if error else 'success', 'error': error,
                'counts': asdict(self.budget.delta(before)),
                'population_before': _population_ids(original_tasks),
                'population_after': _population_ids(task_populations)})
            self.emit('search_checkpoint', {
                'label': f'round-{round_index:06d}',
                'populations': {**task_populations, 'meta-optimizer': population},
                'active': active, 'winner': population.best()})
        # Local invocation failures may fall back to verified programs, but the
        # deadline supplied to the whole search defines whether the run completed.
        if not deadline.remaining():
            return ProgramRunResult('failed', None, None, (), task_populations, (),
                                    self.budget.counts)
        return ProgramRunResult('success', population.best(), active, population.members,
                                task_populations, (), self.budget.counts)
