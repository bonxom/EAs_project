import numpy as np
import json
import random
import sys
import time

from .hifo_interface_EC import InterfaceEC
from ...utils.progress import sparkline, fmt_duration, fmt_gap
from ...utils.tsp_reference import get_baseline, gap_percent

class HiFo:

    def __init__(self, paras, problem, select, manage, **kwargs):
        self.prob = problem
        self.select = select
        self.manage = manage
        
        self.use_local_llm = paras.llm_use_local
        self.llm_local_url = paras.llm_local_url
        self.api_endpoint = paras.llm_api_endpoint
        self.api_key = paras.llm_api_key
        self.llm_model = paras.llm_model

        self.pop_size = paras.ec_pop_size
        self.n_pop = paras.ec_n_pop

        self.operators = paras.ec_operators
        self.operator_weights = paras.ec_operator_weights
        if paras.ec_m > self.pop_size or paras.ec_m == 1:
            print("m should not be larger than pop size or smaller than 2, adjust it to m=2")
            paras.ec_m = 2
        self.m = paras.ec_m

        self.debug_mode = paras.exp_debug_mode
        self.ndelay = 1

        self.use_seed = paras.exp_use_seed
        self.seed_path = paras.exp_seed_path
        self.load_pop = paras.exp_use_continue
        self.load_pop_path = paras.exp_continue_path
        self.load_pop_id = paras.exp_continue_id

        self.output_path = paras.exp_output_path
        self.exp_n_proc = paras.exp_n_proc
        self.timeout = paras.eva_timeout
        self.use_numba = paras.eva_numba_decorator

        self.use_hifo_prompt = kwargs.get('use_hifo_prompt', True)

        self._reference_cost = None
        if type(problem).__name__ == 'TSPCONST':
            instance_data = getattr(problem, 'instance_data', None)
            if instance_data:
                self._reference_cost = get_baseline(instance_data, problem.problem_size)
        
        print("- HiFo parameters loaded -")
        
        if self.use_hifo_prompt:
            self.hifo_prompt_log_path = self.output_path + "/results/hifo_prompt_log.json"
            print("- HiFo-Prompt enabled: Insight pool and Evolutionary Navigator will guide the evolution -")

        random.seed(2024)

    def add2pop(self, population, offspring):
        for off in offspring:
            for ind in population:
                if ind['objective'] == off['objective']:
                    if (self.debug_mode):
                        print("duplicated result, retrying ... ")
            population.append(off)

    def run(self):
        print("- Evolution Start -")
        time_start = time.time()

        interface_prob = self.prob

        interface_ec = InterfaceEC(
            self.pop_size, self.m, self.api_endpoint, self.api_key, self.llm_model, 
            self.use_local_llm, self.llm_local_url, self.debug_mode, interface_prob, 
            select=self.select, n_p=self.exp_n_proc, timeout=self.timeout, use_numba=self.use_numba
        )

        population = []
        if self.use_seed:
            with open(self.seed_path) as file:
                data = json.load(file)
            population = interface_ec.population_generation_seed(data, self.exp_n_proc)
            filename = self.output_path + "/results/pops/population_generation_0.json"
            with open(filename, 'w') as f:
                json.dump(population, f, indent=5)
            n_start = 0
        else:
            if self.load_pop:
                print("load initial population from " + self.load_pop_path)
                with open(self.load_pop_path) as file:
                    data = json.load(file)
                for individual in data:
                    population.append(individual)
                print("initial population has been loaded!")
                n_start = self.load_pop_id
            else:
                print("creating initial population:")
                population = interface_ec.population_generation()
                population = self.manage.population_management(population, self.pop_size)
                
                print(f"Pop initial: ")
                for off in population:
                    print(" Obj: ", off['objective'], end="|")
                print()
                print("initial population has been created!")
                filename = self.output_path + "/results/pops/population_generation_0.json"
                with open(filename, 'w') as f:
                    json.dump(population, f, indent=5)
                n_start = 0

        hifo_prompt_logs = []
        n_op = len(self.operators)
        best_history = []
        self._print_header()

        # Freeze the operator status line unless stdout is an interactive terminal,
        # so redirected logs do not end up full of carriage-return noise.
        self._live_status = sys.stdout.isatty()

        for pop in range(n_start, self.n_pop):
            iter_start = time.time()

            for i in range(n_op):
                op = self.operators[i]
                if self._live_status:
                    print(f"  op {op} [{i + 1}/{n_op}] ...".ljust(30), end="\r", flush=True)
                op_w = self.operator_weights[i]
                if (np.random.rand() < op_w):
                    parents, offsprings = interface_ec.get_algorithm(population, op)
                self.add2pop(population, offsprings)
                size_act = min(len(population), self.pop_size)
                population = self.manage.population_management(population, size_act)

            if self._live_status:
                print(f"  op {n_op} operators done".ljust(30), end="\r", flush=True)

            filename = self.output_path + "/results/pops/population_generation_" + str(pop + 1) + ".json"
            with open(filename, 'w') as f:
                json.dump(population, f, indent=5)

            filename = self.output_path + "/results/pops_best/population_generation_" + str(pop + 1) + ".json"
            with open(filename, 'w') as f:
                json.dump(population[0] if population else None, f, indent=5)

            best = population[0]["objective"] if population else None
            if best is not None:
                best_history.append(best)
            diversity = interface_ec.diversity_history[-1] if interface_ec.diversity_history else None
            gap = gap_percent(best, self._reference_cost)

            if not population:
                print("Warning: population is empty - no valid algorithm was generated or "
                      "evaluated in this generation. Enable exp_debug_mode for details.")

            self._print_iteration(pop + 1, best, gap, diversity, population,
                                  time.time() - iter_start,
                                  time.time() - time_start, best_history)

            if self.use_hifo_prompt:
                hifo_prompt_log = {
                    "generation": pop + 1,
                    "timestamp": time.time(),
                    "best_fitness": best,
                    "gap_percent": round(gap, 4) if gap is not None else None,
                    "objective_values": [ind["objective"] for ind in population],
                    "diversity": diversity,
                    "elapsed_s": round(time.time() - time_start, 1),
                    "current_insight_count": len(interface_ec.insight_pool.tips),
                    "top_insights": list(interface_ec.insight_pool.tips)[:2],
                    "navigator_guidance": interface_ec.navigator.last_guidance
                }
                hifo_prompt_logs.append(hifo_prompt_log)

                # Rewritten periodically so progress is readable while running,
                # without rewriting the whole file on every single generation.
                if (pop + 1) % 5 == 0 or pop + 1 == self.n_pop:
                    with open(self.hifo_prompt_log_path, 'w') as f:
                        json.dump(hifo_prompt_logs, f, indent=2)

    def _print_header(self):
        if self._reference_cost is not None:
            print(f"Reference (nearest-neighbour) cost: {self._reference_cost:.5f}")
            print("Gap < 0 means the evolved heuristic beats that reference.")
        print(f"{'gen':>4} {'best':>9} {'gap':>8} {'div':>6} {'n':>3} "
              f"{'iter':>7} {'total':>7} {'eta':>7}  progress (best, taller = better)")

    def _print_iteration(self, gen, best, gap, diversity, population,
                         iter_time, total_time, best_history):
        remaining = self.n_pop - gen
        eta = total_time / gen * remaining if gen else None
        best_str = "   --   " if best is None else f"{best:9.5f}"
        div_str = " -- " if diversity is None else f"{diversity:5.2f}"
        print(f"{gen:>4} {best_str} {fmt_gap(gap)} {div_str} {len(population):>3} "
              f"{fmt_duration(iter_time):>7} {fmt_duration(total_time):>7} {fmt_duration(eta):>7}  "
              f"{sparkline(best_history)}")
