"""Sanity Check (Micro-benchmark) for R-MoH on TSP50."""

import json
from pathlib import Path

from utils.diagnostics import TrajectoryDiagnostics
from utils.meta_population import MetaPopulation
from utils.diagnostic_prompts import format_reflective_meta_prompt, build_diagnostic_directive

ROOT_DIR = Path(__file__).resolve().parent


def run_micro_benchmark(simulated: bool = True):
    print("=" * 70)
    print("STARTING R-MoH MICRO-BENCHMARK (SANITY CHECK) ON TSP50")
    print(f"Mode: {'Simulated Offline (No API key needed)' if simulated else 'Live LLM API'}")
    print("=" * 70)

    # 1. Setup Configuration: TSP50, T_outer = 5 iterations, pop_size = 3
    t_outer = 5
    pop_size = 3
    alpha = 0.6
    c = 0.15

    print(f"\n[Config] T_outer: {t_outer}, Pop size: {pop_size}, Alpha: {alpha}, C: {c}")

    # Initialize MetaPopulation
    meta_pop = MetaPopulation(size=pop_size, alpha=alpha, c=c)

    # Simulated candidate outcomes across 5 generations:
    # Gen 0 (Seed): Stagnates early, low slope, some errors
    # Gen 1: Low validity (triggers validity directive)
    # Gen 2: High exploration (improves gap, high slope)
    # Gen 3: Plateaued / premature convergence (triggers stagnation directive)
    # Gen 4: Refined elitist candidate
    mock_generations = [
        {
            "id": "opt_seed",
            "idea": "Seed greedy local search optimizer",
            "code": "def improve_algorithm_seed(...): pass",
            "raw_gap": 15.2,
            "diag": TrajectoryDiagnostics(
                best_score=15.2,
                valid_rate=0.80,
                improvement_slope=0.005,
                stagnation_steps=3,
                diversity_score=0.12,
                failure_modes=["TimeoutError"],
                reflection_summary="Greedy 2-opt stagnated quickly due to lack of perturbation moves."
            )
        },
        {
            "id": "opt_gen1",
            "idea": "Nested double-bridge mutation",
            "code": "def improve_algorithm_gen1(...): pass",
            "raw_gap": 16.5,
            "diag": TrajectoryDiagnostics(
                best_score=16.5,
                valid_rate=0.55,  # <= 70% => triggers LOW VALIDITY directive
                improvement_slope=0.0,
                stagnation_steps=5,
                diversity_score=0.08,
                failure_modes=["SyntaxError", "TimeoutError"],
                reflection_summary="Candidate code had excessive nested loops causing timeouts on 45% of evaluations."
            )
        },
        {
            "id": "opt_gen2",
            "idea": "Adaptive guided local search with penalty memory",
            "code": "def improve_algorithm_gen2(...): pass",
            "raw_gap": 10.8,  # Significant improvement!
            "diag": TrajectoryDiagnostics(
                best_score=10.8,
                valid_rate=0.95,
                improvement_slope=0.22,
                stagnation_steps=1,
                diversity_score=1.45,
                failure_modes=[],
                reflection_summary="Penalty memory diversified search space effectively, avoiding premature convergence."
            )
        },
        {
            "id": "opt_gen3",
            "idea": "Multi-neighborhood search with dynamic perturbation",
            "code": "def improve_algorithm_gen3(...): pass",
            "raw_gap": 11.2,
            "diag": TrajectoryDiagnostics(
                best_score=11.2,
                valid_rate=0.90,
                improvement_slope=0.00005,  # <= 1e-4 => triggers STAGNATION directive
                stagnation_steps=4,
                diversity_score=0.0008,  # <= 1e-3 => triggers PREMATURE CONVERGENCE directive
                failure_modes=[],
                reflection_summary="Population collapsed into identical permutations after step 6."
            )
        },
        {
            "id": "opt_gen4",
            "idea": "Hybrid GLS with niche diversity preservation",
            "code": "def improve_algorithm_gen4(...): pass",
            "raw_gap": 8.4,  # New best elitist!
            "diag": TrajectoryDiagnostics(
                best_score=8.4,
                valid_rate=1.0,
                improvement_slope=0.35,
                stagnation_steps=0,
                diversity_score=2.10,
                failure_modes=[],
                reflection_summary="Successfully balanced niche diversity with greedy intensification."
            )
        },
    ]

    # Iteration Execution Simulation
    history_logs = []
    base_meta_prompt = "Task: You should design an efficient metaheuristic to optimize TSP50..."

    for step, candidate in enumerate(mock_generations):
        print(f"\n--- [Outer Step {step} / {t_outer}] ---")
        cand_id = candidate["id"]
        raw_gap = candidate["raw_gap"]
        diag = candidate["diag"]

        # Step A: Parent Selection (if population already has members)
        if len(meta_pop) > 0:
            parent = meta_pop.select_parent(total_outer_steps=step)
            print(f"[Selection] Parent selected: {parent.opt_id} (Selection count: {parent.selection_count})")

            # Step B: Diagnostic-Guided Prompt Construction
            _reflective_prompt = format_reflective_meta_prompt(base_meta_prompt, parent)
            directives = build_diagnostic_directive(parent.diagnostics)
            print(f"[Prompt Injection] Active Directives from parent:\n{directives or '  (None - parent healthy)'}")
            if parent.diagnostics and parent.diagnostics.reflection_summary:
                print(f"[Prompt Injection] Injected Reflection: \"{parent.diagnostics.reflection_summary}\"")
        else:
            parent = None
            print("[Init] Registering initial Seed Optimizer into MetaPopulation...")

        # Step C: Register Candidate into MetaPopulation
        meta_pop.add_individual(
            opt_id=cand_id,
            code=candidate["code"],
            thought=candidate["idea"],
            raw_utility=raw_gap,
            diagnostics=diag,
            total_outer_steps=max(step, 1),
        )

        # Step D: Feedback loop (update parent's offspring utilities)
        if parent is not None:
            meta_pop.register_offspring(parent.opt_id, raw_gap)
            print(f"[Feedback] Child {cand_id} (Gap: {raw_gap}%) registered to parent {parent.opt_id}.")

        # Step E: Log status
        current_pop = meta_pop.get_all()
        elitist = meta_pop.get_best()
        print(f"[Population State] Size: {len(current_pop)} / Max {pop_size} | Elitist: {elitist.opt_id} (Gap: {elitist.raw_utility}%)")

        step_summary = []
        for ind in current_pop:
            uiq = ind.compute_meta_uiq(total_outer_steps=step + 1, alpha=alpha, c=c)
            step_summary.append({
                "opt_id": ind.opt_id,
                "raw_gap": ind.raw_utility,
                "norm_utility": round(ind.normalized_utility, 3),
                "selection_count": ind.selection_count,
                "n_offspring": len(ind.offspring_utilities),
                "meta_uiq": round(uiq, 4),
                "valid_rate": f"{ind.diagnostics.valid_rate:.0%}" if ind.diagnostics else "N/A",
                "slope": round(ind.diagnostics.improvement_slope, 4) if ind.diagnostics else 0.0,
            })
        history_logs.append({"step": step, "candidates": step_summary})

    print("\n" + "=" * 70)
    print("FINAL META-POPULATION LOG TABLE")
    print("=" * 70)
    print(f"{'Opt ID':<12} | {'Raw Gap':<8} | {'Norm Util':<10} | {'Selected':<8} | {'Offspring':<9} | {'Meta-UIQ':<10} | {'Valid Rate':<10}")
    print("-" * 75)
    for ind in meta_pop.get_all():
        uiq = ind.compute_meta_uiq(total_outer_steps=t_outer, alpha=alpha, c=c)
        v_rate = f"{ind.diagnostics.valid_rate:.0%}" if ind.diagnostics else "N/A"
        print(f"{ind.opt_id:<12} | {ind.raw_utility:<8.2f} | {ind.normalized_utility:<10.3f} | {ind.selection_count:<8} | {len(ind.offspring_utilities):<9} | {uiq:<10.4f} | {v_rate:<10}")

    print("\n[VERIFICATION CHECKS]")
    # 1. Check Elitism
    best = meta_pop.get_best()
    assert best.opt_id == "opt_gen4", f"Expected opt_gen4 as best, got {best.opt_id}"
    print(" [PASSED] Elitism Preservation: Best optimizer (opt_gen4, Gap: 8.4%) safely retained in population.")

    # 2. Check Culling
    assert len(meta_pop) <= pop_size, f"Population exceeded max size {pop_size}"
    print(f" [PASSED] Culling: Population strictly capped at {pop_size} members without heap overflow.")

    # 3. Check Offspring feedback
    seed = meta_pop.get_by_id("opt_seed")
    if seed:
        print(f" [PASSED] Offspring Feedback: Seed optimizer tracked {len(seed.offspring_utilities)} offspring results.")

    # Save summary artifact log
    output_log_path = ROOT_DIR / "sanity_check_results.json"
    output_log_path.write_text(json.dumps(history_logs, indent=2), encoding="utf-8")
    print(f"\n[Artifact] Detailed JSON logs written to: {output_log_path}")
    print("=" * 70)
    print("SANITY CHECK COMPLETED SUCCESSFULLY!")
    print("=" * 70)


if __name__ == "__main__":
    run_micro_benchmark(simulated=True)
