# R-MoH: Reflective & Uncertainty-Guided Meta-Optimization of Heuristics Design Spec

## 1. Intent and Scope

This document specifies the design for **R-MoH** (Reflective & Uncertainty-Guided Meta-Optimization of Heuristics) implemented directly on the existing MoH codebase.

The goal is to eliminate three fundamental bottlenecks of baseline MoH without breaking backwards compatibility:
1. **Inner Loop Blindness:** Baseline MoH discards all intermediate generation history of a heuristic optimizer $\tilde{I}_m$, returning only the best fitness $U(\tilde{I}_m)$ and code $H^*$. R-MoH captures a typed **Trajectory Diagnostics** packet ($\mathcal{T}_m$).
2. **Outer Loop Greedy Culling:** Baseline MoH evaluates optimizers statically and immediately discards lower-fitness candidates, ignoring exploration potential or offspring quality. R-MoH introduces **Meta-UIQ**, balancing exploitation of parent and offspring utility with an upper confidence bound (UCB) exploration bonus.
3. **Information-Blind Meta-Prompts:** Baseline Meta-Optimizer prompts recombine parent ideas without knowing why an optimizer failed or stagnated. R-MoH injects **Diagnostic-Guided Directives** and a concise LLM reflection summary into self-invocation prompts.

### Dual-Mode Execution via Feature Flag
The entire enhancement is gated by `cfg.rmoh.enabled`:
- When `cfg.rmoh.enabled = False`: MoH runs strictly as the original baseline (identical selection, prompt, and evaluation paths).
- When `cfg.rmoh.enabled = True`: R-MoH hooks are activated in the inner loop, outer population manager, and self-invocation prompt builder.
This ensures 100% reproducible baseline comparison and ablation experiments.

---

## 2. Global Constraints (from AGENTS.md)

- Python 3.12.
- Use `uv`.
- All experiments must be reproducible with seeds.
- Unit tests must never call external LLM APIs (use `FakeLLM` mandatory for tests).
- Never execute generated code in the main process (use `GeneratedImprover` multiprocessing sandbox with timeouts).
- Failure of one generated candidate must not crash an experiment.
- Keep core abstractions provider-independent.
- Do not hardcode OpenAI-specific logic outside `utils/llm_client/openai.py`.
- Validation checks before completion: `uv run ruff check .` and `uv run pytest -q`.

---

## 3. Architecture & The Three Hooks

```text
+-----------------------------------------------------------------------------------------+
|                                    OUTER LOOP                                           |
|                                                                                         |
|  [Meta-Population Manager] <---------------------------------------------+              |
|  - Tracks genealogy: n(I), offspring_utilities                           |              |
|  - Computes Meta-UIQ: exploitation (self + offspring) + UCB exploration  |              |
|  - Selection by Meta-UIQ; Culling by worst Meta-UIQ with Elitism         |              |
|                          |                                               |              |
|                          v Parent I_parent                               | Offspring    |
|  [Diagnostic-Guided Self-Invocation Prompt]                              | child_utility|
|  - Injects parent diagnostics: low validity, stagnation, diversity       |              |
|  - Injects reflection summary (<= 60 words)                              |              |
|                          |                                               |              |
|                          v Generates new optimizer candidate I_child     |              |
+--------------------------|-----------------------------------------------|--------------+
                           |
                           v
+--------------------------------------------------------------------------|--------------+
|                                    INNER LOOP                            |              |
|                                                                          |              |
|  [GeneratedImprover Subprocess Sandbox]                                  |              |
|  - Runs I_child to search heuristics H for subtasks                      |              |
|  - Parent interceptor collects:                                          |              |
|      * ast validity, syntax/timeout errors                               |              |
|      * score progression over evaluations                                |              |
|  - At completion: computes TrajectoryDiagnostics (valid_rate, slope,     |              |
|    stagnation_steps, diversity_score) + generates reflection_summary ----+              |
+-----------------------------------------------------------------------------------------+
```

---

## 4. Module Specifications

### Module 1: Trajectory Diagnostics (`MoH/utils/diagnostics.py`)
Dataclass `TrajectoryDiagnostics`:
```python
@dataclass
class TrajectoryDiagnostics:
    best_score: float                  # Best score achieved in inner loop
    valid_rate: float                  # Valid code fraction (0.0 to 1.0)
    improvement_slope: float           # Slope of fitness improvement over evaluations
    stagnation_steps: int              # Consecutive steps without improvement at the end
    diversity_score: float             # Standard deviation of population scores
    failure_modes: list[str]           # Frequent error categories (Syntax, Timeout, Logic)
    reflection_summary: str = ""       # Max 60-word reflection by LLM
```

- **Collector:** Hooked into the utility evaluation callback passed to `GeneratedImprover`. Every heuristic evaluated by child code is recorded (code, validity, score, error string).
- **Reflection:** A single lightweight LLM prompt at the end of `get_improver` (invoked only if `cfg.rmoh.reflection` is enabled). It takes the optimizer code, the 2 best heuristics, and the 2 worst/failed heuristics, returning $\le 60$ words on why the optimizer stagnated.

### Module 2: Meta-UIQ Outer Population Manager (`MoH/utils/meta_population.py`)
Class `OptimizerIndividual`:
```python
class OptimizerIndividual:
    opt_id: str
    code: str
    thought: str
    raw_utility: float                 # Original gap / objective (lower is better in TSP)
    normalized_utility: float          # Normalized to [0, 1] (higher is better)
    diagnostics: TrajectoryDiagnostics | None
    selection_count: int = 0           # Number of times selected as parent
    offspring_utilities: list[float]   # Normalized utilities of offspring optimizers
```

**Meta-UIQ Formula:**
$$\text{Meta-UIQ}(I) = \alpha \cdot U(I) + (1 - \alpha) \cdot \bar{Q}_{\text{offspring}}(I) + c \cdot \sqrt{\frac{\ln(\max(T_{\text{outer}}, 1) + 1)}{n(I) + 1}}$$
- If no offspring yet: $\bar{Q}_{\text{offspring}} = 0.8 \cdot U(I) + 0.2 \cdot \rho_{\text{valid}}$.
- Min-Max normalization converts TSP/CVRP gaps to $[0, 1]$ where higher is better.
- **Selection:** Roulette-wheel or softmax proportional to Meta-UIQ, or Top-K.
- **Culling:** Prunes the individual with lowest Meta-UIQ, but never prunes the global elitist (individual with highest raw utility).

### Module 3: Diagnostic-Guided Self-Invocation (`MoH/utils/diagnostic_prompts.py`)
Generates actionable directives based on parent diagnostics:
- Low validity ($\rho_{\text{valid}} \le 0.70$): Enforce strict syntax checks, avoid nested loops.
- Stagnation ($stagnation \ge 3$ or $|\Delta_{\text{imp}}| \le 10^{-4}$): Inject adaptive mutation or perturbation.
- Low diversity ($diversity \le 10^{-3}$): Maintain niche diversity and diverse selection.
- Appends `reflection_summary` to guide the Meta-Optimizer.

---

## 5. Configuration & Baseline Preservation

In `MoH/cfg/config.yaml`:
```yaml
rmoh:
  enabled: false
  alpha: 0.6
  c: 0.15
  reflection: true
  summary_max_words: 60
```
When `rmoh.enabled == False`, `moh.py` runs the exact baseline code without any behavior alteration.
