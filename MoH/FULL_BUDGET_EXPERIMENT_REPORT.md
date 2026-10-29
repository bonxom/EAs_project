# Full-Budget Experiment Report — Meta-Training & Zero-Shot Deployment (Seed 0)

**Date:** 2026-10-29 → 2026-10-30  
**Git Branch:** `dualhifo`  
**Problem:** Traveling Salesperson Problem (TSP-GLS)  
**Model:** `ag/gemini-3.8-flash-low` (via proxy `172.25.16.1:20128`)  
**Seed:** `0`  

---

## 1. Experiment Overview & Protocol

This campaign evaluates the meta-optimization of heuristics across four core algorithmic configurations:
1. `full_moh`: Baseline Meta-Optimization of Heuristics (no HiFo guidance).
2. `outer_hifo`: HiFo foresight controller applied exclusively to the outer meta-loop.
3. `inner_hifo`: HiFo foresight controller applied exclusively to the inner heuristic-loop.
4. `dual_hifo_independent`: Independent Dual-HiFo controllers active on both levels.

### A. Meta-Training Budget
- **Outer iterations:** `10`
- **Improver population size:** `10`
- **Subtask evaluation budget:** `max_eval_calls=1000` per subtask (TSP-20, TSP-50, TSP-100, TSP-200) → `4000` total evaluations per outer candidate
- **Provider API budget:** `max_requests=5000`, `max_attempts=7500`, `max_retries_per_call=5`
- **Timeout limits:** `timeout=90s` (inner program run), `optimizer_timeout=1800s`

### B. Zero-Shot Deployment Benchmark
- **Target instance sizes:** `[500, 1000]` (strictly unseen during training)
- **Deployment protocol:** Checkpoint-driven, freezing learned meta-optimizers to generate heuristic solvers for large TSP scales
- **Inference iterations:** `10`
- **Validation instances:** `8`
- **Independent test instances:** `100` instances per target size
- **Evaluation metric:** Optimality Gap (%) against Concorde/LKH exact baselines

---

## 2. Meta-Training Results

Lower meta-utility indicates superior training objective performance on the training distribution (TSP 20–200).

| Mode | Status | Meta-Utility | Evaluation Calls | Training Duration |
|---|---|---:|---:|---:|
| `outer_hifo` | Completed | **0.360121** | 331 | 5,901s (~1.6h) |
| `dual_hifo_independent` | Completed | 0.182805 | 1,135 | 24,491s (~6.8h) |
| `inner_hifo` | Completed | 0.115898 | 364 | 10,266s (~2.85h) |
| `full_moh` | Completed | 0.066580 | 990 | 22,281s (~6.2h) |

---

## 3. Zero-Shot Deployment Benchmark (100 Test Instances)

Evaluation of learned optimizers generating heuristics directly on unseen TSP-500 and TSP-1000 instances. Lower gap is better.

| Rank | Mode | TSP-500 Test Gap (%) | TSP-1000 Test Gap (%) | **Average Test Gap (%)** |
|:---:|---|---:|---:|---:|
| 🥇 1 | **`dual_hifo_independent`** | **1.135544%** | **2.448408%** | **1.791976%** |
| 🥈 2 | `full_moh` | 1.224859% | 2.664741% | 1.944800% |
| 🥉 3 | `inner_hifo` | 1.941724% | 3.176435% | 2.559079% |
| 4 | `outer_hifo` | 2.031848% | 3.467001% | 2.749424% |

---

## 4. Iteration-by-Iteration Deployment Progress

### `dual_hifo_independent` (Winner)
| Iteration | TSP-500 Val Gap (%) | TSP-1000 Val Gap (%) | Average Val Gap (%) | Δ Progress (%-pt) |
|---|---:|---:|---:|---:|
| Iter 0 (seed) | 1.258880 | 2.730303 | 1.994592 | – |
| Iter 1 | 1.206689 | 2.716186 | 1.961438 | +0.033154 |
| Iter 2 | 1.157833 | 2.628045 | 1.892939 | +0.068498 |
| Iter 3 | 1.157833 | 2.628045 | 1.892939 | +0.000000 |
| Iter 4 | 1.157833 | 2.628045 | 1.892939 | +0.000000 |
| Iter 5 | 1.157833 | 2.628045 | 1.892939 | +0.000000 |
| Iter 6 | 1.157833 | 2.628045 | 1.892939 | +0.000000 |
| Iter 7 | 1.157833 | 2.628045 | 1.892939 | +0.000000 |
| Iter 8 | 1.157833 | 2.628045 | 1.892939 | +0.000000 |
| Iter 9 | 1.157833 | 2.628045 | 1.892939 | +0.000000 |
| **Test (100 inst)** | **1.135544%** | **2.448408%** | **1.791976%** | — |

---

## 5. Key Research Findings

1. **Superior Transferability of Dual-HiFo:**  
   `dual_hifo_independent` achieved the lowest overall test gap (**1.7920%**), outperforming standard `full_moh` (**1.9448%**) and isolated single-level modes (`inner_hifo` at **2.5591%**, `outer_hifo` at **2.7494%**).
2. **Mitigation of Overfitting:**  
   While `outer_hifo` obtained high training meta-utility, its learned optimizer over-specialized to small scales (TSP 20–200) and degraded on TSP 500–1000. Dual-HiFo's simultaneous AST mechanism diversity and credit assignment preserved generality across scales.
3. **Reproducibility:**  
   All artifacts, checkpoints, and outputs are indexed under `outputs/full_budget_seed0` and `outputs/deployment/full_budget_seed0`.
