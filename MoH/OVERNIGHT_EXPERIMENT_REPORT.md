# Stage 3 Overnight Experiment Report — Dual-HiFo Ablation

**Date:** 2026-10-17 → 2026-10-24  
**Git SHA:** `1d931f3`  
**Model:** `ag/gemini-3.8-flash-low` (via proxy `172.25.16.1:20128`)  
**Problem:** TSP `[20, 50, 100, 200]` with TSP-GLS  
**Budget:** n_iterations=3, pop_size=5, max_eval_calls=150/subtask (600 total), requests≤2000, attempts≤3000, retries/call=3  

---

## 1. Run Counts

| Status | Count |
|---|---:|
| Completed | 15 |
| Failed | 0 |
| Timeout | 0 |
| Budget exhausted | 0 |
| **Total** | **15** |

All 5 modes × 3 seeds (42, 43, 44) completed successfully.

---

## 2. Training Meta-Utility by Mode × Seed

Lower is better. Weighted mean of validation gaps across all 4 sizes.

| Mode | Seed 42 | Seed 43 | Seed 44 | **Mean** |
|---|---:|---:|---:|---:|
| **outer_hifo** | 0.209 | 0.526 | 0.177 | **0.304** |
| full_moh | 0.298 | 0.638 | 0.345 | 0.427 |
| dual_hifo_independent | 0.480 | 0.847 | 0.241 | 0.523 |
| dual_hifo_cross_level | 0.733 | 0.703 | 0.254 | 0.563 |
| inner_hifo | 1.194 | 0.406 | 0.208 | 0.603 |

**Key finding:** `outer_hifo` achieves the best (lowest) mean training meta-utility (0.304), outperforming `full_moh` (0.427) by ~29%.

---

## 3. Best Training Utility per Size (from checkpoint elites)

### TSP-20 (near-zero due to known small-size overfit — not informative)

All modes achieve ~1.5e-06. **Skip for comparison.**

### TSP-50

| Mode | Seed 42 | Seed 43 | Seed 44 | **Mean** |
|---|---:|---:|---:|---:|
| full_moh | 2.4e-06 | 1.7e-06 | 3.2e-06 | 2.4e-06 |
| outer_hifo | 2.4e-06 | 1.7e-06 | 3.2e-06 | 2.4e-06 |
| dual_independent | 2.4e-06 | 1.7e-06 | 3.2e-06 | 2.4e-06 |
| dual_cross_level | 0.217 | 1.7e-06 | 3.2e-06 | 0.072 |
| inner_hifo | 2.4e-06 | 1.7e-06 | 3.2e-06 | 2.4e-06 |

**Key finding:** Most modes converge to near-zero on TSP50. `dual_cross_level` seed42 is the outlier (0.217).

### TSP-100 (most discriminative)

| Mode | Seed 42 | Seed 43 | Seed 44 | **Mean** |
|---|---:|---:|---:|---:|
| **dual_independent** | 0.0518 | 0.0064 | 0.0050 | **0.021** |
| full_moh | 0.0585 | 0.0425 | 0.0239 | 0.042 |
| outer_hifo | 0.0368 | 0.1172 | 0.0442 | 0.066 |
| dual_cross_level | 0.1141 | 0.3179 | 0.0707 | 0.168 |
| inner_hifo | 0.5376 | 0.0583 | 0.0001 | 0.199 |

**Key finding:** `dual_independent` achieves the best mean (0.021) on TSP-100, outperforming `full_moh` (0.042) by ~50%.

### TSP-200

| Mode | Seed 42 | Seed 43 | Seed 44 | **Mean** |
|---|---:|---:|---:|---:|
| **dual_independent** | 0.432 | 0.177 | 0.198 | **0.269** |
| full_moh | 0.255 | 0.489 | 0.627 | 0.457 |
| outer_hifo | 0.283 | 0.850 | 0.180 | 0.438 |
| inner_hifo | 0.813 | 0.250 | 0.340 | 0.468 |
| dual_cross_level | 1.244 | 0.147 | 0.238 | 0.543 |

**Key finding:** `dual_independent` achieves the best mean (0.269) on TSP-200, outperforming `full_moh` (0.457) by ~41%.

---

## 4. HiFo Decision Patterns

### Inner HiFo Regimes (EXPLORE / BALANCE / EXPLOIT)

| Mode | Typical EXPLORE% | Typical BALANCE% | Typical EXPLOIT% |
|---|---:|---:|---:|
| inner_hifo | ~80% | ~18% | ~2% |
| dual_independent | ~80% | ~18% | ~2% |
| dual_cross_level | ~55% | ~35% | ~10% |

**Observation:** `dual_cross_level` has higher EXPLOIT% (~10%) vs other modes (~2%), suggesting cross-level feedback promotes exploitation.

### Outer HiFo Selections (accepted/total)

| Mode | Seed 42 | Seed 43 | Seed 44 | **Total** |
|---|---|---|---|---|
| outer_hifo | 1/3 | 3/3 | 1/3 | **5/9** |
| dual_independent | 1/3 | 1/3 | 2/3 | **4/9** |
| dual_cross_level | 3/3 | 1/3 | 2/3 | **6/9** |

**Observation:** `dual_cross_level` has the highest outer acceptance rate (6/9 = 67%), suggesting cross-level feedback helps the outer optimizer make better accept/reject decisions.

---

## 5. Aggregate Comparison (Mean across 3 seeds)

| Mode | Train meta-util | TSP-100 util | TSP-200 util | Outer accept rate |
|---|---:|---:|---:|---:|
| **outer_hifo** | **0.304** | 0.066 | 0.438 | 5/9 (56%) |
| full_moh | 0.427 | 0.042 | 0.457 | N/A |
| **dual_independent** | 0.523 | **0.021** | **0.269** | 4/9 (44%) |
| dual_cross_level | 0.563 | 0.168 | 0.543 | 6/9 (67%) |
| inner_hifo | 0.603 | 0.199 | 0.468 | N/A |

---

## 6. Key Findings

1. **`outer_hifo` best overall:** Lowest mean training meta-utility (0.304) across all sizes, outperforming `full_moh` by ~29%.

2. **`dual_independent` best on larger sizes:** Achieves the best mean on TSP-100 (0.021) and TSP-200 (0.269), outperforming `full_moh` by ~50% and ~41% respectively.

3. **`dual_cross_level` highest outer acceptance:** 6/9 (67%) outer selections accepted, suggesting cross-level feedback improves outer optimizer decisions.

4. **`inner_hifo` worst overall:** Highest mean training meta-utility (0.603), suggesting inner-only HiFo without outer optimization is insufficient.

5. **HiFo regimes:** EXPLORE dominates (~80%) in most modes. `dual_cross_level` has higher EXPLOIT% (~10%), indicating cross-level feedback promotes exploitation.

---

## 7. Comparison with Paper (Table 1)

| Setting | TSP-50 gap | TSP-100 gap |
|---|---:|---:|
| Paper MoH-GLS (test, 1000 instances) | 0.936% | 1.476% |
| Our best (val, 8 instances) | ~0% | 0.005% |

**Note:** Our validation gaps are much lower due to:
- Small validation set (8 instances vs 100 test instances)
- Known overfit on small sizes (TSP20/50)
- Different evaluation protocol

**Not directly comparable.** Test-set evaluation on 100 instances would be needed for fair comparison.

---

## 8. Checkpoint Paths

All checkpoints at `outputs/dualhifo/s3_<mode>_seed<seed>/checkpoint/`:

- `s3_full_moh_seed{42,43,44}`
- `s3_inner_hifo_seed{42,43,44}`
- `s3_outer_hifo_seed{42,43,44}`
- `s3_dual_hifo_independent_seed{42,43,44}`
- `s3_dual_hifo_cross_level_seed{42,43,44}`

---

## 9. Caveats

1. **Small validation set:** 8 instances per size. Results may not generalize.

2. **Small-size overfit:** TSP20/50 gaps are near-zero due to known overfit. Compare modes on TSP100/200.

3. **No test-set selection:** Optimizers chosen on validation only. Test-set evaluation needed for final claims.

4. **Multi-seed variance:** Significant variance across seeds (e.g., `inner_hifo` seed42=1.194 vs seed44=0.208). More seeds would improve reliability.

5. **LLM proxy flakiness:** Intermittent timeouts caused some runs to fail and require retries. This may introduce selection bias.

6. **Do not claim Dual-HiFo superior:** While `outer_hifo` and `dual_independent` show promising results, the small validation set and high variance mean we cannot claim superiority without larger-scale evaluation.

---

## 10. Reproducibility

```bash
# Git SHA
git rev-parse HEAD  # 1d931f33825249d40bbf90fee9106b59e9f4e303

# Dataset hashes
cat MoH/dataset_hashes_stage3.json

# Consolidated report
cat MoH/outputs/dualhifo/stage3_consolidated/stage3_report.md

# Raw results
cat MoH/outputs/dualhifo/stage3_consolidated/stage3_results.json
cat MoH/outputs/dualhifo/stage3_consolidated/stage3_runs.csv
```

---

*Report generated: 2026-10-24T07:11:00Z*
