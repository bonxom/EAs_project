# MoH-implement Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Tái hiện tìm kiếm heuristic TSP-GLS và tự cải tiến chương trình optimizer của repo MoH, với pipeline offline chạy thật và cấu trúc dễ đọc.

**Architecture:** Process cha giữ quần thể, điểm số, LLM và ngân sách. Process con chạy chương trình optimizer qua callback JSON; evaluator của cha mở worker GLS để chạy heuristic. Cùng giao diện `improve_algorithm` được dùng ở inner và outer, nhưng callback đánh giá khác nhau.

**Tech Stack:** Python 3.12, uv, NumPy, Pydantic, PyYAML, pytest, Ruff, Linux subprocess/pipe/selectors. Adapt GLS bằng Python/NumPy trước; Numba không là điều kiện để smoke chạy được.

**Spec:** `docs/superpowers/specs/2026-10-03-moh-implement-design.md` — người dùng đã duyệt trong hội thoại ngày 2026-10-03.

## Trạng thái thực hiện

Hoàn tất Tasks 1–10 bằng subagent, với review độc lập và regression cho các
lỗi được phát hiện. Checklist bên dưới giữ nguyên nội dung kế hoạch ban đầu.

- Pipeline CLI offline đã chạy thành công với hai vòng outer tự cải tiến,
  hai task held-out thành công và checkpoint đầy đủ.
- Hai lần chạy FakeLLM cho kết quả và semantic events giống nhau.
- Source worker được giữ lại và đối chiếu SHA256; điểm và population IDs
  được xác nhận bằng kết quả đánh giá của process cha.
- Review toàn nhánh: approve sau khi sửa lỗi deadline toàn search và ranh giới
  search/held-out bằng regression RED→GREEN.
- Kiểm tra cuối: `uv run ruff check .` sạch; `uv run pytest -q`: **580 passed**.

Đọc [hướng dẫn triển khai và đối chiếu upstream](../../moh-reproduction.md)
để xem cấu trúc hiện tại, cách chạy và những khác biệt có chủ ý. Chưa merge
hoặc push; kết quả này chưa xác nhận tái lập benchmark của paper.

## Global Constraints

- Python 3.12, uv, Linux; giữ các chế độ mini-MoH và các tests cũ.
- Tests dùng FakeLLM hoặc provider transport stubs; không external LLM APIs.
- Mã sinh ra không chạy trong process chính; module-level code cũng chạy trong worker.
- Generated programs có timeout; một ứng viên lỗi không làm hỏng thí nghiệm.
- Core, optimizer và runner độc lập provider; OpenAI-specific logic chỉ trong `llm/openai_client.py`.
- Fitness mới là gap %, minimize; record failed có `utility=None`, callback penalty là `1e6`.
- Quần thể heuristic tồn tại suốt run; kết quả một optimizer invocation lỗi không commit quần thể.
- Smoke dùng exact optimum cho tối đa 12 thành phố; dataset NPZ dùng `allow_pickle=False`.
- Validation/test split không giao nhau; held-out scores không quay lại search.
- Không sửa checkout `/home/bonxom/Code/MoH`, tải dataset hoặc gọi API trả phí.
- Giữ MIT attribution cho code/prompt adapt upstream trong `third_party/moh/`.
- Chạy `uv run ruff check .` và `uv run pytest -q` trước khi hoàn thành.

## Review Focus

1. Hết deadline outer khi callback đang mở worker inner: mọi process do cha tạo thuộc invocation phải được dọn, không chỉ group ngoài. Test ở Task 4 và 8.
2. Worker trả source/score không khớp hoặc sửa snapshot: cha giữ fitness và source lineage thật; không nhận score tự khai. Test ở Task 7.
3. Dataset tham chiếu hợp lệ về shape nhưng cost/tour sai hoặc split trùng: lỗi dữ liệu phải xuất hiện trước LLM/artifacts. Test ở Task 2 và 9.
4. Outer callback đánh giá thành công vài task rồi task sau lỗi: không commit nửa quần thể; công việc vẫn được tính. Test ở Task 8.
5. Batch LLM/response quá lớn hoặc không đủ ngân sách: kiểm tra trước khi gọi, counts không đếm gấp đôi và giữ mini-MoH cũ hoạt động. Test ở Task 6, 7 và 9.

## File map and execution order

| Task | Files chính | Deliverable |
| --- | --- | --- |
| 1 | `core/programs.py`, `core/program_population.py` | Records minimize, quần thể và transaction |
| 2 | `problems/tsp_gls/{task,dataset,exact,tour}.py`, `tools/convert_moh_dataset.py` | Dữ liệu kiểm tra được, optimum thật, split rõ |
| 3 | `problems/tsp_gls/{local_search,solver,baselines}.py` | Solver GLS có thể kiểm thử độc lập |
| 4 | `execution/{process,optimizer_protocol,optimizer_worker,optimizer_runner}.py` | Worker chạy mã optimizer và RPC có deadline |
| 5 | `execution/{gls_worker,gls_runner}.py`, `problems/tsp_gls/evaluation.py` | Chấm heuristic trong worker, fitness do cha xác nhận |
| 6 | `llm/{base,fake,recording,openai_client}.py`, prompts, helpers, seed programs | LLM interface và chương trình seed ở hai tầng |
| 7 | `optimizers/program_inner.py`, `execution/budgets.py` | Inner callback và quần thể có transaction |
| 8 | `optimizers/program_meta.py` | Outer tự cải tiến và nested evaluation |
| 9 | `program_config.py`, `experiments/`, CLI/configs | Pipeline chạy thật, artifacts và held-out |
| 10 | README, `docs/moh-reproduction.md`, tests integration | Hướng dẫn đọc, attribution và kiểm chứng toàn nhánh |

Các thư mục package mới có `__init__.py` trống. Tests mới đặt trong `tests/programs/`.
Không đổi `core.models.EvaluationResult` hoặc `rank_key()` cũ sang minimize.
Tất cả commit ở `MoH-implement`; chưa merge hoặc push.

---

### Task 1: Records và quần thể minimize

**Files:** Create `src/moh/core/programs.py`, `src/moh/core/program_population.py`, `tests/programs/test_records.py`, `tests/programs/test_program_population.py`.

**Interfaces:**
- Consumes: `Heuristic`, `WorkCounts` từ `moh.core.models`; `derive_seed` giữ nguyên.
- Produces: frozen `OptimizerProgram(id: str, source_code: str, idea: str | None = None)`.
- Produces: frozen `GapEvaluation(candidate_id, task_id, split, status, utility, costs, gaps, tours, error=None, counts=WorkCounts())`; `costs/gaps/tours` là tuples; split là `validation` hoặc `test`.
- Produces: frozen `TaskOutcome(task_id: str, selected: ScoredProgram | None)` và `ProgramEvaluation(candidate_id, status, utility, task_results: tuple[TaskOutcome, ...], error=None, counts=WorkCounts())`.
- Produces: frozen `ScoredProgram(candidate: Heuristic | OptimizerProgram, evaluation: GapEvaluation | ProgramEvaluation)`; properties `id`, `source_code`, `idea`, `utility` lấy từ records.
- Produces: `ProgramPopulation(capacity: int, members: tuple[ScoredProgram, ...] = ())`; `add(item) -> ProgramPopulation`, `best() -> ScoredProgram | None`, `snapshot(task: str) -> dict[str, list[dict]]`, `select(rng: random.Random) -> ScoredProgram`.
- Produces: `PopulationTransaction(populations: dict[str, ProgramPopulation])`; `.staged` là copy mapping immutable populations, `add(task, item)`, `commit() -> dict[str, ProgramPopulation]`; bỏ transaction là rollback.
- Produces: frozen `ProgramRunResult(status, winner, active, population, task_populations, test_results, counts)`.

- [ ] **Step 1: Write tests for minimize, identity, failure and snapshots.**

```python
import math
import random
import pytest
from moh.core.models import Heuristic, WorkCounts
from moh.core.programs import GapEvaluation, ScoredProgram
from moh.core.program_population import ProgramPopulation, PopulationTransaction

def scored(identity, gap, source=None):
    candidate = Heuristic(identity, source or f"# {identity}")
    evaluation = GapEvaluation(identity, "tsp4", "validation", "success", gap,
                               (4.0,), (gap,), ((0, 1, 2, 3, 0),),
                               counts=WorkCounts(1, 1, 0))
    return ScoredProgram(candidate, evaluation)

def test_minimize_and_deduplicate():
    population = ProgramPopulation(2).add(scored("b", 2)).add(scored("a", 1))
    assert [x.id for x in population.members] == ["a", "b"]
    assert population.add(scored("c", 3)).members == population.members
    assert len(population.add(scored("d", 1, "# a")).members) == 2

def test_transaction_and_snapshot_cannot_change_parent():
    original = {"tsp4": ProgramPopulation(2).add(scored("a", 2))}
    transaction = PopulationTransaction(original)
    transaction.add("tsp4", scored("b", 1))
    snapshot = original["tsp4"].snapshot("tsp4")
    snapshot["tsp4"][0]["utility"] = -100
    assert original["tsp4"].best().utility == 2
    assert transaction.commit()["tsp4"].best().utility == 1

def test_rank_selection_reproducible():
    population = ProgramPopulation(3)
    for i in range(3):
        population = population.add(scored(str(i), i))
    a, b = random.Random(42), random.Random(42)
    assert [population.select(a).id for _ in range(20)] == [population.select(b).id for _ in range(20)]

@pytest.mark.parametrize("gap", [math.nan, math.inf, -1.0, True])
def test_invalid_success_score(gap):
    with pytest.raises(ValueError):
        scored("a", gap)
```

- [ ] **Step 2: Run `uv run pytest -q tests/programs/test_records.py tests/programs/test_program_population.py`.** Expected: collection fails because new modules do not exist.
- [ ] **Step 3: Implement frozen records, validation and pure population updates.** Success requires finite nonnegative utility and no error; failed requires null utility and bounded error. `GapEvaluation` success requires nonempty matching tuples and mean gaps equal utility; `ProgramEvaluation` success requires a selected successful heuristic per task. Reject ID mismatch and failed members in populations. De-duplicate exact UTF-8 source; keep existing entry for identical source. Validate safe IDs for retained artifacts.

```python
def program_rank(item):
    return (item.utility, item.id)

def selection_weights(count, capacity):
    return [1.0 / (rank + capacity / 2.0) for rank in range(1, count + 1)]

# ProgramPopulation.add returns a new population sorted by program_rank.
# snapshot emits new dictionaries with id, best_sol, idea, utility.
# select uses rng.choices(self.members, weights=selection_weights(...), k=1)[0].
```

- [ ] **Step 4: Run Task 1 command again.** Expected: all tests pass; add parameterized tests for failed/nonfinite records, source duplicate tie and zero capacity.
- [ ] **Step 5: Commit with `git add src/moh/core/programs.py src/moh/core/program_population.py tests/programs` then `git commit -m 'feat: add program search records and minimizing populations'`.** Expected: commit created, legacy files unchanged.

### Task 2: Validated TSP data and exact smoke fixtures

**Files:** Create `src/moh/problems/tsp_gls/{__init__,task,dataset,exact,tour}.py`, `tools/convert_moh_dataset.py`, `tests/programs/test_gls_data.py`, `tests/programs/test_gls_tour.py`.

**Interfaces:**
- Consumes: `derive_seed`; `tour_length` cũ chỉ dùng cho đối chiếu Euclidean fixture.
- Produces: `GLSInstance(id: str, coordinates: ndarray, distances: ndarray, optimal_cost: float, optimal_tour: tuple[int, ...])`; frozen, read-only arrays, n>=4.
- Produces: `GLSTask(id: str, size: int, validation: tuple[GLSInstance, ...], test: tuple[GLSInstance, ...], provenance: dict)`; nonempty splits, unique instance IDs, size consistent.
- Produces: `tour_cost(distances, tour) -> float`, `tour_to_route(tour) -> ndarray`, `route_to_tour(route) -> tuple[int, ...]`, `nearest_neighbor(distances) -> tuple[int, ...]`; public tours start/end at 0, have n+1 cities; routes shape `(n, 2)` int predecessors/successors.
- Produces: `held_karp(distances) -> tuple[float, tuple[int, ...]]`, n<=12.
- Produces: `synthetic_task(size, validation_count, test_count, root_seed) -> GLSTask`; task ID is `f"tsp_gls{size}"`.
- Produces: `load_npz_task(path: Path, validation_indices: tuple[int, ...], test_indices: tuple[int, ...]) -> GLSTask`.

- [ ] **Step 1: Write exact and split tests.**

```python
import numpy as np
import pytest
from moh.problems.tsp_gls.exact import held_karp
from moh.problems.tsp_gls.dataset import synthetic_task, load_npz_task
from moh.problems.tsp_gls.tour import tour_to_route, route_to_tour, tour_cost

def square_distances():
    xy = np.array([[0., 0.], [1., 0.], [1., 1.], [0., 1.]])
    return np.linalg.norm(xy[:, None, :] - xy[None, :, :], axis=-1)

def test_square_optimum_and_route_roundtrip():
    distance = square_distances()
    cost, tour = held_karp(distance)
    assert cost == pytest.approx(4.0)
    assert tour_cost(distance, tour) == pytest.approx(cost)
    assert route_to_tour(tour_to_route(tour)) == tour

def test_synthetic_repeat_and_disjoint_splits():
    left = synthetic_task(6, 2, 2, 42)
    right = synthetic_task(6, 2, 2, 42)
    assert {x.id for x in left.validation}.isdisjoint(x.id for x in left.test)
    for a, b in zip((*left.validation, *left.test), (*right.validation, *right.test), strict=True):
        np.testing.assert_array_equal(a.distances, b.distances)
        assert a.optimal_cost == b.optimal_cost
        assert not a.distances.flags.writeable

def test_exact_size_guard():
    with pytest.raises(ValueError):
        held_karp(np.ones((13, 13)))

def test_reject_overlap_before_reading_file(tmp_path):
    with pytest.raises(ValueError, match="split"):
        load_npz_task(tmp_path / "absent.npz", (0, 1), (1, 2))
```

- [ ] **Step 2: Run `uv run pytest -q tests/programs/test_gls_data.py tests/programs/test_gls_tour.py`.** Expected: imports fail before implementation.
- [ ] **Step 3: Implement Held–Karp and dataset validation.** DP keyed `(visited_bitmask, last_city)`, start at 0, retain predecessor, deterministic tie by smallest predecessor; close at 0 and reconstruct. Reject unsupported size before allocating DP. NPZ required arrays are `coordinates: (k,n,2)`, `distance_matrix: (k,n,n)`, `cost: (k,)`, `optimal_tour: (k,n)`; normalize tour to start/end 0. Distances nonnegative, symmetric, zero diagonal and finite; coordinates finite; reference tour cost must agree within `rtol=1e-8, atol=1e-10`. Do not require distances to be raw Euclidean for precomputed benchmark matrices.

```python
validation_seed = derive_seed(root_seed, "gls", size, "validation", index)
test_seed = derive_seed(root_seed, "gls", size, "test", index)
# Generate coordinates using default_rng(seed).uniform(0.0, 1.0, (size, 2)).
# IDs include split/index; provenance contains seed labels or SHA256 of NPZ bytes.
```

Converter CLI is `uv run python tools/convert_moh_dataset.py --trusted-pickle INPUT --output OUTPUT`. It maps upstream `coordinate`, `distance_matrix`, `cost`, `optimal_tour` to numeric arrays and validates before `np.savez_compressed`. Never load a pickle automatically from dataset mode. Retain explicit warning in argparse help that pickle may execute code. Do not overwrite an existing output.
- [ ] **Step 4: Add tests for wrong reference cost, boolean/float tours, disconnected routes, NPZ object arrays, negative/NaN/asymmetric distances, out-of-range/repeated indices and pickle conversion of a locally created fixture. Run Task 2 command.** Expected: all tests pass and conversion output loads with `allow_pickle=False`.
- [ ] **Step 5: Commit `git add src/moh/problems/tsp_gls tools/convert_moh_dataset.py tests/programs` and `git commit -m 'feat: add validated GLS datasets and exact smoke instances'`.** Expected: commit created.

### Task 3: Adapt deterministic Guided Local Search

**Files:** Create `src/moh/problems/tsp_gls/{local_search,solver,baselines}.py`, `third_party/moh/LICENSE`, `third_party/moh/README.md`, `tests/programs/test_gls_solver.py`.

**Interfaces:**
- Consumes: Task 2 tour/route helpers and `GLSInstance`.
- Produces: frozen `GLSOptions(iterations: int = 10, perturbation_moves: int = 1, edges_per_move: int = 5, neighborhood_size: int = 100, reset_interval: int = 50)`.
- Produces: `two_opt(route, distances, neighbors, city=None) -> tuple[float, ndarray]`, `relocate(route, distances, neighbors, city=None) -> tuple[float, ndarray]`, `local_search(route, distances, neighbors) -> ndarray`.
- Produces: `solve_gls(distances, update_edge_distance, options: GLSOptions) -> tuple[int, ...]`; only hand-written test callbacks may call this in parent, generated callbacks only in GLS worker.
- Produces: `IDENTITY_SOURCE`, `PENALTY_SOURCE`, `UPSTREAM_BASELINE_SOURCE`, all define `update_edge_distance` and import NumPy explicitly.

- [ ] **Step 1: Write tests showing crossed-tour improvement and trusted scoring.**

```python
import numpy as np
import pytest
from moh.problems.tsp_gls.solver import GLSOptions, solve_gls
from moh.problems.tsp_gls.local_search import local_search
from moh.problems.tsp_gls.tour import tour_to_route, route_to_tour, tour_cost

def test_local_search_uncrosses_square():
    xy = np.array([[0., 0.], [1., 0.], [1., 1.], [0., 1.]])
    d = np.linalg.norm(xy[:, None, :] - xy[None, :, :], axis=-1)
    crossed = tour_to_route((0, 2, 1, 3, 0))
    neighbors = np.argsort(d, axis=1, kind="stable")[:, 1:]
    solved = route_to_tour(local_search(crossed, d, neighbors))
    assert tour_cost(d, solved) == pytest.approx(4.0)

def test_guidance_cannot_replace_true_distances():
    xy = np.array([[0., 0.], [1., 0.], [1., 1.], [0., 1.]])
    d = np.linalg.norm(xy[:, None, :] - xy[None, :, :], axis=-1)
    original = d.copy()
    seen = []
    def guide(distances, tour, penalties):
        seen.append(penalties.copy())
        distances[:] = 0.0
        return distances
    tour = solve_gls(d, guide, GLSOptions(iterations=3))
    np.testing.assert_array_equal(d, original)
    assert tour_cost(d, tour) == pytest.approx(4.0)
    assert len(seen) == 3
```

- [ ] **Step 2: Run `uv run pytest -q tests/programs/test_gls_solver.py`.** Expected: import fails.
- [ ] **Step 3: Port upstream move deltas and GLS iteration into the separated modules.** Read `/home/bonxom/Code/MoH/problems/tsp_gls/{gls,eval}.py`. Keep predecessor/successor representation and original neighbor traversal; remove unused functions and Numba decorators. Preserve per-city guided moves, original-distance local search, top-five penalty updates and reset interval. Return canonical closed tour at 0. Stable ordering for distance ties, `np.argmax` for guided difference, and zero both directions after choosing an edge. Copies prevent original distance/penalty mutation by callback. Validate returned matrix before using it.

```python
neighbors = np.argsort(distances, axis=1, kind="stable")[:, 1:1 + min(options.neighborhood_size, n - 1)]
current = local_search(tour_to_route(nearest_neighbor(distances)), distances, neighbors)
best = current.copy()
penalties = np.zeros_like(distances)
# Each perturbation: guided = validate_guidance(guide(distances.copy(), current_tour, penalties.copy())).
# Pick maximum guided-minus-original entries, increment both penalty directions,
# run two_opt/relocate around endpoints using guided weights, then original local_search.
# Compare current cost using original distances and copy best only when improved.
```

Retain upstream license and list adapted source files/revision. `UPSTREAM_BASELINE_SOURCE` adds missing `import numpy as np`; seed program fixes documented later. Do not vendor executable upstream evaluator or its pickle reader.
- [ ] **Step 4: Add tests for relocation delta against full cost, 2-opt delta against full cost, no guidance calls at zero iterations, reset interval, invalid matrix results and penalty symmetry. Add a small fixture regression using upstream-derived tour/cost recorded with attribution; do not import mutable external checkout from tests. Run Task 3 command.** Expected: deterministic tours, valid permutations, baseline cost never worsens nearest-neighbor plus local search.
- [ ] **Step 5: Commit `git add src/moh/problems/tsp_gls third_party/moh tests/programs/test_gls_solver.py` and `git commit -m 'feat: implement deterministic guided local search'`.** Expected: commit created.

### Task 4: Bounded process supervisor and optimizer RPC

**Files:** Create `src/moh/execution/{process,optimizer_protocol,optimizer_worker,optimizer_runner}.py`, `tests/programs/test_program_process.py`, `tests/programs/test_optimizer_rpc.py`.

**Interfaces:**
- Consumes: `OptimizerProgram`, existing `strict_json`, `_subreaper`/`_cleanup` behavior (reuse or extract without changing legacy semantics).
- Produces: `Deadline.after(seconds: float, parent: Deadline | None = None) -> Deadline`, `.remaining() -> float`, `.check() -> None` raising `CandidateFailure("timeout")`.
- Produces: frozen `ProgramLimits(timeout_seconds=30.0, source_bytes=65536, request_bytes=1048576, result_bytes=1048576, output_bytes=65536, max_callbacks=100, batch_size=5)`; strict positive finite values.
- Produces: `CandidateFailure(code: str)` for candidate failures, distinct from infrastructure exceptions.
- Produces: `ProcessSupervisor.scope(deadline, parent_scope=None)` context manager returning an ownership scope; closing/aborting scope kills/reaps registered processes and every descendant scope. Threads belong to scopes and join on close.
- Produces: `ProcessSupervisor.exchange(module: str, request: dict, dispatch, *, limits: ProgramLimits, deadline: Deadline, scope) -> dict`; dispatch signature `dispatch(operation: str, payload: dict, deadline: Deadline) -> JSONValue`. Module is a trusted constant, never worker input.
- Produces: frozen `OptimizerRequest(program, snapshot, task, function_format, seed, batch_size)` and `OptimizerWorkerResult(status, idea=None, source_code=None, claimed_utility=None, error=None)`.
- Produces: `OptimizerRunner(supervisor, limits).run(request: OptimizerRequest, dispatch, *, deadline: Deadline, scope) -> OptimizerWorkerResult`; actual invocation deadline is `Deadline.after(limits.timeout_seconds, parent=deadline)`.

- [ ] **Step 1: Write real-worker callback and timeout tests.**

```python
from moh.core.programs import OptimizerProgram
from moh.execution.process import Deadline, ProcessSupervisor, ProgramLimits
from moh.execution.optimizer_runner import OptimizerRequest, OptimizerRunner

def test_worker_uses_evaluation_callback_and_separate_output():
    source = '''def improve_algorithm(population, utility, language_model, function_format, task):
    print('{"op":"finish","utility":-100}')
    code = "def update_edge_distance(d, t, p): return d.copy()"
    score = utility(code, "identity", task)
    return "identity", code, score
'''
    calls = []
    def dispatch(operation, payload, deadline):
        assert operation == "evaluate"
        calls.append(payload)
        return 2.5
    supervisor = ProcessSupervisor()
    deadline = Deadline.after(5.0)
    with supervisor.scope(deadline) as scope:
        result = OptimizerRunner(supervisor, ProgramLimits()).run(
            OptimizerRequest(OptimizerProgram("o1", source), {}, "tsp4", "format", 42, 2),
            dispatch, deadline=deadline, scope=scope)
    assert result.status == "success"
    assert result.claimed_utility == 2.5
    assert len(calls) == 1

def test_module_level_infinite_loop_times_out():
    supervisor = ProcessSupervisor()
    deadline = Deadline.after(0.5)
    with supervisor.scope(deadline) as scope:
        result = OptimizerRunner(supervisor, ProgramLimits()).run(
            OptimizerRequest(OptimizerProgram("bad", "while True: pass"), {}, "tsp4", "format", 42, 2),
            lambda *args: None, deadline=deadline, scope=scope)
    assert result.status == "failed"
    assert result.error == "timeout"
```

- [ ] **Step 2: Run `uv run pytest -q tests/programs/test_program_process.py tests/programs/test_optimizer_rpc.py`.** Expected: missing module imports.
- [ ] **Step 3: Implement bounded NDJSON RPC over a result FD, with stdin for initial request/responses.** Request envelope is exactly `{"id": int, "op": str, "payload": dict}`; response envelope `{"id": int, "ok": bool, "value": JSONValue, "error": str | None}`. IDs start at 1 and increase without repeats. Allowed ops are `llm_prompt`, `llm_batch`, `evaluate`, `finish`; `finish` has success/failed schema and ends exchange. Parent validates op-specific types, finite temperature in `[0,2]`, task binding, source length, batch size and exact envelope keys before dispatch. Limit each frame before parsing; reject partial/non-UTF-8/duplicate/nonfinite frames. `finish` is handled by runner, not dispatched as an evaluation callback.

```python
# Worker only:
random.seed(request["seed"])
np.random.seed(request["seed"])
namespace = {"__name__": "__generated_optimizer__"}
exec(compile(request["source"], "<optimizer>", "exec"), namespace)
function = namespace.get("improve_algorithm")
if not callable(function):
    raise CandidateFailure("missing_function")
idea, source, score = function(population_view, utility_proxy, llm_proxy,
                              request["function_format"], request["task"])
# Send finish through result FD; worker catches BaseException and emits bounded failure.
```

Population façade copies input entries and implements Task 1 snapshot methods, using seeded `random.Random` and rank selection. `utility` sends source/idea/bound task; `prompt`/`prompt_batch` send expertise/messages/temperature. Callback errors become bounded worker exceptions; result tuple/string/score types checked in worker and again in parent.

Supervisor spawns `sys.executable -m <trusted module>` in temporary cwd with minimal environment, no API keys, result FD separate from stdout/stderr, process group/session per worker. A per-session monitor drains stdout/stderr nonblockingly while main thread dispatches callbacks; it enforces combined output limit and deadline even during a blocking nested callback. On expiry, mark cancelled and signal all groups in owning scope including child scopes; abort signalling is nonblocking and does not join its own monitor thread. The owning caller performs final reaping/join. RPC reader uses selectors with remaining deadline. Provider calls in later tasks receive remaining deadline too. Use thread-safe ownership registry and single owner of each FD; join monitors before releasing scope. All `finally` paths close FDs, wait/reap worker and clean adopted descendants. Infrastructure exception from dispatch propagates after cleanup; do not blanket-convert it into candidate failure.
- [ ] **Step 4: Add parameterized real-worker tests for syntax, exception/SystemExit, missing function, invalid tuple, stdout flood, forbidden op, forged ID, malformed/duplicate/nonfinite JSON, oversized frame, leaked descendant process and absent credential env. Add test where dispatch creates nested scope/worker and sleeps until outer deadline: both workers are reaped. Run Task 4 command.** Expected: tests pass with bounded runtime and no retained PIDs/FDs/monitor threads. Add cancellation tests for child scopes already closing; no double-close errors.
- [ ] **Step 5: Commit `git add src/moh/execution/process.py src/moh/execution/optimizer* tests/programs` and `git commit -m 'feat: execute optimizer programs through bounded worker RPC'`.** Expected: commit created.

### Task 5: Execute GLS heuristic and verify gap in parent

**Files:** Create `src/moh/execution/{gls_worker,gls_runner}.py`, `src/moh/problems/tsp_gls/evaluation.py`, `tests/programs/test_gls_runner.py`.

**Interfaces:**
- Consumes: `GLSTask`, `GLSOptions`, `solve_gls`, `Heuristic`, `GapEvaluation`, supervisor/deadline from Task 4.
- Produces: `gap_percent(cost: float, optimal_cost: float) -> float`; tolerance `1e-8 * max(cost, optimal_cost) + 1e-10`.
- Produces: `weighted_gap(outcomes: tuple[TaskOutcome, ...], weights: tuple[float, ...]) -> float`; positive finite weights and all selected successes required.
- Produces: `GLSRunner(supervisor, limits, options).evaluate(heuristic: Heuristic, task: GLSTask, *, split: str, root_seed: int, deadline: Deadline, scope) -> GapEvaluation`.

- [ ] **Step 1: Write baseline and isolation tests.**

```python
import pytest
from moh.core.models import Heuristic
from moh.execution.gls_runner import GLSRunner
from moh.execution.process import ProcessSupervisor, ProgramLimits, Deadline
from moh.problems.tsp_gls.dataset import synthetic_task
from moh.problems.tsp_gls.solver import GLSOptions
from moh.problems.tsp_gls.baselines import IDENTITY_SOURCE
from moh.problems.tsp_gls.evaluation import gap_percent

def test_identity_worker_has_valid_mean_gap():
    task = synthetic_task(4, 2, 1, 42)
    supervisor = ProcessSupervisor()
    deadline = Deadline.after(10.0)
    with supervisor.scope(deadline) as scope:
        result = GLSRunner(supervisor, ProgramLimits(), GLSOptions(iterations=2)).evaluate(
            Heuristic("h1", IDENTITY_SOURCE), task, split="validation",
            root_seed=42, deadline=deadline, scope=scope)
    assert result.status == "success"
    assert result.utility == pytest.approx(sum(result.gaps) / 2)
    assert result.counts.instance_attempts == 2

def test_gap_uses_minimization_and_rejects_bad_reference():
    assert gap_percent(105.0, 100.0) == pytest.approx(5.0)
    with pytest.raises(ValueError, match="reference"):
        gap_percent(90.0, 100.0)
```

- [ ] **Step 2: Run `uv run pytest -q tests/programs/test_gls_runner.py`.** Expected: new runner module missing.
- [ ] **Step 3: Implement worker and parent verification.** Worker loads seed, compiles/execs heuristic once, validates callable, reconstructs numeric data, invokes solver and emits tour-only `finish`. No callbacks allowed from GLS worker. Parent validates canonical tour and recomputes cost/gap from immutable original matrix. One failed instance fails evaluation with lengths/gaps already completed retained and accurate attempts. Dataset inconsistency `cost < optimum - tolerance` propagates as infrastructure error, not candidate penalty.

```python
cost = tour_cost(instance.distances, decoded_tour)
gap = gap_percent(cost, instance.optimal_cost)
# Worker-reported cost or fitness is never read.
# worker seed = derive_seed(root_seed, "gls", task.id, split, instance.id, "worker").
# Evaluation counts are WorkCounts(1, attempted_instances, 0), including failure.
```

- [ ] **Step 4: Test generated heuristic NaN/asymmetric/wrong-size result, input mutation, syntax, module loop, timeout and worker exit. Verify source-limit failure counts one evaluation but zero instance attempts if rejected before spawn. Verify test split uses different instances and no callback. Run Task 5 command.** Expected: all tests pass and next valid heuristic still runs after an invalid one.
- [ ] **Step 5: Commit `git add src/moh/execution/gls* src/moh/problems/tsp_gls/evaluation.py tests/programs/test_gls_runner.py` and `git commit -m 'feat: evaluate GLS heuristics with trusted gap scoring'`.** Expected: commit created.

### Task 6: LLM requests, helpers, prompts and optimizer seeds

**Files:** Modify `src/moh/llm/{base,fake,recording,openai_client}.py`; create `src/moh/llm/program_adapter.py`, `src/moh/optimizers/helpers.py`, `src/moh/optimizers/seeds/{__init__,basic,multi_temperature}.py`, `src/moh/prompts/{program_optimizer,gls_heuristic}.py`, `tests/programs/test_program_llm.py`, `tests/programs/test_seed_programs.py`.

**Interfaces:**
- Consumes: `GenerationError`, `RecordingLLM`, existing legacy `LLMClient.generate(prompt)` unchanged; `OptimizerRunner`, Task 3 baseline sources.
- Produces: frozen `LLMRequest(expertise: str, message: str, temperature: float | None = None, timeout_seconds: float | None = None)` with nonempty bounded text and finite valid numbers.
- Produces: `generate_request(request: LLMRequest) -> str` on FakeLLM, RecordingLLM and OpenAILLMClient; legacy `.generate(prompt)` retains exact existing behavior and retries.
- Produces: `ProgramLLM(client, *, batch_size: int).prompt(expertise, message, temperature, *, deadline: Deadline) -> str`, `.prompt_batch(expertise, messages, temperature, *, deadline: Deadline) -> list[str]`; sequential, input order retained.
- Produces: `extract_code(response: str | list[str]) -> str | list[str]`, `extract_idea(response: str | list[str]) -> str | list[str]`; braces in comment hold idea, fenced python contains code. For directions JSON, same extraction accepts json fence; reject empty/ambiguous multiple code blocks with `GenerationError`.
- Produces: `HEURISTIC_FORMAT`, `OPTIMIZER_FORMAT`, `seed_direction_prompt(task_id, previous_ideas) -> LLMRequest`, `seed_heuristic_prompt(task_id, direction) -> LLMRequest`; program-generated prompt text includes received function format.
- Produces: `BASIC_SOURCE`, `MULTI_TEMPERATURE_SOURCE`, imported as strings read from bundled seed `.py` files, never executed in parent. Seed files define upstream-compatible `improve_algorithm` and import `moh.optimizers.helpers`.

- [ ] **Step 1: Write FakeLLM/provider contract and extraction tests.**

```python
from moh.llm.base import LLMRequest
from moh.llm.fake import FakeLLM
from moh.optimizers.helpers import extract_code, extract_idea
from moh.prompts.program_optimizer import OPTIMIZER_FORMAT
from moh.prompts.gls_heuristic import HEURISTIC_FORMAT

def test_fake_distinguishes_program_and_heuristic_code():
    fake = FakeLLM(42)
    outer = fake.generate_request(LLMRequest("expert", OPTIMIZER_FORMAT, 0.7))
    inner = fake.generate_request(LLMRequest("expert", HEURISTIC_FORMAT, 0.7))
    assert "def improve_algorithm" in extract_code(outer)
    assert "def update_edge_distance" in extract_code(inner)

def test_idea_and_code_lists_preserve_alignment():
    values = ["# {first}\n```python\nx = 1\n```", "# {second}\n```python\nx = 2\n```"]
    assert extract_idea(values) == ["first", "second"]
    assert extract_code(values) == ["x = 1", "x = 2"]
```

- [ ] **Step 2: Run `uv run pytest -q tests/programs/test_program_llm.py tests/programs/test_seed_programs.py`.** Expected: missing LLMRequest/helper imports.
- [ ] **Step 3: Implement request adapters and seed strings.** FakeLLM has independent per-kind cursors for `directions`, `heuristic_program`, `optimizer_program`; use a first-line `KIND:` marker in new function formats and seed direction messages, rather than classifying by incidental code inside parent examples. Directions response is `{"insights": ["...", "..."]}`. New scripted response streams injectable, existing fake kinds preserved. Default responses vary by seed/cursor among at least three optimizer sources and three GLS heuristic sources. Two optimizer sources are the seeds; the third is a bundled variant of basic seed using best-parent selection instead of random selection, so outer can assess a program source not already initialized. No API usage.

```python
# openai_client.py only: provider request mapping, retain current validation/redaction/retries.
params = {"model": self.model, "input": request.message, "store": False,
          "timeout": min(self.timeout, request.timeout_seconds or self.timeout)}
if request.expertise:
    params["instructions"] = request.expertise
if request.temperature is not None:
    params["temperature"] = request.temperature
# Responses API call uses params; all attempts and retry sleeps fit one monotonic deadline.
```

Do not silently drop unsupported temperature: surface provider error via `GenerationError` and keep details inside provider adapter. RecordingLLM stores expertise/message/temperature and increments exactly once per logical request. ProgramLLM clamps timeout to deadline and rechecks between batch entries. Batch response length/bytes bounded again by runner before writing pipe.

Adapt basic seed's direction→batch→evaluate→min pipeline. Adapt multi-temperature seed's caches, `[0.7,1.0]` and top batch size selection, correcting descending sort to ascending. Direction messages start with `KIND: directions`; code messages start with the first line of received function_format (`KIND: heuristic_program` or `KIND: optimizer_program`), then include full format, parents and guidance. Generated direction count and batch truncated to façade `batch_size` before batch call. No parent import of executable seed module: source loading via `Path(...).read_text()` or package resources.
- [ ] **Step 4: Tests run both seed sources inside optimizer worker with FakeLLM-backed dispatch and finite evaluation fixtures. Assert temperatures, input-order alignment, code functions, minimize choice, de-dup eval cache and three distinct optimizer sources within one response cycle. Add provider transport-stub tests for instructions/temperature/timeout, deadline-aware retries, oversized response and legacy `generate()` regression. Run `uv run pytest -q tests/programs/test_program_llm.py tests/programs/test_seed_programs.py tests/test_fake_llm.py tests/test_openai_client.py tests/test_generation.py`.** Expected: all tests pass without network; no `exec` in seed loader/helpers.
- [ ] **Step 5: Commit `git add src/moh/llm src/moh/optimizers/helpers.py src/moh/optimizers/seeds src/moh/prompts/program_optimizer.py src/moh/prompts/gls_heuristic.py tests/programs` and `git commit -m 'feat: add program-generation LLM adapters and MoH optimizer seeds'`.** Expected: commit created.

### Task 7: Inner program search with authoritative fitness

**Files:** Create `src/moh/execution/{budgets,program_callbacks}.py`, `src/moh/optimizers/program_inner.py`, `tests/programs/test_program_inner.py`, `tests/programs/test_program_budgets.py`.

**Interfaces:**
- Consumes: Task 1 records/populations, Task 4 runner/scope/deadline, Task 5 GLSRunner and Task 6 ProgramLLM.
- Produces: `WorkBudget(max_llm_calls: int, max_heuristic_evaluations: int)` with `.counts: WorkCounts`, `require_llm(count)`, `charge_llm()`, `charge_evaluation()`, `charge_instances(count)` and `delta(before: WorkCounts) -> WorkCounts`. Limits global for entire run, not renewed for nested callbacks.
- Produces: `InvocationDispatcher(task: str, llm: ProgramLLM, budget: WorkBudget, evaluate, *, max_batch_size: int)`, `.dispatch(operation, payload, deadline) -> JSONValue`; `evaluate` signature `(source: str, idea: str | None, deadline: Deadline) -> float`.
- Produces: frozen `InnerProgramResult(status, selected: ScoredProgram | None, population: ProgramPopulation, evaluated: tuple[ScoredProgram, ...], error: str | None, counts: WorkCounts)`.
- Produces: `ProgramInner(optimizer_runner, gls_runner, budget, emit).search(program: OptimizerProgram, task: GLSTask, population: ProgramPopulation, llm: ProgramLLM, *, root_seed: int, invocation_id: str, deadline: Deadline, scope) -> InnerProgramResult`.
- Produces: `initialize_task_population(task, gls_runner, llm, budget, *, capacity: int, seed_attempts: int, threshold: float | None, root_seed: int, deadline: Deadline, scope, emit) -> ProgramPopulation`.

- [ ] **Step 1: Write tests for trusted score and rollback.** Test factory `run_inner(source, fake=None)` in this file constructs synthetic TSP4, ProcessSupervisor, real optimizer/GLS runners, baseline population assessed by GLSRunner, WorkBudget(30,30), FakeLLM(42), and calls ProgramInner.search under Deadline.after(15.0). Return `(result, original_population, budget, events)`; no generated code runs outside worker.

```python
def run_inner(source, fake=None):
    from moh.core.models import Heuristic
    from moh.core.programs import OptimizerProgram, ScoredProgram
    from moh.core.program_population import ProgramPopulation
    from moh.execution.budgets import WorkBudget
    from moh.execution.process import ProcessSupervisor, ProgramLimits, Deadline
    from moh.execution.optimizer_runner import OptimizerRunner
    from moh.execution.gls_runner import GLSRunner
    from moh.llm.fake import FakeLLM
    from moh.llm.program_adapter import ProgramLLM
    from moh.optimizers.program_inner import ProgramInner
    from moh.problems.tsp_gls.dataset import synthetic_task
    from moh.problems.tsp_gls.solver import GLSOptions
    from moh.problems.tsp_gls.baselines import IDENTITY_SOURCE
    task = synthetic_task(4, 1, 1, 42)
    budget, supervisor, events = WorkBudget(30, 30), ProcessSupervisor(), []
    emit = lambda event, payload: events.append({"event": event, **payload})
    deadline = Deadline.after(15.0)
    gls = GLSRunner(supervisor, ProgramLimits(), GLSOptions(iterations=2))
    optimizer = OptimizerRunner(supervisor, ProgramLimits())
    with supervisor.scope(deadline) as scope:
        baseline = Heuristic("baseline", IDENTITY_SOURCE)
        budget.charge_evaluation()
        evaluation = gls.evaluate(baseline, task, split="validation", root_seed=42,
                                  deadline=deadline, scope=scope)
        budget.charge_instances(evaluation.counts.instance_attempts)
        original = ProgramPopulation(3).add(ScoredProgram(baseline, evaluation))
        result = ProgramInner(optimizer, gls, budget, emit).search(
            OptimizerProgram("o1", source), task, original,
            ProgramLLM(fake or FakeLLM(42), batch_size=2), root_seed=42,
            invocation_id="inner1", deadline=deadline, scope=scope)
    return result, original, budget, events

def test_fabricated_score_rolls_back():
    source = '''def improve_algorithm(population, utility, language_model, function_format, task):
    code = "import numpy as np\\ndef update_edge_distance(d,t,p): return d.copy()"
    utility(code, "identity", task)
    return "identity", code, 12345.0
'''
    result, original, budget, events = run_inner(source)
    assert result.status == "failed"
    assert result.error == "unverified_result"
    assert result.population == original
    assert result.counts.heuristic_evaluations == 1

def test_unassessed_source_is_rejected():
    source = '''def improve_algorithm(population, utility, language_model, function_format, task):
    return "invented", "def update_edge_distance(d,t,p): return d.copy()", 0.0
'''
    result, original, budget, events = run_inner(source)
    assert result.status == "failed"
    assert result.population == original

def test_batch_budget_checked_before_any_call():
    from moh.execution.budgets import WorkBudget
    from moh.execution.process import CandidateFailure
    import pytest
    budget = WorkBudget(1, 5)
    with pytest.raises(CandidateFailure, match="budget"):
        budget.require_llm(2)
    assert budget.counts.llm_calls == 0
```

- [ ] **Step 2: Run `uv run pytest -q tests/programs/test_program_inner.py tests/programs/test_program_budgets.py`.** Expected: missing inner/budget modules.
- [ ] **Step 3: Implement callback dispatch, assessment registry and staged commit.** Start counts snapshot before invocation. GLS evaluation charges one heuristic evaluation before spawning and adds actual instance attempts afterward, including failure. Callback source receives monotonic invocation-local candidate ID, source retained by emit before evaluate. On success add ScoredProgram to invocation registry; on failed evaluation retain typed result/event and return `1e6`. Registry key is exact UTF-8 source, not worker ID. Snapshot entries are also known successful results; worker cannot alter them in parent.

```python
before = budget.counts
known = {item.source_code: item for item in population.members}
staged = population
# evaluate(source, idea, deadline): assess in GLS worker, store verified result,
# emit candidate/evaluation; staged = staged.add(item) only for success.
worker_result = optimizer_runner.run(request, dispatcher.dispatch,
                                     deadline=deadline, scope=scope)
selected = known.get(worker_result.source_code) if worker_result.status == "success" else None
verified = (selected is not None and
            math.isclose(selected.utility, worker_result.claimed_utility,
                         rel_tol=1e-10, abs_tol=1e-10))
# If not verified return failure with original population, preserving consumed counts.
# If verified return selected and staged population; chosen code/fitness come from parent records.
```

For LLM batch, validate count/bytes/temperature, `require_llm(len(messages))` once before work, then charge each logical call immediately before provider call; include failed calls. Dispatch cannot select arbitrary task. Separate GenerationError/CandidateFailure from OSError/data corruption/artifact errors; propagate latter after cleanup. `InnerProgramResult.evaluated` lists successes assessed in invocation even if population rolled back, for audit only.

Initialization evaluates identity, penalty and adapted upstream baseline under global budget, then uses direction/code prompts with finite `seed_attempts`. Save candidates below optional threshold; keep valid baselines even if no seed passes threshold. Source duplicates reuse assessed results within invocation/initialization, do not charge a second evaluation, and emit cache-hit event. Worker callbacks over `max_callbacks` fail independent of cache.
- [ ] **Step 4: Test real-worker success, failed heuristic returning penalty, callback wrong task, snapshot score mutation, nonfinite claimed score, duplicate source cache, exhausted budget and failed provider. Verify no partial population commit after callback then exception. Test seed_attempts bound with FakeLLM repeatedly invalid code and no qualifying threshold. Run Task 7 command.** Expected: all tests pass; before/after budget delta exactly matches recorded work.
- [ ] **Step 5: Commit `git add src/moh/execution/budgets.py src/moh/execution/program_callbacks.py src/moh/optimizers/program_inner.py tests/programs` and `git commit -m 'feat: run inner program search with verified fitness and budgets'`.** Expected: commit created.

### Task 8: Self-improving outer loop and nested transactions

**Files:** Create `src/moh/optimizers/program_meta.py`, `tests/programs/test_program_meta.py`.

**Interfaces:**
- Consumes: ProgramInner, InvocationDispatcher, WorkBudget, TaskOutcome/ProgramEvaluation/ScoredProgram/ProgramRunResult, OPTIMIZER_FORMAT and seed sources.
- Produces: `ProgramEvaluator(tasks: tuple[GLSTask, ...], weights: tuple[float, ...], inner: ProgramInner, llm_factory, budget, emit)`; `.evaluate(program: OptimizerProgram, populations: dict[str, ProgramPopulation], *, root_seed: int, invocation_id: str, deadline: Deadline, scope) -> tuple[ProgramEvaluation, dict[str, ProgramPopulation]]`.
- Produces: `ProgramMeta(optimizer_runner, evaluator, budget, emit).search(seeds: tuple[OptimizerProgram, ...], populations: dict[str, ProgramPopulation], meta_llm: ProgramLLM, *, iterations: int, capacity: int, root_seed: int, deadline: Deadline, scope) -> ProgramRunResult`; result test_results initially empty, set by Task 9 composition root.
- `llm_factory(scope: tuple[str, ...]) -> ProgramLLM`; scope names determine seed/cursor and recording lineage. All workers reuse supervisor and pass descendant scopes/deadlines.

- [ ] **Step 1: Write self-improvement trace and rollback tests.** Factory `run_meta_fixture(meta_responses=None, iterations=2, tasks=2)` constructs real runners, small synthetic tasks, initialized baseline populations, FakeLLM clients, budget and emit list. Provide scripted optimizer responses whose source changes behavior and records distinguishable LLM prompt markers; the fixture returns `(result, events)`.

```python
def run_meta_fixture(meta_responses=None, iterations=2, tasks=2):
    from moh.core.programs import OptimizerProgram
    from moh.execution.budgets import WorkBudget
    from moh.execution.process import ProcessSupervisor, ProgramLimits, Deadline
    from moh.execution.optimizer_runner import OptimizerRunner
    from moh.execution.gls_runner import GLSRunner
    from moh.llm.fake import FakeLLM
    from moh.llm.program_adapter import ProgramLLM
    from moh.optimizers.program_inner import ProgramInner, initialize_task_population
    from moh.optimizers.program_meta import ProgramEvaluator, ProgramMeta
    from moh.optimizers.seeds import BASIC_SOURCE, MULTI_TEMPERATURE_SOURCE
    from moh.problems.tsp_gls.dataset import synthetic_task
    from moh.problems.tsp_gls.solver import GLSOptions
    sizes = (4, 5)[:tasks]
    task_list = tuple(synthetic_task(size, 1, 1, 42) for size in sizes)
    supervisor, budget, events = ProcessSupervisor(), WorkBudget(100, 100), []
    emit = lambda event, payload: events.append({"event": event, **payload})
    optimizer = OptimizerRunner(supervisor, ProgramLimits())
    gls = GLSRunner(supervisor, ProgramLimits(), GLSOptions(iterations=2))
    factory = lambda scope: ProgramLLM(FakeLLM(42), batch_size=2)
    inner = ProgramInner(optimizer, gls, budget, emit)
    evaluator = ProgramEvaluator(task_list, tuple(float(s) for s in sizes), inner, factory, budget, emit)
    variant = BASIC_SOURCE.replace("get_random_solution(task)", "get_best_solution(task)")
    responses = list(meta_responses) if meta_responses is not None else [variant] * 20
    meta_llm = ProgramLLM(FakeLLM(42, {"optimizer_program": responses}), batch_size=2)
    deadline = Deadline.after(45.0)
    with supervisor.scope(deadline) as scope:
        populations = {task.id: initialize_task_population(
            task, gls, factory((task.id,)), budget, capacity=3, seed_attempts=0,
            threshold=None, root_seed=42, deadline=deadline, scope=scope, emit=emit)
            for task in task_list}
        seeds = (OptimizerProgram("seed1", BASIC_SOURCE),
                 OptimizerProgram("seed2", MULTI_TEMPERATURE_SOURCE))
        result = ProgramMeta(optimizer, evaluator, budget, emit).search(
            seeds, populations, meta_llm, iterations=iterations, capacity=3,
            root_seed=42, deadline=deadline, scope=scope)
    return result, events

def test_next_round_executes_selected_optimizer_program():
    result, events = run_meta_fixture(iterations=2)
    rounds = [e for e in events if e["event"] == "meta_round_started"]
    accepted = [e for e in events if e["event"] == "active_optimizer_changed"]
    assert len(rounds) == 2
    assert rounds[1]["active_id"] == accepted[0]["optimizer_id"]
    assert rounds[1]["source_hash"] == accepted[0]["source_hash"]
    assert result.winner.evaluation.utility == min(x.utility for x in result.population)

def test_failure_on_second_task_rolls_back_first_task():
    fail_second = '''def improve_algorithm(population, utility, language_model, function_format, task):
    if task.endswith("5"):
        raise RuntimeError("scripted second-task failure")
    source = "import numpy as np\\ndef update_edge_distance(d,t,p): return d.copy() + 0.1*p"
    return "penalty", source, utility(source, "penalty", task)
'''
    result, events = run_meta_fixture(meta_responses=[fail_second] * 20)
    failure = next(e for e in events if e["event"] == "optimizer_evaluated" and e["status"] == "failed")
    assert failure["population_before"] == failure["population_after"]
    assert failure["counts"]["heuristic_evaluations"] > 0
    assert any(e["event"] == "meta_round_finished" for e in events)
```

The failure source creates a valid novel heuristic for task one and raises for task two; fixtures use actual worker execution and FakeLLM, never parent exec. Task sizes 4 and 5 exercise unequal default weights. Also assert `rounds[1]["source_hash"] != rounds[0]["source_hash"]` so executing the same initial seed cannot satisfy the self-improvement test.
- [ ] **Step 2: Run `uv run pytest -q tests/programs/test_program_meta.py`.** Expected: missing meta module.
- [ ] **Step 3: Implement whole-optimizer transaction and outer registry.** ProgramEvaluator copies populations, searches each task on staged copy under inherited deadline, accumulates selected results, commits only after every task succeeds. Failed result retains consumed counts but returns original mapping. Use budget counter delta for total assessment, not sum plus delta; no double accounting.

```python
weights = supplied_weights or tuple(float(task.size) for task in tasks)
# evaluation.utility = weighted_gap(tuple(task_outcomes), weights).
# Seed programs are assessed with same evaluator before entering meta population.
# Choose basic successful seed as initial active, fallback to best successful seed.
# Each outer round:
#   - snapshot meta population; evaluate callback assesses new optimizer program;
#   - successful callback assessments stage their task populations and ScoredPrograms;
#   - validate final selected source/score against parent-owned outer registry;
#   - commit all staged state only if outer invocation finishes with verified selection;
#   - selected assessed program becomes active, even if it is not new global best;
#   - population keeps best capacity, winner is its minimum; active retained separately.
# On failure discard outer staged state, use best already-assessed optimizer as active.
```

Within one outer invocation, sequential candidate optimizer assessments see staged heuristic populations from prior successful assessments, matching shared-state protocol. Failed assessment leaves prior staged state intact; failed outer invocation discards the entire outer transaction. Callback depth remains fixed; inner task utility only evaluates heuristics, not more optimizers. Parent binds `meta-optimizer` scope, creates descendant ownership scopes and min-deadlines for evaluator callbacks. Registry keys exact source; IDs deterministic by outer round/callback index. Never evaluate worker-supplied functions in parent.
- [ ] **Step 4: Test worse-but-valid active vs best distinction, zero outer iterations, all seed failures, invalid optimizer followed by recovery, callback duplicate source, weighted gap `[4,5]`, global budget stop, source/score mismatch and nested timeout cleanup. Verify a failed outer invocation does not leak inner commits. Run Task 8 command.** Expected: tests pass; source transition proves self-improvement rather than JSON configuration search.
- [ ] **Step 5: Commit `git add src/moh/optimizers/program_meta.py tests/programs/test_program_meta.py` and `git commit -m 'feat: add self-improving meta-optimization of program search'`.** Expected: commit created.

### Task 9: Configuration, artifacts, CLI and held-out evaluation

**Files:** Create `src/moh/program_config.py`, `src/moh/experiments/{__init__,program_search,artifacts}.py`, `configs/{moh_smoke,moh_dataset}.yaml`, `tests/programs/{test_program_config,test_program_artifacts,test_program_smoke,test_program_cli}.py`; modify `src/moh/{main,experiment}.py`.

**Interfaces:**
- Consumes: all Task 1–8 interfaces; existing `RunRecorder`, `StrictConfig`, `UniqueKeyLoader`, `LLMConfig` and legacy CLI modes retained.
- Produces: `ProgramConfig` (Pydantic strict/frozen/extra forbid), `load_program_config(path) -> ProgramConfig`.
- Produces: `run_program_experiment(config: ProgramConfig) -> tuple[ProgramRunResult, Path]`; `experiment.py` exposes thin lazy wrapper of same name.
- Produces: `ProgramRecorder` wrapping or subclassing RunRecorder: `.save_optimizer(OptimizerProgram)` writes `.py`, `.save_heuristic(Heuristic)`, `.checkpoint(label, populations, active, winner)`, `.emit`, `.finish`, `.fail`; schema version 2 and `.metadata` provenance.
- CLI `--mode moh --config configs/moh_smoke.yaml`; default mode and default config remain legacy. Output `status`, `winner`, `utility`, `objective="mean_gap_percent"`, `run_dir`; existing exit codes 0/1/2 retained.

- [ ] **Step 1: Write offline smoke repeatability and validation-before-side-effects tests.**

```python
import json
import pytest
from moh.program_config import ProgramConfig
from moh.experiments.program_search import run_program_experiment

def semantic_events(path):
    return [json.loads(line) for line in (path / "events.jsonl").read_text().splitlines()]

def test_program_smoke_reproducible(tmp_path):
    config = ProgramConfig(output_dir=tmp_path / "a")
    left, lp = run_program_experiment(config)
    right, rp = run_program_experiment(config.model_copy(update={"output_dir": tmp_path / "b"}))
    assert left == right
    assert semantic_events(lp) == semantic_events(rp)
    assert left.status == "success"
    assert left.test_results
    assert list((lp / "optimizers").glob("*.py"))
    assert not list((lp / "optimizers").glob("*.json"))

def test_invalid_config_creates_no_artifacts(tmp_path, monkeypatch):
    def forbid(*args, **kwargs):
        pytest.fail("side effect before validation")
    monkeypatch.setattr("moh.experiments.artifacts.ProgramRecorder.create", forbid)
    with pytest.raises(ValueError):
        run_program_experiment(ProgramConfig().model_copy(update={"seed": True}))
```

- [ ] **Step 2: Run `uv run pytest -q tests/programs/test_program_config.py tests/programs/test_program_artifacts.py tests/programs/test_program_smoke.py tests/programs/test_program_cli.py`.** Expected: missing modules.
- [ ] **Step 3: Implement exact configuration and composition root.** Config has independent `heuristic_llm` and `meta_llm`, positive normalized `weights` or default sizes, nested solver/execution/budget/dataset settings. `outer_iterations` may be zero; seed attempts may be zero; population capacities positive; all source/frame limits strictly integer. Synthetic sizes `[4,6]`, two validation and one test per task, GLS iterations 3, outer rounds 2, capacity 3, seed_attempts 1, batch_size 2, run timeout 120s, optimizer timeout 30s, heuristic timeout 5s, max_llm_calls 100, max_heuristic_evaluations 100 are smoke defaults. Timeouts are safety bounds, not search budgets.

```yaml
# configs/moh_smoke.yaml
seed: 42
output_dir: outputs
outer_iterations: 2
population_size: 3
seed_attempts: 1
tasks:
  source: synthetic
  sizes: [4, 6]
  validation_count: 2
  test_count: 1
heuristic_llm:
  provider: fake
meta_llm:
  provider: fake
solver:
  iterations: 3
execution:
  batch_size: 2
budgets:
  max_llm_calls: 100
  max_heuristic_evaluations: 100
```

Dataset config names local NPZ files for sizes 100/200, validation indices `[0,1,2,3,4,5,6,7]` and test indices `[8,9,10,11,12,13,14,15,16,17]`, GLS iterations 1000, and includes instructions to replace paths/model names before use. Default provider in this template is fake until user explicitly configures real models; no live run in this task. Validate path existence, data/reference consistency and split before recorder/provider initialization; header metadata contains data hashes and solver config.

Composition root revalidates config even after `model_copy`, loads tasks, validates provider environment only for selected provider through provider adapter, opens recorder and supervisor, initializes populations, evaluates seeds then runs meta search. LLM factory scopes seed FakeLLM via `derive_seed(config.seed, "program_llm", *scope)`. Recording logs deterministic scope/request IDs and request/response; all executed sources saved first. Counter state owned by WorkBudget; no credentials/environment in events.

After search, use heuristic selected in winning optimizer's validation TaskOutcome for each task, not whichever heuristic happens to be newest. Evaluate held-out using same GLSRunner and global budget/deadline with split `test`; record outcomes separately without population updates. Insufficient remaining budget produces explicit failed held-out records with null scores and counts, preserves search winner, and marks held-out completeness separately in run metadata. Validation success status refers to search; metadata has `test_status` to distinguish incomplete held-out evaluation. No post-hoc fitness replacement from test.

Artifacts: `.py` code, validated events, parent-approved evaluations, `populations/initial.json`, per-round checkpoints, `run.json` with winner/active/test results/counts/provenance. Record failed sources and attempts too; source hashes tie worker invocations to code. Avoid nondeterministic duration/timestamp fields in semantic events. Paths/time/version metadata remain in run/config artifacts, not result equality.
- [ ] **Step 4: Test missing dataset/wrong optimum before recorder, recorder OSError propagation with run error cleanup, all failed search, separate test_status, no held-out search feedback, safe artifact IDs/collisions, separate heuristic/meta client scopes, fake smoke two runs, and CLI error output. Run Task 9 command plus `uv run pytest -q tests/test_cli.py tests/test_smoke.py`.** Expected: new and old modes pass, no network.
- [ ] **Step 5: Run `uv run python -m moh.main --mode moh --config configs/moh_smoke.yaml`.** Expected: exit 0, objective mean_gap_percent, actual optimizer `.py` artifacts, active transition events and nonempty held-out results. Inspect run.json/checkpoints/source hashes; remove no artifacts from other runs.
- [ ] **Step 6: Commit `git add src/moh/program_config.py src/moh/experiments src/moh/main.py src/moh/experiment.py configs/moh_smoke.yaml configs/moh_dataset.yaml tests/programs` and `git commit -m 'feat: expose reproducible MoH program experiments and artifacts'`.** Expected: commit created.

### Task 10: Reading guide, fidelity notes and full-branch verification

**Files:** Modify `README.md`, `third_party/moh/README.md`; create `docs/moh-reproduction.md`, `tests/programs/test_program_regressions.py`.

**Interfaces:**
- Consumes: CLI, artifacts, seed/GLS algorithms and current baseline tests.
- Produces: Vietnamese guide showing mini-MoH vs program MoH, reading order, original-to-new file mapping, command examples, dataset converter and intentional differences.

- [ ] **Step 1: Add regression assertions for worker-only generated code and retained sources.** Use AST scan of repository modules to assert `exec`/`compile` for generated code only in execution worker modules; seed source loaders read files without executing them. Run one real smoke and assert every `optimizer_worker_started` source hash matches saved code, all selected programs have parent-verified evaluations, and every population-update ID is assessed.

```python
def test_selected_optimizer_has_retained_verified_source(tmp_path):
    from moh.program_config import ProgramConfig
    from moh.experiments.program_search import run_program_experiment
    import hashlib
    import json
    result, path = run_program_experiment(ProgramConfig(output_dir=tmp_path))
    code = (path / "optimizers" / f"{result.winner.id}.py").read_text()
    assert code == result.winner.source_code
    expected = hashlib.sha256(code.encode("utf-8")).hexdigest()
    events = [json.loads(x) for x in (path / "events.jsonl").read_text().splitlines()]
    assert any(e["event"] == "optimizer_worker_started" and e["source_hash"] == expected for e in events)
```

- [ ] **Step 2: Run `uv run pytest -q tests/programs/test_program_regressions.py`.** Expected: may already pass if preceding tasks satisfy contracts; treat a pre-existing pass as evidence, not claim a missing feature was implemented here. If it fails, investigate root cause with systematic-debugging before edits.
- [ ] **Step 3: Write reading guide and fidelity notes.** README contains new command and tree, two levels explained with `utility` as minimize fitness, sequence `main → program_search → program_meta → program_inner → optimizer_runner → gls_runner → solver`. `docs/moh-reproduction.md` maps upstream `moh.py`, Pop, seeds, GLS, eval and prompt files to modules, explains transactions/active vs best/ownership scopes, fixed GLS iterations vs wall-clock, no dataset/results claim, provider-independent API and exact smoke limitation. Include converter command and explicit trusted-pickle note. Keep old mini-MoH documentation labeled as original mode instead of contradicting new mode.

```text
uv sync --locked
uv run python -m moh.main --mode moh --config configs/moh_smoke.yaml
uv run python tools/convert_moh_dataset.py --trusted-pickle /path/input.pkl --output /path/output.npz
uv run ruff check .
uv run pytest -q
```

Record all upstream adaptations and fixes: missing NumPy import in baseline, seed descending ranking, test split sample bug, main-process exec removal, score/source acceptance consistency, strict output checks, transaction rollback and deterministic iteration budget. Do not claim OS hostile-code sandbox or live LLM reproducibility. Record upstream commit if available; otherwise SHA256 of adapted source files, with MIT license intact.
- [ ] **Step 4: Run `uv run ruff check .` and `uv run pytest -q`.** Expected: Ruff passes, all old/new tests pass. Then run smoke command from Task 9 once if Task 10 changed behavior; otherwise retain Task 9 evidence. Do not repeat entire suites absent new changes/failures.
- [ ] **Step 5: Commit `git add README.md docs/moh-reproduction.md third_party/moh/README.md tests/programs/test_program_regressions.py` and `git commit -m 'docs: explain program-based MoH reproduction and fidelity limits'`.** Expected: commit created; `git status --short` clean except intentional run artifacts ignored by git.

## Final review and completion

Review entire branch against the spec, concentrating on the five Review Focus conditions, not just the happy path. Review counts/deadlines across nested callbacks, source/score identity, rollback, shared-state lineage, data conversion and legacy compatibility. With Native execution, implementer handles tasks and one fresh reviewer checks the whole branch; with Subagent-driven execution, use the chosen per-task gates plus final review. Do not run subagents until authorized by the chosen execution method or applicable skill.

If skill helper scripts reference unavailable sibling skills, record the missing tooling in this plan's ledger and preserve the same task/test/commit tracking manually; never fabricate a dispatch or test result. Keep ledger under `.superpowers/sdd/2026-10-03-moh-implement/progress.md`, beginning with the exact plan path. Any interface change requires a ledger ruling with affected consumers and practical cost.

Completion means a running offline pipeline and green checks, with a reviewable committed branch. Leave `MoH-implement` checked out in the IDE; do not merge/push without instruction. Final response links reading guide, reports checks and smoke results, identifies any concrete deviations or remaining limitations, and gives the command to run it.

## Spec coverage and plan self-review

| Spec sections | Tasks |
| --- | --- |
| 1–2 scope, constraints, attribution | Global Constraints, 3, 10 |
| 3 inner and outer interfaces/self-improvement | 4, 6, 7, 8 |
| 4 minimize fitness, persistent populations, rollback | 1, 5, 7, 8 |
| 5 GLS, data, exact optimum and held-out | 2, 3, 5, 9 |
| 6 process/callback/deadline/limits | 4, 6, 7, 8 |
| 7 readable structure and legacy preservation | File map, 1–10 |
| 8 config, provenance, counters/reproducibility | 2, 6, 7, 9 |
| 9 intentional differences and limits | 3, 8, 10 |
| 10 acceptance criteria | Tests in 1–10, smoke in 9, final review |

Review checks: all cross-task signatures above are authoritative; optional defaults are documented; no task asks for network/paid API or benchmark dataset. Every Review Focus input has an owning test task. The configuration defaults form one small end-to-end run. No product code has been changed while writing this plan.
