# Đọc và đối chiếu MoH bằng chương trình

Tài liệu này mô tả chế độ MoH bằng chương trình theo
[spec đã duyệt](superpowers/specs/2026-10-03-moh-implement-design.md).
Pipeline ghép dữ liệu, GLS, worker, LLM, inner và outer, có CLI offline và
artifacts để kiểm tra source, điểm số và các quyết định tìm kiếm.

## Hai tầng dùng cùng một giao diện

Một **heuristic** định nghĩa:

```python
def update_edge_distance(edge_distance, local_opt_tour, edge_n_used):
    return updated_edge_distance
```

Nó trả ma trận dẫn hướng perturbation trong Guided Local Search (GLS).
`local_opt_tour` không lặp thành phố cuối; ma trận penalty là đối xứng. Worker
truyền bản sao dữ liệu, kiểm tra shape, kiểu số thực, finite, không âm và đối
xứng. Prompt yêu cầu zero diagonal; evaluator chấp nhận diagonal khác zero để
chạy baseline upstream. Cost tour luôn dùng distance matrix gốc.

Một **chương trình optimizer** định nghĩa:

```python
def improve_algorithm(population, utility, language_model, function_format, task):
    return best_idea, best_solution, best_utility
```

| Bối cảnh | Population chứa | `function_format` mô tả | `utility(...)` đánh giá |
| --- | --- | --- | --- |
| Inner (`ProgramInner`) | Heuristic của một task | `update_edge_distance` | GLS trên validation instances |
| Outer (module `program_meta.py`) | Chương trình optimizer | `improve_algorithm` | Chạy optimizer ứng viên ở inner trên từng task |

Worker có `population.get_random_solution(task)`, `get_best_solution(task)` và
các phương thức đọc snapshot. Entry có `idea`, `best_sol`, `utility`. Snapshot
là bản sao; sửa nó không sửa quần thể ở cha. `language_model.prompt(...)` và
`prompt_batch(...)` gửi RPC về cha. Chương trình optimizer quyết định cách chọn
cha, tạo prompt và chọn ứng viên. `LLMClient` chỉ sinh văn bản.

Seed cơ bản chọn cha → xin directions JSON → sinh batch code → đánh giá → lấy
score thấp nhất. Seed multi-temperature thử `[0.7, 1.0]`, cache đánh giá cùng
source trong invocation và xếp tăng dần. `best_parent.py` là biến thể local chọn
cha tốt nhất, tạo optimizer thứ ba cho FakeLLM. Các nguồn seed được đọc bằng
`Path.read_text()`, không import module thực thi trong cha.

**Tự cải tiến** nghĩa là outer đề xuất và chấm một chương trình optimizer mới,
sau đó thực thi chương trình được chọn ở vòng kế tiếp. Cơ chế này không bảo đảm
mọi vòng cải thiện điểm. Active optimizer là chương trình dùng để đề xuất tiếp;
global best giữ chương trình có điểm tốt nhất đã gặp. Khi đề xuất lỗi, outer
có thể tiếp tục bằng chương trình tốt nhất đã chấm.

## Thứ tự đọc

1. `main.py`, `program_config.py`, `configs/moh_smoke.yaml`: cách chọn mode,
   seed, splits, số vòng GLS, deadline và ngân sách.
2. `experiments/program_search.py`: composition root ghép task, LLM hai tầng,
   runners, populations, outer loop, held-out evaluation và artifacts.
3. `optimizers/program_meta.py`: quần thể optimizer, active/global best và
   đánh giá lồng trên nhiều task.
4. `optimizers/program_inner.py`: khởi tạo baseline/seed, dispatch LLM và
   evaluate, xác minh source/fitness trả về, commit quần thể.
5. `execution/optimizer_runner.py`, `optimizer_protocol.py`,
   `optimizer_worker.py`: chạy source trong process con và callback JSON.
6. `execution/gls_runner.py`, `gls_worker.py`,
   `problems/tsp_gls/evaluation.py`: tour do worker trả, cost/gap do cha kiểm tra.
7. `problems/tsp_gls/solver.py`, `local_search.py`, `tour.py`: nearest neighbor,
   2-opt/relocate, perturbation, penalty và reset best route.

Tra records ở `core/programs.py` và quần thể ở `core/program_population.py`.
Đọc `prompts/` và `llm/` sau khi hiểu luồng callback; chỉ đọc prompt không cho
thấy ai giữ quần thể hoặc tính fitness.

## Điểm số và ai có quyền quyết định

Với mỗi validation instance:

```python
gap_percent = max(0.0, (tour_cost / optimal_cost - 1.0) * 100.0)
```

Utility heuristic là mean gap %; **thấp hơn tốt hơn**. Clamp chỉ xử lý sai số
nhỏ; cost tốt hơn reference quá tolerance là lỗi dữ liệu tham chiếu. Cha kiểm
tra tour và tính cost từ distance gốc, không nhận cost/score tự khai của worker.
Utility optimizer là weighted mean score heuristic đã chọn trên các task;
trọng số mặc định theo kích thước task. Thiếu kết quả inner hợp lệ ở bất kỳ task
nào làm optimizer thất bại. Chế độ mini-MoH cũ vẫn dùng âm mean tour length và
maximize; không so trực tiếp utility giữa hai chế độ.

Record ứng viên thất bại giữ `utility=None` (`null` trong JSON). Callback
upstream-compatible có thể trả penalty `1e6`; penalty không biến record lỗi
thành ứng viên hợp lệ. Source trả về phải có đánh giá thành công trong callback
ledger của invocation hoặc có trong snapshot hợp lệ. Cha đối chiếu source,
lineage và điểm trước khi chấp nhận kết quả.

Quần thể heuristic theo task tồn tại suốt run, xếp theo utility tăng dần,
tie-break ID ổn định và không lưu lặp source. Mỗi invocation dùng transaction:
chỉ commit các ứng viên đã được chấm khi invocation thành công. Outer đánh giá
optimizer trên nhiều task phải commit tất cả hoặc rollback tất cả khi một task
lỗi. Calls/evaluations/instance attempts đã tiêu thụ vẫn được tính khi rollback.

## Dữ liệu và cách chạy

Chạy CLI offline và kiểm tra repository:

```bash
uv sync --locked
uv run python -m moh.main --mode moh --config configs/moh_smoke.yaml
uv run ruff check .
uv run pytest -q
```

Smoke template dùng seed 42, tasks 4/6 thành phố, validation count 2, test count
1, GLS 3 iterations và FakeLLM ở cả hai tầng. `seed_attempts` giới hạn số lần
sinh seed; `seed_threshold` tùy chọn lọc seed theo gap (mặc định `None`). Baseline
hợp lệ vẫn được giữ nếu LLM không tìm được seed dưới ngưỡng. Synthetic cho 4–12 thành phố dùng
Held–Karp tính optimum thật; validation/test được sinh bằng seed riêng. Không
lấy nearest-neighbor làm optimum.

Dataset template `configs/moh_dataset.yaml` dùng file NPZ local. Thay đường dẫn
và chọn validation/test indices không giao nhau, không lặp và nằm trong range.
Loader kiểm tra toàn bộ references trước khi expose split. NPZ được đọc bằng
`allow_pickle=False` và có các mảng:

| Key | Shape | Nội dung |
| --- | --- | --- |
| `coordinates` | `(k, n, 2)` | Tọa độ thành phố |
| `distance_matrix` | `(k, n, n)` | Khoảng cách gốc |
| `cost` | `(k,)` | Cost reference hữu hạn, dương |
| `optimal_tour` | `(k, n)` | Tour reference integer, mỗi thành phố một lần |

Shape, finite values, tính đối xứng, tour và cost tham chiếu phải nhất quán.
Kiểm tra này xác minh tour/cost reference; với task lớn, nó không tự chứng minh
reference là optimum toàn cục. Chất lượng benchmark còn phụ thuộc provenance.

Converter cho dữ liệu pickle upstream là opt-in riêng:

```bash
uv run python tools/convert_moh_dataset.py --trusted-pickle /path/input.pkl --output /path/output.npz
```

**`pickle.load` có thể thực thi mã trong process converter.** Chỉ dùng file local
đã tin cậy; validation sau load không bảo vệ khỏi pickle độc hại. Converter
nhận dict có `coordinate`, `distance_matrix`, `cost`, `optimal_tour`, đổi key
`coordinate` thành `coordinates`, kiểm tra dữ liệu và từ chối ghi đè output.
Loader thí nghiệm không tự load pickle. Repo không kèm hoặc tự tải dataset
upstream; dataset template không phải bằng chứng benchmark đã chạy.

Held-out test được chạy sau tìm kiếm trên heuristic chọn theo validation;
score test không quay lại callback search hay quần thể. Lỗi ghi artifacts,
cấu hình hoặc dữ liệu phải được báo là lỗi hạ tầng, không đổi thành penalty.

## Deadline, ngân sách, tái lập và artifacts

Optimizer source và heuristic source, kể cả module-level statements, chỉ được
compile/exec trong worker. Process cha giữ LLM, credentials, quần thể, budget
và ledger đánh giá. JSON callback có ID và schema kiểm tra; stdout/stderr là
output riêng, không phải kênh fitness. Source/request/response/output/batch và
số callback có giới hạn.

Deadline của outer giới hạn cả callback inner và GLS. Supervisor quản lý scope
để dọn các worker do cha tạo trong invocation, kể cả các worker ở process group
khác. Adapter LLM dùng thời gian còn lại cho timeout và retry sleeps. Batch gọi
LLM tuần tự theo input order; kiểm tra budget trước công việc mới và giữ counts
cả ứng viên lỗi. Ngân sách GLS là số vòng cố định; timeout là failed, không lấy
lời giải tại thời điểm máy tình cờ hết giờ làm một kết quả tái lập.

Timeout của một ứng viên cho phép outer dùng best đã chấm để tiếp tục. Hết
deadline toàn search trả kết quả failed; source và checkpoint đã ghi vẫn còn
để audit. Sau khi search hoàn tất, held-out hết ngân sách hoặc deadline giữ
nguyên winner và trạng thái search, đồng thời ghi `test_status="incomplete"`
trong `run.json`. Điểm test không cập nhật quần thể validation.

RNG được dẫn xuất từ root seed theo task/split/instance/search scope; Python và
NumPy trong worker được seed. FakeLLM có cursor độc lập cho directions,
heuristic programs và optimizer programs. Response mặc định là mẫu offline;
FakeLLM không suy luận hay chứng minh chất lượng prompt. API provider thật
không có bảo đảm response giống nhau với cùng seed.

Artifacts schema 2 gồm `config.yaml`, `run.json`, `events.jsonl`, code
`heuristics/<id>.py`, `optimizers/<id>.py` và checkpoint trong `populations/`.
Đọc config/provenance trước, sau đó source và các events liên quan đánh giá,
commit/rollback, active optimizer và global best. Khi so hai FakeLLM runs, so
source, score, quyết định và thứ tự semantic events; loại tên run directory,
thời lượng và metadata máy. Artifacts là dữ liệu audit, chưa có CLI replay.

## Đối chiếu upstream và giới hạn fidelity

Nguồn đối chiếu: checkout read-only `/home/bonxom/Code/MoH`, commit
`e8c6154911d8182f2a53bcc17b11be0a063a53de`. MIT notice và chi tiết adaptation ở
[third_party/moh](../third_party/moh/README.md).

| Upstream | Module tương ứng trong `src/moh/` | Vai trò |
| --- | --- | --- |
| `main.py`, `moh.py` | `main.py`, `experiments/program_search.py`, `optimizers/program_meta.py`, `program_inner.py` | CLI, ghép hai tầng, orchestration |
| `utils/population.py` (`Pop`) | `core/program_population.py`, `execution/optimizer_protocol.py` | Population cha và snapshot worker |
| `problems/meta/seed_algorithm.py` | `optimizers/seeds/basic.py` | Directions → batch → evaluate → min |
| `problems/meta/seed_algorithm_improved.py` | `optimizers/seeds/multi_temperature.py` | Nhiều temperature, cache, chọn score thấp |
| `utils/utils.py` | `optimizers/helpers.py` | Extract idea/code, không thực thi |
| `problems/tsp_gls/gls.py` | `problems/tsp_gls/{tour,local_search,solver}.py` | Tour/route, local search và GLS |
| `problems/tsp_gls/eval.py` | `problems/tsp_gls/evaluation.py`, `execution/gls_{runner,worker}.py` | Evaluate, tour/cost/gap validation |
| `problems/tsp_gls/gpt.py` | `problems/tsp_gls/baselines.py` | Baseline upstream |
| `prompts/meta/desc.txt`, `prompts/tsp_gls/{desc,task,plan,size}.txt` | `prompts/{program_optimizer,gls_heuristic}.py`, bundled seeds | Giao diện và guidance viết lại |
| `utils/llm_client/`, shared `gpt.py` | `llm/`, `execution/optimizer_worker.py` | Provider adapter và RPC thay global state |
| `utils/run_logger.py` | `experiments/artifacts.py`, `logging.py` | Artifacts/source/lineage |

Các thay đổi có chủ ý:

- Thêm NumPy import còn thiếu trong baseline upstream.
- Sửa seed mở rộng xếp `reverse=True` thành thứ tự tăng dần để minimize; batch
  giới hạn theo façade trước khi gọi LLM, helpers từ chối fence mơ hồ.
- Thay test mode upstream `n_tests=10` nhưng sample 5 index bằng explicit split
  đầy đủ và không giao nhau; không giữ sampling ngầm.
- Dời main-process `exec` sang worker, thay kiểm tra số vòng bằng LLM và đọc
  stdout làm điểm bằng deadline thật và structured protocol.
- Xác minh score/source trong cha; dùng transaction rollback và tách active
  khỏi global best, tránh giữ fitness của source khác khi nhận optimizer.
- Dùng Python/NumPy thay Numba và fixed iterations thay 20 giây wall-clock;
  strict input/output checks và sửa mutable alias/tie handling theo attribution.

Đây là reproduction giao diện, luồng thuật toán và cơ chế tự cải tiến phục vụ
đọc code/nghiên cứu. Không tuyên bố tương đương tốc độ Numba, ngân sách evaluator,
chất lượng dữ liệu hoặc kết quả paper. Fixture local search đã được đối chiếu
upstream; đối chiếu toàn bộ guided solver với cùng profile vẫn là giới hạn
coverage. Không chạy campaign LLM trả phí trong triển khai này.

Worker giúp chứa crash, exception và treo trong thí nghiệm local. Nó **không
phải OS sandbox cho mã thù địch**: Python sinh ra vẫn có thể truy cập filesystem
và network; process-group cleanup không ngăn mọi hành vi thoát group. Muốn chạy
mã adversarial cần isolation mạnh hơn. Các tests dùng FakeLLM/provider transport
stubs và không gọi API LLM ngoài.
