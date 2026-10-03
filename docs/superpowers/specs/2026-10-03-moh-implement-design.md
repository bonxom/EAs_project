# MoH-implement: tái hiện cơ chế MoH với cấu trúc dễ đọc

Ngày: 2026-10-03. Nhánh: `MoH-implement`.

## 1. Mục tiêu và cơ sở thiết kế

Người dùng yêu cầu tái hiện project `/home/bonxom/Code/MoH`, đồng thời tổ chức
thư mục cho dễ đọc. Hướng thiết kế đã được duyệt trong hội thoại: dùng TSP-GLS,
tìm kiếm mã Python của optimizer, và tách optimizer, problem, execution, prompt
và LLM thành các thành phần có trách nhiệm rõ ràng.

Nguồn đối chiếu là bản checkout local của repo gốc, đặc biệt:

- `moh.py`: `get_improver`, `meta_utility`, `run_meta_optimizer`.
- `problems/meta/seed_algorithm.py` và `seed_algorithm_improved.py`.
- `problems/tsp_gls/eval.py` và `gls.py`.
- `prompts/meta/desc.txt`, `prompts/tsp_gls/desc.txt` và `utils/population.py`.

Đây là tái hiện cơ chế của bản checkout đó, chưa phải xác nhận tái lập kết quả
paper. README upstream nói code đã được refactor; không suy ra mọi chi tiết
của checkout giống hệt mọi thí nghiệm trong paper.

Các ràng buộc từ `AGENTS.md` vẫn áp dụng: Python 3.12, uv, seed, tests dùng
FakeLLM, mã sinh ra không chạy trong process chính, timeout, ứng viên lỗi
không làm hỏng thí nghiệm và adapter độc lập provider.

## 2. Các lựa chọn và quyết định

Ba hướng đã được cân nhắc:

1. Chép nguyên repo gốc: nhanh nhưng giữ `exec()` optimizer trong process chính,
   global state và các trách nhiệm ghép trong `moh.py`; không đáp ứng quy tắc repo.
2. Giữ JSON `OptimizerSpec` và chỉ thêm GLS: ít thay đổi nhưng bỏ mất cơ chế
   tìm kiếm và tự cải tiến chương trình optimizer.
3. Tái hiện các giao diện chương trình và luồng thuật toán, tách việc thực thi
   qua process con và callback JSON: chọn hướng này.

Không sửa bản checkout `/home/bonxom/Code/MoH`. Các thuật toán và prompt được
adapt từ upstream phải giữ attribution và giấy phép MIT trong
`third_party/moh/LICENSE` cùng ghi chú nguồn và revision nếu có.

## 3. Hai tầng tối ưu

### Inner: chạy chương trình optimizer để cải tiến heuristic

Một heuristic là chương trình định nghĩa:

```python
def update_edge_distance(edge_distance, local_opt_tour, edge_n_used):
    return updated_edge_distance
```

Đầu vào là ma trận khoảng cách gốc, tour hiện tại không lặp điểm cuối, và ma
trận đếm penalty đối xứng. Heuristic trả về ma trận khoảng cách được điều chỉnh
để dẫn hướng perturbation trong GLS. Không dùng heuristic này để tính cost thật.

Một optimizer là chương trình định nghĩa:

```python
def improve_algorithm(population, utility, language_model, function_format, task):
    return best_idea, best_solution, best_utility
```

`HeuristicOptimizer` điều phối chạy chương trình trên quần thể heuristic của
một task. Chương trình tự chọn cha, xây prompt, gọi LLM và đánh giá các heuristic
nó sinh ra. Không đóng khung lựa chọn bằng `OptimizerSpec`.

Hai chương trình optimizer ban đầu được adapt từ hai seed upstream: seed cơ bản
sinh directions rồi sinh batch ứng viên; seed mở rộng thử các temperature và
cache ứng viên. Các sửa lỗi của seed được ghi rõ, không âm thầm giữ hành vi
chọn ứng viên điểm cao khi mục tiêu là minimize.

### Outer: dùng optimizer để cải tiến chương trình optimizer

`MetaOptimizer` có quần thể chương trình optimizer được chấm điểm và một
chương trình đang hoạt động. Nó chạy chương trình đang hoạt động với:

- `population`: quần thể optimizer;
- `function_format`: mô tả chữ ký `improve_algorithm` và các callback;
- `language_model`: LLM ở tầng meta;
- `utility`: callback đánh giá một chương trình optimizer trên các task TSP-GLS.

Callback này lần lượt chạy chương trình ứng viên ở inner trên từng task, với
LLM ở tầng heuristic. Vì vậy một optimizer worker ở outer có thể kích hoạt các
optimizer worker ở inner; việc điều phối và kiểm soát ngân sách thuộc process cha.

Mỗi chương trình mới phải được đánh giá trước khi được đưa vào quần thể.
Chương trình được chọn để tiếp tục outer phải có đánh giá hợp lệ. Nếu đề xuất
không thành công, dùng chương trình tốt nhất đã chấm thay vì dừng thí nghiệm.
Chương trình hoạt động có thể thay đổi dù không cải thiện global best; chương
trình tốt nhất và điểm tốt nhất vẫn được lưu riêng, không bị ghi đè bởi ứng viên
kém hơn. Đây là phần tự cải tiến cần thể hiện rõ trong code và test.

## 4. Quần thể, khởi tạo và fitness

Quần thể giữ tối đa `capacity` ứng viên thành công, xếp theo score tăng dần;
tie-break theo ID ổn định. Cùng source không lưu lặp. Random selection dùng
RNG có seed với xác suất theo rank tương tự `Pop.get_random_solution` upstream.
Không dùng global RNG trong process cha.

Quần thể heuristic theo task tồn tại suốt run. Optimizer tiếp theo nhận snapshot
của quần thể đã được cập nhật từ các optimizer trước. Khởi tạo gồm baseline
được chấm điểm và một giai đoạn sinh seed bằng LLM có giới hạn attempts; không
dùng vòng lặp vô hạn để đạt threshold. Ngưỡng upstream có thể cấu hình, nhưng
không loại bỏ toàn bộ baseline hợp lệ nếu LLM không tìm được seed dưới ngưỡng.

Trong một lần chạy optimizer, các đánh giá callback được giữ trong kho kết quả
do cha quản lý. Một lần chạy trả về thành công mới cập nhật shared population
bằng các ứng viên thành công của lần đó. Lần chạy lỗi không commit các thay đổi
quần thể; số lần gọi và đánh giá đã tiêu thụ vẫn được ghi nhận. Seed generation
cũng chỉ cập nhật bằng ứng viên đã được chấm hợp lệ.

Score heuristic là mean gap % trên các validation instance:

```python
gap = max(0.0, (tour_cost / optimal_cost - 1.0) * 100.0)
```

Cost và tour được kiểm tra lại từ ma trận gốc ở cha. `optimal_cost` phải hữu hạn
và dương. Clamp chỉ xử lý sai số nhỏ; nếu cost nhỏ hơn nghiệm tham chiếu quá
tolerance thì báo dữ liệu tham chiếu không nhất quán thay vì che giấu lỗi.

Score optimizer là weighted mean của score heuristic được chọn trên mỗi task;
trọng số mặc định là kích thước task như upstream. Nếu một task không có kết
quả inner hợp lệ thì optimizer thất bại.

Callback tương thích upstream trả `1e6` cho đánh giá ứng viên thất bại; đây chỉ
là giá trị phạt của giao diện chương trình. Record thật vẫn có
`status="failed"`, `utility=None` và lỗi. Cha không chấp nhận ứng viên chỉ vì
worker khai báo điểm hữu hạn: source trả về phải có kết quả đánh giá thành công
trong kho callback của lần chạy, hoặc là một thành viên hợp lệ trong snapshot.
Điểm trả về được đối chiếu với điểm của cha. Không cho worker tự tạo fitness.

## 5. TSP-GLS và dữ liệu

Solver adapt nearest-neighbor initialization, local search với 2-opt/relocate,
guided perturbation, edge penalties và reset về best route sau mỗi 50 vòng
từ upstream. Giữ route representation predecessor/successor để giảm sai khác
thuật toán; các phép chuyển tour được đặt trong `tour.py`.

Mỗi vòng gọi heuristic trên bản sao dữ liệu. Kiểm tra ma trận kết quả có đúng
shape, kiểu số thực hữu hạn, không âm và đối xứng trong tolerance; lỗi trả về
làm thất bại lần đánh giá. Best cost luôn tính trên distance matrix gốc.
Chương trình heuristic và toàn bộ solver chạy cùng một worker cho mỗi instance,
không mở process mới cho mỗi lần gọi hàm cập nhật cạnh.

Số GLS iterations, perturbation moves, số cạnh chọn và kích thước neighborhood
đều được ghi trong config. Dùng ngân sách vòng lặp cố định để smoke run tái lập;
deadline process là giới hạn chống treo. Run chạm deadline là failed, không
trả lời giải phụ thuộc tốc độ máy rồi gọi đó là kết quả seed-reproducible.
Profile upstream có thể dùng 1000 vòng và neighborhood 100, nhưng không tuyên bố
tương đương ngân sách wall-clock 20 giây của evaluator gốc.

Hai đường dữ liệu:

- Smoke offline: tọa độ sinh bằng seed, tối đa 12 thành phố, tính optimum bằng
  Held–Karp có giới hạn kích thước. Không dùng nearest-neighbor làm "optimum".
- Dataset: file NPZ với `allow_pickle=False`, có coordinates, distance matrices,
  optimal costs và optimal tours. Kiểm tra shape, finite values, tính đối xứng,
  tour và cost tham chiếu nhất quán. Validation và test split được cấu hình
  bằng chỉ số xác định, không chọn mẫu ngẫu nhiên ngầm.

Checkout upstream hiện không chứa dataset. Không tự tải dữ liệu từ tài khoản
ngoài. Cung cấp công cụ chuyển đổi pickle upstream sang NPZ chỉ chạy khi người
dùng chủ động chỉ định file local đáng tin cậy; giải thích pickle thực thi mã
khi đọc. Các tests không cần dataset upstream hoặc network.

Held-out test chạy sau tìm kiếm, trên heuristic được chọn theo validation; điểm
test không quay lại quần thể. Với smoke, tạo tập validation/test khác seed và
lưu provenance. Dataset mode yêu cầu hai tập index không giao nhau.

## 6. Worker và callback protocol

Optimizer code được compile và exec chỉ trong optimizer worker. Worker nhận
snapshot quần thể và cung cấp façade có các phương thức upstream cần:

- `get_solution_by_index`, `get_best_solution`, `get_random_solution`,
  `get_subtask_size`, `get_population`;
- `language_model.prompt(expertise, message, temperature)` và
  `prompt_batch(expertise, messages, temperature)`;
- `utility(source_code, idea=None, problem_type=None)`.

Các entry có `best_sol`, `idea`, `utility`. Worker dùng helpers ở module
`moh.optimizers.helpers` để extract code/idea; prompt và seed được cập nhật
import path, không tạo package `utils` toàn cục giả.

LLM và evaluation façade gửi request JSON đến cha bằng pipe riêng, với request
ID tăng dần. Cha chỉ xử lý các operation cho phép: `llm_prompt`, `llm_batch`,
`evaluate`, `finish`. `evaluate` chỉ được trỏ đến task hiện tại của inner, hoặc
`meta-optimizer` trong outer; không dùng tùy ý task name do worker cung cấp.

Cha giữ LLM client và credentials. Worker không nhận API key hoặc environment
của cha. Batch được xử lý theo thứ tự input để FakeLLM và event ordering ổn
định. Adapter hỗ trợ role/temperature ở provider boundary; không đưa logic
OpenAI vào optimizer, runner hoặc solver.

Protocol giới hạn source, request/result, stdout/stderr, batch size và số
callback. JSON từ worker phải từ chối duplicate keys, nonfinite values, ID sai,
operation lạ và dữ liệu sai schema. Output print không thể giả mạo callback.

Mỗi optimizer invocation có wall-clock deadline; request LLM và các worker con
nhận deadline còn lại. Khi deadline ngoài hết, không tiếp tục callback lồng;
dọn tất cả worker thuộc invocation, kể cả inner/heuristic worker mà cha đã tạo.
Không chỉ kill process group của worker ngoài vì process con do cha tạo có thể
ở group khác. Giới hạn toàn run gồm số LLM calls và heuristic evaluations;
kiểm tra trước khi gọi, batch tính từng response, không cho một callback vượt
ngân sách. Có depth cố định outer → inner → heuristic, không có meta recursion
không giới hạn.

Syntax error, missing function, exception, invalid return, timeout, protocol
error và budget exhaustion là lỗi ứng viên được ghi nhận và phục hồi. Lỗi
ghi artifact, dữ liệu và cấu hình là lỗi hạ tầng; kết thúc run với thông báo
rõ thay vì âm thầm đổi thành fitness xấu. Đây là crash/timeout containment,
không phải OS sandbox cho mã thù địch.

## 7. Cấu trúc thư mục và khả năng đọc

```text
src/moh/
  main.py                          # CLI chọn chế độ chạy
  experiment.py                    # Entry points ghép các thành phần
  config.py                        # Cấu hình mini-MoH đang có
  program_config.py                # Cấu hình tái hiện bằng chương trình
  core/
    models.py                     # Records mini-MoH hiện tại
    programs.py                   # OptimizerProgram, GapEvaluation, RunResult
    program_population.py         # Quần thể minimize và selection theo rank
    seeds.py                      # Derive seed dùng chung
  optimizers/
    inner.py                      # Luồng mini-MoH cũ và giao diện inner
    program_inner.py              # Chạy chương trình tìm heuristic
    meta.py                       # MetaOptimizer cũ
    program_meta.py               # Tự cải tiến chương trình optimizer
    helpers.py                    # Extract code và idea
    seeds/
      basic.py                    # Seed optimizer cơ bản
      multi_temperature.py        # Seed optimizer mở rộng
  problems/tsp_gls/
    task.py                       # Task, instance và context
    dataset.py                    # NPZ loader và synthetic fixtures
    exact.py                      # Held–Karp cho smoke nhỏ
    tour.py                       # Tour/route representation, validation, cost
    local_search.py               # 2-opt và relocate adapt upstream
    solver.py                     # Vòng Guided Local Search
    evaluation.py                 # Trusted scoring và aggregation
    baselines.py                  # Heuristic chương trình ban đầu
  execution/
    process.py                    # Lifecycle, deadline, bounded pipes
    optimizer_runner.py           # Parent dispatch callback
    optimizer_worker.py           # Execute improve_algorithm trong child
    optimizer_protocol.py         # RPC schema và worker façades
    gls_runner.py                 # Chạy và kiểm tra kết quả GLS
    gls_worker.py                  # Execute heuristic và solver trong child
  prompts/
    program_optimizer.py          # Function format và guidance meta
    gls_heuristic.py               # Format heuristic và seed prompts
  llm/
    base.py                       # Provider-independent request contract
    fake.py                       # Scripted deterministic responses
    recording.py                  # Prompt/response và counts
    openai_client.py              # SDK, credentials và temperature mapping
  experiments/
    program_search.py             # Composition root của chế độ MoH mới
    artifacts.py                  # Lưu .py optimizer, lineage, checkpoints
```

Giữ chế độ mini-MoH hiện tại để đọc đối chiếu và giữ các tests đã có. Chế độ mới
dùng CLI `--mode moh --config configs/moh_smoke.yaml`; chưa đổi default CLI.
Các modules mới có docstring nêu nhiệm vụ và giao diện. README tiếng Việt chỉ
rõ thứ tự đọc: CLI → experiment → meta → inner → runner → GLS. Core records mới
không tái sử dụng invariant negative mean length của `EvaluationResult` cũ.

## 8. Cấu hình, artifacts và reproducibility

Hai cấu hình mới: `moh_smoke.yaml` chạy FakeLLM với instance nhỏ, và
`moh_dataset.yaml` làm mẫu dùng dữ liệu local 100/200 thành phố, cấu hình riêng
cho heuristic LLM và meta LLM. Không tự gọi API thật khi chạy tests hoặc smoke.

Config ghi seed, split, task weights, population capacities, số outer rounds,
seed attempts, GLS iterations, request limits, deadlines và ngân sách calls.
Validate tất cả trước khi tạo artifacts hoặc gọi LLM. Seed derivation phân biệt
khởi tạo, population selection, worker, task, round và LLM scope.

Artifacts gồm config, dataset hash/provenance, upstream attribution/revision,
optimizer `.py`, heuristic `.py`, populations/checkpoints, validation/test
results, calls/counts, nguồn đề xuất, optimizer đang hoạt động và global best.
Events lưu request/response đã qua kiểm tra của cha; không dump environment.
Counts tính cả công việc của ứng viên thất bại và callbacks nested, không tính
gấp đôi. Lưu các transition population trước/sau để giải thích ảnh hưởng của
thứ tự đánh giá trong shared-state protocol.

Hai run FakeLLM cùng config và seed phải tái lập source, score, quần thể,
transition và event ordering; tên thư mục, thời lượng và metadata máy không
nằm trong phép so sánh. Live LLM không bảo đảm tái lập theo seed; giữ response
để audit, không tuyên bố bảo đảm mà adapter không cung cấp.

## 9. Sửa lỗi có chủ ý và phạm vi giới hạn

Ghi rõ trong tài liệu các thay đổi so với checkout upstream:

- Di chuyển toàn bộ `exec` mã sinh ra sang worker, bỏ shared `gpt.py`.
- Dùng structured protocol thay vì đọc dòng cuối stdout làm điểm.
- Không dùng LLM kiểm tra số vòng làm cơ chế timeout; áp dụng deadline thực.
- Chọn ứng viên theo minimize; sửa `reverse=True` trong seed mở rộng.
- Sửa test mode upstream dùng `n_tests=10` nhưng chỉ sample 5 index.
- Seed và nguồn dữ liệu/split được ghi rõ; synthetic optimum chỉ cho smoke nhỏ.
- Cập nhật quần thể theo transaction; không giữ điểm/source lệch nhau trong
  bước accept optimizer; global best và active optimizer là hai records riêng.
- Ngân sách cố định GLS và strict input/output validation có thể tạo sai khác
  với evaluator upstream; không quảng bá các kết quả như benchmark tương đương.

Không bao gồm tải dataset từ Drive, chạy chiến dịch LLM trả phí, benchmarking
paper, nhiều bài toán ngoài TSP-GLS, distributed execution hoặc hostile-code
security sandbox. Việc triển khai hoàn thành phải có pipeline chạy thật offline,
không chỉ scaffold hoặc mock solver.

## 10. Tiêu chí nghiệm thu

1. CLI mới chạy end-to-end với FakeLLM: optimizer source → callback sinh
   heuristic → GLS worker → gap validation → optimizer score → optimizer mới
   được dùng trong vòng tiếp theo → held-out evaluation → artifacts.
2. Chứng minh self-improvement bằng test có hai chương trình optimizer khác
   nhau và trace xác nhận vòng tiếp theo thực thi chương trình vừa chọn.
3. Tests cho nearest neighbor, route conversion, 2-opt/relocate, local search,
   penalty update và cost trên distance gốc; fixture nhỏ đối chiếu adapted GLS
   với upstream cho cùng cấu hình vòng lặp khi có thể.
4. Tests cho dữ liệu sai, optimum sai, gap formula, weights theo size, splits
   không giao nhau và exact optimum cho fixture đã biết.
5. Tests thực sự chạy worker: syntax, module-level loop, exception, timeout,
   stdout flood, JSON malformed, fabricated fitness, source chưa chấm, batch
   vượt limit, budget exhaustion và cleanup worker lồng. Một ứng viên lỗi
   không làm dừng vòng tiếp theo và không cập nhật shared population.
6. Tests chỉ dùng FakeLLM/provider transport stubs; không external APIs.
7. Hai smoke runs cùng seed cho kết quả và semantic events bằng nhau; artifacts
   có code optimizer thực sự chạy và mọi ứng viên đã chấm.
8. Các tests mini-MoH cũ tiếp tục đạt. Chạy `uv run ruff check .` và
   `uv run pytest -q` trước khi hoàn thành.
9. README giải thích hai tầng, tree, lệnh smoke, dataset conversion và các sai
   khác với upstream; retain MIT attribution cho các phần adapt.

## 11. Trạng thái và bước tiếp theo

Hướng thiết kế được duyệt; tài liệu này là spec để người dùng review. Chưa sửa
product code. Sau khi spec được duyệt, viết implementation plan chia theo
interfaces và tests, rồi chọn cách thực thi trước khi triển khai.
