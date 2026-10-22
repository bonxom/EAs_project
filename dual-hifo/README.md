# Dual-HiFo

## Ý tưởng

Dual-HiFo giải TSP bằng hai vòng tiến hoá lồng nhau:

- **Outer loop (meta-level)** tiến hoá các *improver* — hàm Python `improve_algorithm` do LLM viết. Mỗi improver nhận một quần thể heuristic TSP và sinh ra heuristic mới/tốt hơn. Cách này lấy từ MoH.
- **Inner loop (task-level)** tiến hoá trực tiếp các heuristic `select_next_node` cho TSP. Cách này lấy từ HiFo.

Mỗi vòng đều dùng kiến trúc HiFo: quần thể, các operator `i1/e1/e2/m1/m2/m3`, chọn parent theo rank, `InsightPool`, `Navigator` và population management. Khác biệt là inner loop nhận thêm offspring do improver tốt nhất của outer loop sinh ra ở mỗi generation.

Mỗi kích thước TSP được chạy độc lập. Fitness của heuristic là gap (%) so với optimal tour:

```text
gap = (tour_cost / optimal_cost - 1) * 100
```

## Kiến trúc

```text
run.py
  └── với mỗi size trong sizes:
        └── dual.run_one_size
              ├── datasets/tsp/TSP{size}.pkl  (n_instances đầu tiên)
              │
              ├── OUTER HiFoLoop
              │     quần thể: improve_algorithm(...)
              │     fitness: áp improver lên seed population,
              │              rồi evaluate heuristic nó sinh ra
              │
              └── best improver
                    ↓
                  INNER HiFoLoop
                        quần thể: select_next_node(...)
                        fitness: gap vs optimal
                        offspring: operator HiFo + best improver
```

| Thành phần | Vai trò |
|---|---|
| `run.py` | Đọc `.env`/config, tạo LLM client, chạy từng size, ghi summary và `results.md` |
| `dual_hifo/dual.py` | Ghép outer loop → best improver → inner loop, ghi log từng size |
| `dual_hifo/hifo_loop.py` | Một lớp HiFo dùng chung cho cả outer và inner |
| `dual_hifo/prompts.py` | Prompt cho `i1/e1/e2/m1/m2/m3` của TSP và meta-level |
| `dual_hifo/insight.py` | `InsightPool` và `Navigator` (explore/exploit/balanced) |
| `dual_hifo/tsp.py` | Load dataset cố định, chạy constructive greedy, tính gap |
| `dual_hifo/evaluator.py` | Evaluate heuristic và thực thi improver |
| `dual_hifo/llm.py` | Client OpenAI-compatible, mock LLM, logging wrapper |
| `dual_hifo/run_logger.py` | Ghi LLM calls, TSP results, generation, code, summary |
| `dual_hifo/population.py` | Quần thể giới hạn kích thước, sắp theo objective |

Outer loop chỉ đánh giá improver bằng **một lần áp dụng** lên seed population, không chạy một inner loop đầy đủ cho từng candidate. Nhờ vậy chi phí LLM của outer loop được giữ thấp; quá trình tìm kiếm chính diễn ra ở inner loop.

## Luồng chạy

1. `run.py` load `.env`, `config.yaml`, override từ CLI và tạo LLM client.
2. Với mỗi size:
   - Load `datasets/tsp/TSP{size}.pkl`, lấy `n_instances` instance đầu tiên.
   - Evaluate seed heuristic để có baseline.
3. **Outer loop**:
   - Khởi tạo quần thể từ `seeds/seed_improver.py`.
   - Mỗi generation: chọn operator theo `operator_weights`, chọn parent theo rank, gọi LLM để sinh improver mới.
   - Evaluate improver: áp nó lên seed population, lấy heuristic trả về và tính gap.
   - Giữ lại các improver tốt nhất trong `outer.pop_size`.
   - Kết thúc outer loop, lấy `best_improver`.
4. **Inner loop**:
   - Khởi tạo quần thể từ `seeds/seed_select_next_node.py`.
   - Mỗi generation:
     1. `Navigator` chọn regime (`explore`/`exploit`/`balanced`) và directive dựa trên lịch sử best và diversity.
     2. `InsightPool` chọn tối đa 3 design principle.
     3. Với từng operator được bật: chọn parent, dựng prompt, gọi LLM, extract code, evaluate và thêm vào quần thể nếu hợp lệ.
     4. Sinh thêm `meta_offspring` heuristic bằng `best_improver`.
     5. Cập nhật best/avg/diversity, trích xuất insight mới và ghi log generation.
   - Giữ lại các heuristic tốt nhất trong `inner.pop_size`.
5. Ghi `summary.json`, `results.md`, code của best improver/heuristic và toàn bộ log LLM/TSP.

## Yêu cầu

- Python >= 3.9
- Cài dependency:

```bash
pip install -r requirements.txt
```

## Dataset

Dataset TSP20/50/100/200 nằm trong `datasets/tsp/`:

| File | Số instance | Số node |
|---|---:|---:|
| `TSP20.pkl` | 1000 | 20 |
| `TSP50.pkl` | 1000 | 50 |
| `TSP100.pkl` | 1000 | 100 |
| `TSP200.pkl` | 128 | 200 |

Nguồn: repo HiFo-Prompt, thư mục `examples/user_tsp_gls/TestingData/`.
Nếu cần tải lại:

```bash
git clone --depth 1 https://github.com/Challenger-XJTU/HiFo-Prompt.git /tmp/HiFo-Prompt
mkdir -p datasets/tsp
cp /tmp/HiFo-Prompt/examples/user_tsp_gls/TestingData/TSP20.pkl datasets/tsp/
cp /tmp/HiFo-Prompt/examples/user_tsp_gls/TestingData/TSP50.pkl datasets/tsp/
cp /tmp/HiFo-Prompt/examples/user_tsp_gls/TestingData/TSP100.pkl datasets/tsp/
cp /tmp/HiFo-Prompt/examples/user_tsp_gls/TestingData/TSP200.pkl datasets/tsp/
```

Mỗi file là một pickle chứa `coordinate`, `distance_matrix`, `optimal_tour`, `cost`.
Chương trình dùng `cost` làm chi phí tối ưu để tính gap.

## Config

Tạo file `.env` từ mẫu:

```bash
cp .env.example .env
```

Điền endpoint, API key và model:

```env
LLM_BASE_URL=http://localhost:20128/v1
LLM_API_KEY=sk-...
LLM_MODEL=ag/gemini-3.7-flash-low
```

Các tham số còn lại sửa trong `config.yaml`:

| Khóa | Ý nghĩa |
|---|---|
| `sizes` | các kích thước TSP cần chạy |
| `n_instances` | số instance đầu tiên của mỗi file dataset (mặc định 100) |
| `neighbor_size` | số candidate truyền cho heuristic; `null` = tất cả node chưa thăm |
| `random_seed` | seed cho các operator tiến hoá |
| `outer` | `pop_size`, `n_pop`, `m`, `operators`, `operator_weights` |
| `inner` | `pop_size`, `n_pop`, `m`, `operators`, `operator_weights`, `meta_offspring` |
| `llm` | `model`, `base_url`, `api_key`, `temperature`, `timeout`, `max_retries` |
| `output_dir` | thư mục ghi kết quả |

Có thể truyền endpoint/key trực tiếp thay vì dùng `.env`:

```bash
python run.py \
  --endpoint http://localhost:20128/v1 \
  --api-key sk-... \
  --model ag/gemini-3.7-flash-low
```

## Chạy

```bash
# dùng cấu hình trong config.yaml
python run.py

# chạy offline, không cần API key
python run.py --mock

# chọn size và số instance
python run.py --sizes 20 50 --n-instances 8
```

Một số tuỳ chọn:

```text
--sizes 20 50 100 200
--n-instances N
--endpoint URL
--api-key KEY
--model MODEL
--outer-pop N --outer-gens N
--inner-pop N --inner-gens N
--output-dir DIR
--mock
--verbose
```

Kết quả được ghi vào `runs/<timestamp>_<mock|llm>/`:

```text
summary.json                 tổng hợp các size
results.md                   bảng gap và heuristic tốt nhất
size_<N>/
  summary.json               kết quả của size
  run.log                    log dạng text
  llm_calls.jsonl            prompt và response của mọi lần gọi LLM
  llm_outputs/               mỗi lần gọi LLM một file markdown
  tsp_results.jsonl          kết quả evaluate từng heuristic
  generations.jsonl          trạng thái quần thể qua các generation
  outer_population.json
  inner_population.json
  code/best_improver.py
  code/best_heuristic.py
```
