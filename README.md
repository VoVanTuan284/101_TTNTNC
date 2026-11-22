# 101_TTNTNC — Hệ Thống Game AI Cờ Vua Đa Cấp Độ ($8 \times 8$ FIDE)

**Môn học:** Trí tuệ Nhân tạo Nâng cao  
**Chủ đề:** Xây dựng và Đánh giá Game AI Đa Cấp Độ  
**Trò chơi:** Cờ vua chuẩn FIDE $8 \times 8$ _(Đề tài và phạm vi 8 cấp độ đã được giảng viên phê duyệt chính thức)_  
**Phiên bản Python chuẩn:** Python $\ge 3.11$ (ghim tại [`.python-version`](.python-version) và [`pyproject.toml`](pyproject.toml))  
**Quy mô nhóm:** 3 Thành viên (`TV1`, `TV2`, `TV3`) | Phân công đóng góp: $33.3\% - 33.3\% - 33.4\%$ (chi tiết tại [`docs/CONTRIBUTIONS.md`](docs/CONTRIBUTIONS.md))

---

## 1. Tổng Quan Dự Án & Kiến Trúc 8 Cấp Độ Tác Tử

Dự án hiện thực hóa một hệ thống Game AI Cờ vua chuẩn FIDE $8 \times 8$ đóng gói trong package trung tâm `chess_ai/` theo kiến trúc sạch (Clean / Hexagonal Architecture), bao gồm đầy đủ 8 cấp độ tác tử:

|  Cấp độ   | Tên Tác Tử          | Phân loại | Phương pháp & Đặc trưng Kỹ thuật Chính                                                                                                                   |
| :-------: | :------------------ | :-------: | :------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Cấp 1** | `Level1_Random`     | Bắt buộc  | Chọn ngẫu nhiên đồng đều từ `legal_actions()`, làm mốc đối chứng sàn (Baseline).                                                                         |
| **Cấp 2** | `Level2_Heuristic`  | Bắt buộc  | Luật ưu tiên 3 tầng (Mate-in-1 $\to$ Phòng thủ & MVV-LVA $\to$ PST Michniewski + King Safety).                                                           |
| **Cấp 3** | `Level3_AlphaBeta`  | Bắt buộc  | Iterative Deepening Negamax $\alpha$-$\beta$ + Move Ordering + Zobrist TT + Quiescence Search.                                                           |
| **Cấp 4** | `Level4_MCTS_UCT`   | Bắt buộc  | Vanilla MCTS (UCB1/UCT) với mô phỏng cắt ngưỡng (`max_rollout_depth: 60`) và định giá tĩnh.                                                              |
| **Cấp 5** | `Level5_Supervised` | Bắt buộc  | Mạng Dual-Head ResNet (6 blocks, 64 filters) đầu vào Tensor `(18, 8, 8)` góc nhìn bên đi, huấn luyện từ **Lichess Elite Database** có khử trùng Zobrist. |
| **Cấp 6** | `Level6_RL_KRK`     |  Mở rộng  | Học tăng cường Tabular Q-Learning & Policy Gradient (REINFORCE) trên cờ tàn chuẩn FIDE **KRK**.                                                          |
| **Cấp 7** | `Level7_GuidedMCTS` |  Mở rộng  | AlphaGo-inspired Guided MCTS (PUCT) tích hợp Policy Prior $P(s, a)$ và Value Head $V(s)$ của Cấp 5.                                                      |
| **Cấp 8** | `Level8_AlphaZero`  |  Mở rộng  | Vòng lặp Self-Play PUCT + Replay Buffer + Checkpoint Arena Gating (ngưỡng thắng $55\%$).                                                                 |

---

## 2. Cấu Trúc Thư Mục & Tài Liệu Đặc Tả (`configs/`, `chess_ai/`, `resources/`, `results/`, `docs/`)

- **Cấu hình trung tâm (`configs/` — Single Source of Truth):**
  - [`configs/default_config.yaml`](configs/default_config.yaml): Đường dẫn, 10 Random Seeds chuẩn, cấu hình ETL Lichess Elite, Arena 50 khai cuộc & Logging.
  - [`configs/model_resnet.yaml`](configs/model_resnet.yaml): Đặc tả Tensor `(18, 8, 8)` lật bàn cờ theo bên đi, kiến trúc Dual-Head ResNet & Schema Checkpoint.
  - [`configs/search_params.yaml`](configs/search_params.yaml): Bảng điểm PST Michniewski, MVV-LVA, siêu tham số Alpha-Beta, UCT, PUCT, RL (Cấp 6) & Self-Play (Cấp 8).
- **Tài nguyên thế cờ chuẩn (`resources/` — được commit):**
  - [`resources/benchmark_positions.epd`](resources/benchmark_positions.epd): Bộ 50 thế cờ chiến thuật chuẩn Win At Chess (`WAC.001` – `WAC.050`).
  - [`resources/openings_50.epd`](resources/openings_50.epd): Bộ 50 thế cờ khai cuộc cân bằng (6–8 plies) phục vụ giải đấu Arena.
- **Kết quả thực nghiệm & Trọng số nộp bài (`results/`, `checkpoints/`, `report/` — được commit):**
  - `checkpoints/`: Lưu trữ trực tiếp `best_model.pt` ($\sim 5\text{MB}$) và `l6_krk_qtable.pkl` trên Git.
  - `results/`: Lưu trữ log thi đấu thô (`round_robin/matches.jsonl`, `round_robin/games.pgn`), kết quả `ablation1/`, `ablation2/`, `training/` và biểu đồ `figures/`.
  - `report/`: Chứa báo cáo 5 chương, slide thuyết trình, poster A0 và link video demo.
- **Tài liệu kỹ thuật (`docs/`):**
  - [`docs/TTNTNC_De_tai_cuoi_ky_Game_AI.pdf`](docs/TTNTNC_De_tai_cuoi_ky_Game_AI.pdf): Đề bài gốc và Rubric chấm điểm môn Trí tuệ Nhân tạo Nâng cao.
  - [`docs/DECISIONS.md`](docs/DECISIONS.md): Hồ sơ các quyết định kiến trúc & kỹ thuật đã đóng băng (`ADR-01` $\to$ `ADR-08`).
  - [`docs/CONTRIBUTIONS.md`](docs/CONTRIBUTIONS.md): Bảng phân công nhiệm vụ, nhánh Git và tỷ lệ đóng góp (`33.3% / 33.3% / 33.4%`) của 3 thành viên.
  - [`docs/PROMPT_AND_FILE_SPEC.md`](docs/PROMPT_AND_FILE_SPEC.md): Đặc tả kiến trúc Clean Architecture, 8 cấp độ tác tử và hợp đồng giao diện `GameEnv` / `ChessEnv` / `BaseAgent`.
  - [`docs/DATASET_SPEC.md`](docs/DATASET_SPEC.md): Quy trình ETL dữ liệu Lichess Elite Database, bộ lọc 5 tiêu chí, khử trùng Zobrist 64-bit, lưu trữ `np.memmap` và schema checkpoint.
  - [`docs/EXPERIMENT_PROTOCOL.md`](docs/EXPERIMENT_PROTOCOL.md): Giao thức giải đấu Round-Robin 100 ván/cặp trên 50 khai cuộc, kiểm định Wilson/Bootstrap 95% và thiết kế 2 bài Ablation Studies.
  - [`docs/UI_SPEC.md`](docs/UI_SPEC.md): Đặc tả giao diện Desktop Pygame (bố cục 3 cột, đồng hồ 2 đầu bàn cờ, Modal thiết lập ván đấu mới, Tab Nghiên cứu Thực nghiệm & bộ quân cờ `assets/`).
  - [`docs/GITFLOW.md`](docs/GITFLOW.md): Quy trình chia nhánh, chuẩn Conventional Commits, quy định Pull Request và quản lý dữ liệu lớn.
  - [`docs/HANDOFF_AND_QA.md`](docs/HANDOFF_AND_QA.md): Quy ước bàn giao giữa các thành viên và danh mục 61 kịch bản kiểm thử tự động (`TC01`–`TC61`).
  - [`docs/SPRINT_PLAN_15DAYS.md`](docs/SPRINT_PLAN_15DAYS.md): Ma trận trách nhiệm RACI và lộ trình triển khai chi tiết 15 ngày theo 5 pha.
  - [`docs/REPORT_OUTLINE.md`](docs/REPORT_OUTLINE.md): 4 câu hỏi nghiên cứu (`RQ1`–`RQ4`) và dàn ý chi tiết Báo cáo khoa học 5 chương chuẩn IEEE.

---

## 3. Cài Đặt Môi Trường (`.venv`) & Hướng Dẫn Thực Thi Nhanh

> **Lưu ý làm việc nhóm:** Thư mục `.venv/` đã được chặn trong [`.gitignore`](.gitignore) để mỗi thành viên tự tạo môi trường cục bộ theo hệ điều hành của mình mà không xung đột Git.

### 3.1. Khởi tạo môi trường ảo `.venv` (Python 3.11)

**Cách A — Dùng `uv` (Khuyên dùng, tự động tải đúng Python 3.11 theo `.python-version`):**

```powershell
# Windows PowerShell
uv venv --python 3.11 .venv
.\.venv\Scripts\Activate.ps1
uv pip install -e .
```

**Cách B — Dùng `venv + pip` mặc định (Windows / macOS / Linux):**

```powershell
# Windows PowerShell (dùng Python >= 3.11)
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
pip install -e .
```

```bash
# macOS / Linux
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

### 3.2. Các lệnh chạy chính (Quick Start CLI)

Sau khi kích hoạt `.venv`, có thể chạy trực tiếp bằng `python -m ...` hoặc các lệnh CLI đã đăng ký trong [`pyproject.toml`](pyproject.toml):

```bash
# 1. Chạy kiểm thử tự động (Core & L1-L4 hiện có 11 tests pass; khung 61 TCs của TV3)
pytest -v

# 2. Trích xuất dữ liệu từ Lichess Elite Database ra mảng nhị phân np.memmap
chess-etl --config configs/default_config.yaml
# (hoặc: python -m chess_ai.etl.pgn_etl --config configs/default_config.yaml)

# 3. Huấn luyện mạng Dual-Head ResNet (Cấp 5) và RL cờ tàn KRK (Cấp 6)
chess-train --config configs/model_resnet.yaml
python -m chess_ai.training.train_rl_krk --config configs/search_params.yaml

# 4. Vận hành giải đấu đối kháng Round-Robin (100 ván/cặp trên 50 thế cờ khai cuộc cân bằng)
chess-arena --config configs/default_config.yaml --budget-mode workload_control

# 5. Tái tạo bảng kết quả, khoảng tin cậy Wilson/Bootstrap 95% và biểu đồ báo cáo
python scripts/reproduce_experiment.py --from-logs --config configs/default_config.yaml

# 6. Khởi động giao diện đồ họa Pygame (Người vs Người, Người vs Máy, Máy vs Máy & Research Lab)
chess-ui
# (hoặc: python -m chess_ai.ui.desktop_pygame)
```

### 3.3. UI Pygame: Người vs AI và replay

Lớp giao diện nằm tại [`chess_ai/ui/desktop_pygame.py`](chess_ai/ui/desktop_pygame.py).
UI không chứa luật cờ hoặc thuật toán tìm kiếm: bàn cờ được dựng từ `ChessEnv.render("fen")`,
nước đi được lọc từ `legal_actions()` và thực thi bằng `step()`. AI nhận một `clone()` của
môi trường trong worker riêng, sau đó UI áp dụng nước đi vào môi trường chính.

```text
ChessPygameApp
  ├─ ChessEnv: trạng thái, legal actions, SAN, terminal/result
  ├─ BaseAgent: select_move(env_clone, time_limit, max_workload)
  └─ MoveRecord: action + snapshot clone + SAN + thời gian
```

Click một quân rồi click ô đích để đi. Panel hỗ trợ ván mới, đảo bàn, undo và replay.
Phím tắt: `N` (ván mới), `F` (đảo bàn), `U` (undo), `Home/Left/Right/End` (replay),
`P` (đổi phe), `A` (đổi cấp AI), `B` (đổi time limit/số mô phỏng).

---

## 4. Quy Trình Gitflow & Quy Định Nhóm

Toàn bộ quy chuẩn chia nhánh (`main`, `dev`, `feature/tv1-*`, `feature/tv2-*`, `feature/tv3-*`), quy tắc **Conventional Commits**, kiểm tra `pytest` trước khi tạo Pull Request và quy định loại trừ dữ liệu lớn được đặc tả chi tiết tại [`docs/GITFLOW.md`](docs/GITFLOW.md).

---

## 5. Quy Cách Đóng Gói Nộp Bài

Theo Mục 3.7 của đề bài, toàn bộ hồ sơ nộp cuối kỳ được đóng gói thành một tệp ZIP duy nhất:

```text
NhomXX_Chess_TTNTNC_HoTen_MaSV.zip
```
