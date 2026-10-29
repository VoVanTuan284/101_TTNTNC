# 101_TTNTNC — Hệ Thống Game AI Cờ Vua Đa Cấp Độ ($8 \times 8$ FIDE)

**Môn học:** Trí tuệ Nhân tạo Nâng cao  
**Chủ đề:** Xây dựng và Đánh giá Game AI Đa Cấp Độ  
**Trò chơi:** Cờ vua chuẩn FIDE $8 \times 8$ *(Đề tài và phạm vi 8 cấp độ đã được giảng viên phê duyệt chính thức)*  
**Phiên bản Python chuẩn:** Python $\ge 3.11$ (ghim tại [`.python-version`](.python-version) và [`pyproject.toml`](pyproject.toml))  
**Quy mô nhóm:** 3 Thành viên (`TV1`, `TV2`, `TV3`) | Phân công đóng góp: $33.3\% - 33.3\% - 33.4\%$ (chi tiết tại [`docs/CONTRIBUTIONS.md`](docs/CONTRIBUTIONS.md))  

---

## 1. Tổng Quan Dự Án & Kiến Trúc 8 Cấp Độ Tác Tử

Dự án hiện thực hóa một hệ thống Game AI Cờ vua chuẩn FIDE $8 \times 8$ đóng gói trong package trung tâm `chess_ai/` theo kiến trúc sạch (Clean / Hexagonal Architecture), bao gồm đầy đủ 8 cấp độ tác tử:

| Cấp độ | Tên Tác Tử | Phân loại | Phương pháp & Đặc trưng Kỹ thuật Chính |
| :---: | :--- | :---: | :--- |
| **Cấp 1** | `Level1_Random` | Bắt buộc | Chọn ngẫu nhiên đồng đều từ `legal_actions()`, làm mốc đối chứng sàn (Baseline). |
| **Cấp 2** | `Level2_Heuristic` | Bắt buộc | Luật ưu tiên 3 tầng (Mate-in-1 $\to$ Phòng thủ & MVV-LVA $\to$ PST Michniewski + King Safety). |
| **Cấp 3** | `Level3_AlphaBeta` | Bắt buộc | Iterative Deepening Negamax $\alpha$-$\beta$ + Move Ordering + Zobrist TT + Quiescence Search. |
| **Cấp 4** | `Level4_MCTS_UCT` | Bắt buộc | Vanilla MCTS (UCB1/UCT) với mô phỏng cắt ngưỡng (`max_rollout_depth: 60`) và định giá tĩnh. |
| **Cấp 5** | `Level5_Supervised` | Bắt buộc | Mạng Dual-Head ResNet (6 blocks, 64 filters) đầu vào Tensor `(18, 8, 8)` góc nhìn bên đi, huấn luyện từ **Lichess Elite Database** có khử trùng Zobrist. |
| **Cấp 6** | `Level6_RL_KRK` | Mở rộng | Học tăng cường Tabular Q-Learning & Policy Gradient (REINFORCE) trên cờ tàn chuẩn FIDE **KRK**. |
| **Cấp 7** | `Level7_GuidedMCTS` | Mở rộng | AlphaGo-inspired Guided MCTS (PUCT) tích hợp Policy Prior $P(s, a)$ và Value Head $V(s)$ của Cấp 5. |
| **Cấp 8** | `Level8_AlphaZero` | Mở rộng | Vòng lặp Self-Play PUCT + Replay Buffer + Checkpoint Arena Gating (ngưỡng thắng $55\%$). |

---

## 2. Cấu Trúc Thư Mục & Tài Liệu Đặc Tả (`configs/`, `chess_ai/`, `resources/`, `results/`, `docs/`)

* **Cấu hình trung tâm (`configs/` — Single Source of Truth):**
  * [`configs/default_config.yaml`](configs/default_config.yaml): Đường dẫn, 10 Random Seeds chuẩn, cấu hình ETL Lichess Elite, Arena 50 khai cuộc & Logging.
  * [`configs/model_resnet.yaml`](configs/model_resnet.yaml): Đặc tả Tensor `(18, 8, 8)` lật bàn cờ theo bên đi, kiến trúc Dual-Head ResNet & Schema Checkpoint.
  * [`configs/search_params.yaml`](configs/search_params.yaml): Bảng điểm PST Michniewski, MVV-LVA, siêu tham số Alpha-Beta, UCT, PUCT, RL (Cấp 6) & Self-Play (Cấp 8).
* **Tài nguyên thế cờ chuẩn (`resources/` — được commit):**
  * [`resources/benchmark_positions.epd`](resources/benchmark_positions.epd): Bộ 50 thế cờ chiến thuật chuẩn Win At Chess (`WAC.001` – `WAC.050`).
  * [`resources/openings_50.epd`](resources/openings_50.epd): Bộ 50 thế cờ khai cuộc cân bằng (6–8 plies) phục vụ giải đấu Arena.
* **Kết quả thực nghiệm & Trọng số nộp bài (`results/`, `checkpoints/`, `report/` — được commit):**
  * `checkpoints/`: Lưu trữ trực tiếp `best_model.pt` ($\sim 5\text{MB}$) và `l6_krk_qtable.pkl` trên Git.
  * `results/`: Lưu trữ log thi đấu thô (`round_robin/matches.jsonl`, `round_robin/games.pgn`), kết quả `ablation1/`, `ablation2/`, `training/` và biểu đồ `figures/`.
  * `report/`: Chứa báo cáo 5 chương, slide thuyết trình, poster A0 và link video demo.
* **Tài liệu kỹ thuật (`docs/`):**
  * [`docs/DECISIONS.md`](docs/DECISIONS.md): Hồ sơ các quyết định kiến trúc & kỹ thuật đã đóng băng (`ADR-01` $\to$ `ADR-08`).
  * [`docs/CONTRIBUTIONS.md`](docs/CONTRIBUTIONS.md): Bảng phân công nhiệm vụ và tỷ lệ đóng góp của 3 thành viên.
  * [`docs/PROMPT_AND_FILE_SPEC.md`](docs/PROMPT_AND_FILE_SPEC.md), [`docs/DATASET_SPEC.md`](docs/DATASET_SPEC.md), [`docs/EXPERIMENT_PROTOCOL.md`](docs/EXPERIMENT_PROTOCOL.md), [`docs/HANDOFF_AND_QA.md`](docs/HANDOFF_AND_QA.md), [`docs/SPRINT_PLAN_15DAYS.md`](docs/SPRINT_PLAN_15DAYS.md), [`docs/REPORT_OUTLINE.md`](docs/REPORT_OUTLINE.md).

---

## 3. Hướng Dẫn Thực Thi Nhanh (Quick Start CLI)

```bash
# 0. Cài đặt gói dự án ở chế độ editable (Python >= 3.11)
pip install -e .

# 1. Chạy toàn bộ 61 kịch bản kiểm thử tự động
pytest

# 2. Trích xuất dữ liệu từ Lichess Elite Database ra mảng nhị phân np.memmap
python -m chess_ai.etl.pgn_etl --config configs/default_config.yaml

# 3. Huấn luyện mạng Dual-Head ResNet (Cấp 5) và RL cờ tàn KRK (Cấp 6)
python -m chess_ai.training.train_supervised --config configs/model_resnet.yaml
python -m chess_ai.training.train_rl_krk --config configs/search_params.yaml

# 4. Vận hành giải đấu đối kháng Round-Robin (100 ván/cặp trên 50 thế cờ khai cuộc cân bằng)
python -m chess_ai.arena.tournament --config configs/default_config.yaml --budget-mode workload_control

# 5. Tái tạo bảng kết quả, khoảng tin cậy Wilson/Bootstrap 95% và biểu đồ báo cáo
python scripts/reproduce_experiment.py --from-logs --config configs/default_config.yaml

# 6. Khởi động giao diện đồ họa Pygame (Người đấu Máy & Máy đấu Máy)
python -m chess_ai.ui.desktop_pygame
```

---

## 4. Quy Cách Đóng Gói Nộp Bài

Theo Mục 3.7 của đề bài, toàn bộ hồ sơ nộp cuối kỳ được đóng gói thành một tệp ZIP duy nhất:
```text
NhomXX_Chess_TTNTNC_HoTen_MaSV.zip
```
