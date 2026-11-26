# Hướng dẫn Hệ thống Notebook Phân tích (`notebooks/`)

Tài liệu này mô tả cấu trúc, mục tiêu, dữ liệu đầu vào/đầu ra và quy ước sử dụng bộ Jupyter Notebook trong thư mục `notebooks/`.

> **Nguyên tắc thiết kế (theo `DECISIONS.md` & `EXPERIMENT_PROTOCOL.md`):**
> - **Tách theo chuyên đề khoa học, không tách theo thành viên:** Mỗi notebook tập trung vào một mắt xích rõ ràng của dự án (Dữ liệu/ETL → Cổ điển → RL tàn cuộc → Deep Learning → Huấn luyện AlphaZero → Ablation Studies → Giải đấu & Phong cách chơi) để người đọc theo dõi mạch nghiên cứu liền mạch.
> - **Tách biệt tác vụ nặng và phân tích:** Các tác vụ ETL lớn, huấn luyện `Level 5` (`chess-train-sl` / `train_level5_colab.ipynb`), huấn luyện `Level 6` (`chess-train-rl`), giải đấu Round-Robin (`chess-arena`) và Ablation đều có CLI script trong `chess_ai/` và `scripts/`, lưu log/checkpoint ra `results/` hoặc `checkpoints/`. Riêng `train_level8_colab.ipynb` đóng gói quy trình huấn luyện tự chơi `Level 8` trên Colab T4 GPU kèm cơ chế `AUTO_RESUME`.
> - **Xuất hình ảnh chuẩn báo cáo:** Mọi biểu đồ phục vụ báo cáo IEEE 5 chương được xuất ra `results/figures/` với độ phân giải cao.

---

## 1. Tổng quan danh sách Notebook

| STT | Tên file | Chuyên đề | Câu hỏi nghiên cứu / Mục tiêu chính |
| :--- | :--- | :--- | :--- |
| 01 | [`01_data_etl_and_tensor.ipynb`](../notebooks/01_data_etl_and_tensor.ipynb) | Dữ liệu Lichess, ETL & Tensor 18 kênh | Kiểm toán chất lượng dữ liệu thô, bộ lọc 5 tiêu chí, khử trùng Zobrist 64-bit và xác minh `ChessMemmapDataset`. |
| 02 | [`02_classical_agents_and_search.ipynb`](../notebooks/02_classical_agents_and_search.ipynb) | Tác tử Cổ điển (`Level 1`–`Level 4`) | Đánh giá hàm lượng giá tĩnh, cắt tỉa Alpha-Beta (`Level 3`), Quiescence Search, Transposition Table, Vanilla MCTS UCT (`Level 4`) và thông lượng NPS/SPS. |
| 03 | [`03_rl_endgame_level6.ipynb`](../notebooks/03_rl_endgame_level6.ipynb) | Học tăng cường Tàn cuộc (`Level 6`) | Phân tích đường cong học tập Afterstate TD($\lambda$) / Tabular Q-Learning / Batch REINFORCE trên `KQK`, `KRK`, `KPK` và demo giải thế cờ. |
| 04 | [`04_deep_learning_agents_l5_l7_l8.ipynb`](../notebooks/04_deep_learning_agents_l5_l7_l8.ipynb) | Tác tử Học sâu (`Level 5`, `Level 7`, `Level 8`) | Theo dõi quá trình hội tụ `DualHeadResNet`, so sánh UCT vs PUCT (kèm FPU, Zobrist Cache, Batched Virtual Loss) và chu trình Self-Play / Arena Gating. |
| 05 | [`05_ablation_studies.ipynb`](../notebooks/05_ablation_studies.ipynb) | Nghiên cứu Cắt bỏ (*Ablation Studies*) | Phân tích định lượng **Ablation 1** (chiến thuật `WAC-50`, `RQ2`) và **Ablation 2** (độ nhạy $N_{\text{sim}}$ của UCT `Level 4` & PUCT `Level 7`, bóc tách thành phần, `RQ1`, `RQ3`). |
| 06 | [`06_arena_tournament_and_style_rq4.ipynb`](../notebooks/06_arena_tournament_and_style_rq4.ipynb) | Giải đấu Round-Robin & Phong cách (`RQ4`) | Tổng hợp Elo, ma trận thắng/hòa/thua với khoảng tin cậy Wilson/Bootstrap 95%, độ trễ nước đi và đặc trưng phong cách chơi. |
| — | [`level2_heuristic_principles.ipynb`](../notebooks/level2_heuristic_principles.ipynb) | Nguyên tắc chơi `Level 2` | Minh họa trực quan phân tầng ưu tiên **6 Tiers (`Tier 1–6`)**, `SEE`, kiểm soát trung tâm và thống kê xung đột luật của `Level2HeuristicAgent`. |
| — | [`ve_anh_custom.ipynb`](../notebooks/ve_anh_custom.ipynb) | Vẽ ảnh bàn cờ tùy chỉnh | Tái sử dụng `render_board_image()` để vẽ thế cờ bất kỳ, toàn bộ nước đi hợp lệ của `Level 1` và so sánh các tác tử. |
| — | [`train_level8_colab.ipynb`](../notebooks/train_level8_colab.ipynb) | Huấn luyện `Level 8` AlphaZero (Colab) | Thực thi vòng lặp Self-Play $\rightarrow$ Replay Buffer Refinement $\rightarrow$ Arena Gating ($\ge 55\%$) khởi tạo ấm từ `Level 5` trên Colab T4 GPU. |
| — | [`test_lv7_more_sims.ipynb`](../notebooks/test_lv7_more_sims.ipynb) | Thử nghiệm mở rộng `Level 7` | Đối đầu `Level 7` (`400/800/1000/1500 sims`) vs `Level 3 (4000 nodes)` kèm biểu đồ đánh đổi điểm số & độ trễ. |
| — | [`analysis.ipynb`](../notebooks/analysis.ipynb) | Notebook Tổng hợp (*Executive Summary*) | Tóm tắt nhanh các bảng kết quả chính từ toàn bộ pipeline để đối chiếu nhanh khi thuyết trình. |

---

## 2. Chi tiết từng Notebook

### 2.1. `01_data_etl_and_tensor.ipynb` — Dữ liệu, ETL & Kiểm tra Tensor
- **Mục tiêu:** Chứng minh tính sạch, không rò rỉ dữ liệu (*no data leakage*) và tính đúng đắn của biểu diễn trạng thái 18 kênh `(18, 8, 8)`.
- **Đầu vào chính:**
  - `data/raw/lichess_elite_2025-11.pgn` (hoặc mẫu kiểm tra)
  - `data/processed/dataset_metadata.json`, `data/processed/*.dat` (`np.memmap` qua `ChessMemmapDataset` trong `chess_ai/etl/memmap_dataset.py`)
- **Đầu ra kỳ vọng:** Thống kê phân bố Elo/kết quả ván đấu, tỷ lệ lọc 5 tiêu chí, xác minh không trùng lặp Zobrist giữa tập train/val/test, trực quan hóa 18 mặt phẳng đặc trưng bàn cờ (`results/figures/`).

### 2.2. `02_classical_agents_and_search.ipynb` — Tác tử Cổ điển (`Level 1`–`Level 4`)
- **Mục tiêu:** Phân tích sự tiến hóa hiệu năng từ `Level 1` (Random) → `Level 2` (Heuristic phân tầng **6 Tiers** + SEE + Anti-Stalemate + Center Control + ghi nhận xung đột luật) → `Level 3` (Iterative Deepening Alpha-Beta + PVS + Aspiration Windows + Move Ordering + Zobrist TT + Quiescence/SEE) → `Level 4` (Vanilla MCTS UCT).
- **Đầu vào chính:**
  - `chess_ai/agents/level1_random.py` .. `level4_mcts.py`, `chess_ai/search/`
  - Log thống kê tìm kiếm / benchmark tốc độ (NPS, số nút duyệt, tỷ lệ cắt tỉa nhánh, độ sâu đạt được trong giới hạn thời gian).
- **Đầu ra kỳ vọng:** Bảng so sánh NPS/SPS, hệ số nhánh hiệu dụng (*effective branching factor*) và phân tích tác động của sắp xếp nước đi (*MVV-LVA*, *Killer/History heuristic*).

### 2.3. `03_rl_endgame_level6.ipynb` — Học tăng cường Tàn cuộc (`Level 6`)
- **Mục tiêu:** Đánh giá bài toán tàn cuộc `KQK`, `KRK`, `KPK` cho `Level 6` với **Afterstate Value Learning TD($\lambda$)** (không gian đối xứng D4 + chuẩn hóa màu), Tabular Q-Learning và **Batch REINFORCE**.
- **Đầu vào chính:**
  - `results/training/krk_learning_curve.json`
  - `checkpoints/l6_krk_qtable.pkl`
  - `chess_ai/agents/level6_reinforcement.py`, `chess_ai/training/train_rl_krk.py`
- **Đầu ra kỳ vọng:** Biểu đồ đường cong học tập (`results/figures/rl_krk_learning_curve.png`: tỷ lệ chiếu hết, số bước trung bình về đích theo episode), bảng đối chiếu hội tụ giữa `KQK`, `KRK`, `KPK`, `REINFORCE` và demo ván cờ tàn cuộc từ `Level6ReinforcementAgent`.

### 2.4. `04_deep_learning_agents_l5_l7_l8.ipynb` — Tác tử Học sâu (`Level 5`, `Level 7`, `Level 8`)
- **Mục tiêu:** Phân tích hành vi huấn luyện và suy luận của nhóm tác tử sử dụng mạng nơ-ron `DualHeadResNet` (`chess_ai/models/dual_resnet.py`).
- **Đầu vào chính:**
  - `results/training/sl_training_metrics.json`, `results/training/selfplay_metrics.json`
  - `checkpoints/best_model.pt`, `checkpoints/l8_alphazero_best.pt`
  - `chess_ai/agents/level5_supervised.py`, `level7_guided_mcts.py`, `level8_alphazero.py`
- **Đầu ra kỳ vọng:** Đường cong loss & độ chính xác theo epoch (`results/figures/fig_level5_training_curves.png`), phân tích phổ entropy dự đoán chính sách, đánh giá `Level 7` PUCT (FPU Reduction, Zobrist Eval Cache, Batched Virtual Loss) và quy trình Self-Play / Arena Gating của `Level 8`.

### 2.5. `05_ablation_studies.ipynb` — Nghiên cứu Cắt bỏ (*Ablation Studies*)
- **Mục tiêu:** Trả lời trực diện `RQ1`, `RQ2`, `RQ3` bằng cách cô lập đóng góp của từng thành phần thuật toán:
  - **Ablation 1 (Tác động chiến thuật – `WAC-50`, `RQ2`):** So sánh 6 cấu hình lũy tiến của `Level 3` (`Config_A` Minimax $\to$ `Config_B` Alpha-Beta $\to$ `Config_C1` +Move Ordering $\to$ `Config_C2` +Zobrist TT $\to$ `Config_D` +Quiescence Search $\to$ `Config_E` +PVS & Aspiration) và độ nhạy độ sâu $d \in \{2, 3, 4\}$.
  - **Ablation 2 (Thành phần tìm kiếm MCTS/PUCT – `RQ1`, `RQ3`):** Khảo sát độ nhạy $N_{\text{sim}} \in \{50, 100, 200, 400, 800, 1000\}$ cho cả `Level 4` (UCT) và `Level 7` (PUCT), cùng so sánh bóc tách `Level 5` (Policy-only) vs Value-only MCTS vs `Level 7` (Full PUCT).
- **Đầu vào chính:**
  - `results/ablation1/ablation1_results.json`
  - `results/ablation2/ablation2_results.json`
- **Đầu ra kỳ vọng:** Bảng so sánh tỷ lệ giải đúng `WAC-50`, tỷ lệ điểm số $S$, khoảng tin cậy 95% và biểu đồ đánh đổi Độ chính xác vs Thời gian/Mô phỏng (`results/figures/ablation1_*.png`, `ablation2_mcts_sensitivity.png`).

### 2.6. `06_arena_tournament_and_style_rq4.ipynb` — Giải đấu Round-Robin & Phong cách chơi (`RQ4`)
- **Mục tiêu:** Tổng hợp kết quả giải đấu chuẩn hóa giữa các cấp độ tác tử (200 ván/cặp trên 50 khai cuộc đối xứng × 2 seeds) và phân tích phong cách chơi (`RQ4`).
- **Đầu vào chính:**
  - `results/round_robin_200/summary_7x7_200.json`, `results/round_robin_200/leaderboard_7x7_200.csv`
  - `results/dual_mode_200/dual_mode_200_summary.json`
- **Đầu ra kỳ vọng:** Bảng xếp hạng Elo kèm khoảng tin cậy Bootstrap 95%, ma trận thắng/hòa/thua kèm Wilson 95% CI, biểu đồ độ trễ nước đi / NPS / SPS, các chỉ số phong cách chơi (Move Agreement Rate, Castling Rate, độ dài ván, tỷ lệ chiếu hết) và trích xuất một số ván PGN tiêu biểu.

### 2.7. `train_level8_colab.ipynb` — Huấn luyện `Level 8` AlphaZero Self-Play (Colab)
- **Mục tiêu:** Thực thi pipeline huấn luyện `Level 8` trên Google Colab T4 GPU (`--iterations 6`, `--games-per-iter 20`, `--sims 100`, `--arena-games 20`) với cơ chế tự động nối tiếp từ Google Drive (`AUTO_RESUME`).
- **Đầu vào chính:** `checkpoints/best_model.pt` (Warm-start từ `Level 5`), `resources/openings_50.epd`.
- **Đầu ra chính:** `checkpoints/l8_alphazero_best.pt`, `checkpoints/l8_alphazero_latest.pt`, `results/training/selfplay_metrics.json`, `results/figures/fig_level8_selfplay_gating.png`.

---

## 3. Quy ước chạy và bảo trì Notebook

1. **Môi trường:** Luôn chạy bằng kernel của `.venv` ở thư mục gốc dự án.
2. **Định vị thư mục gốc (`PROJECT_ROOT`):** Cell khởi tạo đầu mỗi notebook tự động dò `PROJECT_ROOT` và chèn vào `sys.path` để import được `chess_ai.*`.
3. **Kiểm tra trước khi commit:**
   - Đảm bảo JSON hợp lệ (`nbformat 4.5`).
   - Không nhúng file dữ liệu lớn hoặc output nhị phân nặng vào notebook nếu không cần thiết.
   - Chạy kiểm thử mã nguồn bằng `.\.venv\Scripts\pytest.exe -v` trước khi tạo Pull Request theo quy trình tại [`docs/GITFLOW.md`](GITFLOW.md).
