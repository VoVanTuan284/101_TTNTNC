# Hướng dẫn Hệ thống Notebook Phân tích (`notebooks/`)

Tài liệu này mô tả cấu trúc, mục tiêu, dữ liệu đầu vào/đầu ra và quy ước sử dụng bộ Jupyter Notebook trong thư mục `notebooks/`.

> **Nguyên tắc thiết kế (theo `DECISIONS.md` & `EXPERIMENT_PROTOCOL.md`):**
> - **Tách theo chuyên đề khoa học, không tách theo thành viên:** Mỗi notebook tập trung vào một mắt xích rõ ràng của dự án (Dữ liệu/ETL → Cổ điển → RL tàn cuộc → Deep Learning → Ablation Studies → Giải đấu & Phong cách chơi) để người đọc và ban giám khảo theo dõi mạch nghiên cứu liền mạch.
> - **Không huấn luyện nặng trong Notebook:** Mọi tác vụ ETL lớn, huấn luyện mạng nơ-ron (`Level 5`, `Level 8`), giải đấu Round-Robin 2.800 ván và chạy Ablation đều thực hiện qua các CLI script trong `scripts/` và lưu log/checkpoint ra `results/` hoặc `checkpoints/`. Notebook chỉ dùng để **đọc log, phân tích định lượng, kiểm định thống kê, trực quan hóa biểu đồ và chạy suy luận demo nhẹ**.
> - **Xuất hình ảnh chuẩn báo cáo:** Mọi biểu đồ phục vụ báo cáo IEEE 5 chương nên được xuất ra `results/figures/` với độ phân giải cao (300 DPI).

---

## 1. Tổng quan danh sách Notebook

| STT | Tên file | Chuyên đề | Câu hỏi nghiên cứu / Mục tiêu chính |
| :--- | :--- | :--- | :--- |
| 01 | [`01_data_etl_and_tensor.ipynb`](../notebooks/01_data_etl_and_tensor.ipynb) | Dữ liệu Lichess, ETL & Tensor 18 kênh | Kiểm toán chất lượng dữ liệu thô, bộ lọc 5 tiêu chí, khử trùng Zobrist và xác minh `ChessMemmapDataset`. |
| 02 | [`02_classical_agents_and_search.ipynb`](../notebooks/02_classical_agents_and_search.ipynb) | Tác tử Cổ điển (`Level 1`–`Level 4`) | Đánh giá hàm lượng giá tĩnh, cắt tỉa Alpha-Beta, Quiescence Search, Transposition Table và thông lượng NPS. |
| 03 | [`03_rl_endgame_level6.ipynb`](../notebooks/03_rl_endgame_level6.ipynb) | Học tăng cường Tàn cuộc (`Level 6`) | Phân tích đường cong học tập Q-Learning / SARSA / REINFORCE trên `KRK`, `KQK`, `KPK` và demo giải thế cờ. |
| 04 | [`04_deep_learning_agents_l5_l7_l8.ipynb`](../notebooks/04_deep_learning_agents_l5_l7_l8.ipynb) | Tác tử Học sâu (`Level 5`, `Level 7`, `Level 8`) | Theo dõi quá trình hội tụ Policy/Value Net, so sánh UCT vs PUCT và chu trình Self-Play / Arena Gating. |
| 05 | [`05_ablation_studies.ipynb`](../notebooks/05_ablation_studies.ipynb) | Nghiên cứu Cắt bỏ (*Ablation Studies*) | Phân tích định lượng **Ablation 1** (chiến thuật `WAC-50`, `RQ2`) và **Ablation 2** (thành phần MCTS/PUCT, `RQ1`, `RQ3`). |
| 06 | [`06_arena_tournament_and_style_rq4.ipynb`](../notebooks/06_arena_tournament_and_style_rq4.ipynb) | Giải đấu Round-Robin & Phong cách (`RQ4`) | Tổng hợp Elo, ma trận thắng/hòa/thua với khoảng tin cậy Wilson/Bootstrap 95%, độ trễ nước đi và đặc trưng phong cách chơi. |
| — | [`analysis.ipynb`](../notebooks/analysis.ipynb) | Notebook Tổng hợp (*Executive Summary*) | Tóm tắt nhanh các bảng kết quả chính từ toàn bộ pipeline để đối chiếu nhanh khi thuyết trình. |

---

## 2. Chi tiết từng Notebook

### 2.1. `01_data_etl_and_tensor.ipynb` — Dữ liệu, ETL & Kiểm tra Tensor
- **Mục tiêu:** Chứng minh tính sạch, không rò rỉ dữ liệu (*no data leakage*) và tính đúng đắn của biểu diễn trạng thái 18 kênh `(18, 8, 8)`.
- **Đầu vào chính:**
  - `data/raw/*.pgn` (hoặc mẫu kiểm tra)
  - `data/processed/dataset_metadata.json`, `data/processed/*.npy` (qua `ChessMemmapDataset`)
- **Đầu ra kỳ vọng:** Thống kê phân bố Elo/kết quả ván đấu, tỷ lệ lọc 5 tiêu chí, xác minh không trùng lặp Zobrist giữa tập train/val/test, trực quan hóa 18 mặt phẳng đặc trưng bàn cờ (`results/figures/`).

### 2.2. `02_classical_agents_and_search.ipynb` — Tác tử Cổ điển (Level 1–4)
- **Mục tiêu:** Phân tích sự tiến hóa hiệu năng từ `Level 1` (Random) → `Level 2` (Greedy) → `Level 3` (Minimax) → `Level 4` (Alpha-Beta + Iterative Deepening + Quiescence Search + TT).
- **Đầu vào chính:**
  - `src/agents/classical/`
  - Log thống kê tìm kiếm / benchmark tốc độ (NPS, số nút duyệt, tỷ lệ cắt tỉa nhánh, độ sâu đạt được trong giới hạn thời gian).
- **Đầu ra kỳ vọng:** Bảng so sánh NPS, hệ số nhánh hiệu dụng (*effective branching factor*) và phân tích tác động của sắp xếp nước đi (*MVV-LVA*, *Killer/History heuristic*).

### 2.3. `03_rl_endgame_level6.ipynb` — Học tăng cường Tàn cuộc (Level 6)
- **Mục tiêu:** Đánh giá bài toán tàn cuộc `KRK` (và mở rộng `KQK`, `KPK`) cho `Level 6` với các thuật toán Tabular Q-Learning, SARSA và Policy Gradient (`REINFORCE`).
- **Đầu vào chính:**
  - `results/training/krk_learning_curve.json` (và các file log tương ứng)
  - `checkpoints/l6_krk_qtable.pkl`
  - `src/agents/rl/level6_rl.py`, `src/env/endgame_env.py`
- **Đầu ra kỳ vọng:** Biểu đồ đường cong học tập (tỷ lệ chiếu hết, số bước trung bình về đích, reward theo episode), bảng đối chiếu tốc độ hội tụ giữa các thuật toán và demo ván cờ tàn cuộc từ `Level6ReinforcementAgent`.

### 2.4. `04_deep_learning_agents_l5_l7_l8.ipynb` — Tác tử Học sâu (Level 5, 7, 8)
- **Mục tiêu:** Phân tích hành vi huấn luyện và suy luận của nhóm tác tử sử dụng mạng nơ-ron (`PolicyValueNet`).
- **Đầu vào chính:**
  - `results/training/` (log loss `policy_loss`, `value_loss`, validation accuracy)
  - `checkpoints/best_model.pt`
  - `src/agents/dl/` (`Level5`, `Level7`, `Level8`)
- **Đầu ra kỳ vọng:** Đường cong loss & độ chính xác theo epoch, phân tích phổ entropy dự đoán chính sách, đánh giá tác động của số lượt mô phỏng MCTS (`Level 7` vs `Level 8`) và quy trình Self-Play / Arena Gating.

### 2.5. `05_ablation_studies.ipynb` — Nghiên cứu Cắt bỏ (Ablation Studies)
- **Mục tiêu:** Trả lời trực diện `RQ1`, `RQ2`, `RQ3` bằng cách cô lập đóng góp của từng thành phần thuật toán:
  - **Ablation 1 (Tác động chiến thuật – `WAC-50`, `RQ2`):** So sánh `Level 4` khi bật/tắt *Quiescence Search*, *Transposition Table* và *Move Ordering*.
  - **Ablation 2 (Thành phần tìm kiếm MCTS/PUCT – `RQ1`, `RQ3`):** So sánh `UCT` thuần (`Level 7`) và `PUCT` có hướng dẫn mạng nơ-ron (`Level 8`), cùng các biến thể tắt *Prior Policy* hoặc *Value Leaf Evaluation* và thay đổi ngân sách mô phỏng.
- **Đầu vào chính:**
  - `results/ablation1/`
  - `results/ablation2/`
- **Đầu ra kỳ vọng:** Bảng so sánh tỷ lệ giải đúng `WAC-50`, Elo ước lượng, khoảng tin cậy 95% và biểu đồ đánh đổi Độ chính xác vs Thời gian/Mô phỏng.

### 2.6. `06_arena_tournament_and_style_rq4.ipynb` — Giải đấu Round-Robin & Phong cách chơi (RQ4)
- **Mục tiêu:** Tổng hợp kết quả giải đấu chuẩn hóa giữa cả 8 cấp độ tác tử (100 ván/cặp trên 50 khai cuộc đối xứng) và phân tích phong cách chơi (`RQ4`).
- **Đầu vào chính:**
  - `results/round_robin/matches.jsonl`
  - `results/round_robin/games.pgn`
- **Đầu ra kỳ vọng:** Bảng xếp hạng Elo kèm khoảng tin cậy Bootstrap 95%, ma trận thắng/hòa/thua kèm Wilson 95% CI, biểu đồ độ trễ nước đi / NPS / SPS, các chỉ số phong cách chơi (độ dài ván, tỷ lệ thí quân/tấn công, tốc độ kết thúc tàn cuộc) và trích xuất một số ván PGN tiêu biểu.

---

## 3. Quy ước chạy và bảo trì Notebook

1. **Môi trường:** Luôn chạy bằng kernel của `.venv` ở thư mục gốc dự án.
2. **Định vị thư mục gốc (`PROJECT_ROOT`):** Cell khởi tạo đầu mỗi notebook tự động dò `PROJECT_ROOT` và chèn vào `sys.path` để import được `src.*`.
3. **Kiểm tra trước khi commit:**
   - Đảm bảo JSON hợp lệ (`nbformat 4.5`).
   - Không nhúng file dữ liệu lớn hoặc output nhị phân nặng vào notebook nếu không cần thiết.
   - Chạy kiểm thử mã nguồn bằng `.venv\Scripts\pytest.exe -v` trước khi tạo Pull Request theo quy trình tại [`docs/GITFLOW.md`](GITFLOW.md).
