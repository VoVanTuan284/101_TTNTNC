# BẢNG PHÂN CÔNG & GHI NHẬN TỶ LỆ ĐÓNG GÓP THÀNH VIÊN (CONTRIBUTIONS.md)

**Dự án:** Multi-Level Chess AI Engine ($8 \times 8$ FIDE)  
**Môn học:** Trí tuệ Nhân tạo Nâng cao  
**Quy định nộp bài (Mục 3.7 đề bài):** Cung cấp liên kết GitHub, lịch sử commit, phân công nhiệm vụ và tỷ lệ đóng góp của từng thành viên.

---

## 1. Bảng Tổng Hợp Đóng Góp Cá Nhân

| Mã TV | Họ và Tên | Mã Sinh Viên | Vai Trò Chính (Lead Role) | Nhánh Git Phụ Trách | Tỷ Lệ Đóng Góp |
| :---: | :--- | :---: | :--- | :--- | :---: |
| **TV1** | *(Điền Họ Tên TV1)* | *(Điền MSSV)* | Core, Classical Search, UCT & RL Lead | `feature/tv1-core-search-rl` | **$33.3\%$** |
| **TV2** | *(Điền Họ Tên TV2)* | *(Điền MSSV)* | Deep Learning, ETL, PUCT & Self-Play Lead | `feature/tv2-data-models-puct` | **$33.3\%$** |
| **TV3** | *(Điền Họ Tên TV3)* | *(Điền MSSV)* | UI, Arena, Testing, Configs & Analytics Lead | `feature/tv3-ui-arena-qa` | **$33.4\%$** |
| **Tổng** | — | — | — | `main` / `dev` | **$100.0\%$** |

---

## 2. Chi Tiết Các Phân Hệ Mã Nguồn & Báo Cáo Phụ Trách

### 2.1. Thành viên 1 (`TV1` - $33.3\%$)
* **Mã nguồn (`chess_ai/`):**
  * `chess_ai/core/game_env.py`, `chess_ai/core/chess_env.py`, `chess_ai/core/evaluation.py`, `chess_ai/core/zobrist.py`
  * `chess_ai/search/alphabeta.py`, `chess_ai/search/move_ordering.py`, `chess_ai/search/quiescence.py`, `chess_ai/search/transposition.py`
  * `chess_ai/search/mcts_node.py`, `chess_ai/search/mcts_uct.py`
  * `chess_ai/agents/base_agent.py`, `level1_random.py`, `level2_heuristic.py`, `level3_alphabeta.py`, `level4_mcts.py`, `level6_reinforcement.py`
  * `chess_ai/training/train_rl_krk.py`, `chess_ai/experiments/ablation_search.py`
* **Báo cáo:** Chương 2 (Cơ sở lý thuyết), phần Tìm kiếm cổ điển & RL trong Chương 3 và Chương 4.

### 2.2. Thành viên 2 (`TV2` - $33.3\%$)
* **Mã nguồn (`chess_ai/`):**
  * `chess_ai/core/board_tensor.py` (Tensor 18 kênh góc nhìn bên đi & Action 4096)
  * `chess_ai/etl/pgn_etl.py`, `chess_ai/etl/memmap_dataset.py`
  * `chess_ai/models/dual_resnet.py`, `chess_ai/models/loss.py`
  * `chess_ai/search/mcts_puct.py`
  * `chess_ai/agents/level5_supervised.py`, `level7_guided_mcts.py`, `level8_alphazero.py`
  * `chess_ai/training/train_supervised.py`, `chess_ai/training/selfplay.py`, `chess_ai/training/replay_buffer.py`
* **Báo cáo & Ấn phẩm:** Phần Học sâu & PUCT trong Chương 3, Chương 4, Chương 5 và thiết kế Poster A0.

### 2.3. Thành viên 3 (`TV3` - $33.4\%$)
* **Mã nguồn (`chess_ai/`, `tests/`, `scripts/`, `notebooks/`):**
  * `chess_ai/config.py`, `chess_ai/utils/` (`seeding.py`, `budget.py`, `epd.py`), `chess_ai/agents/registry.py`
  * `chess_ai/arena/` (`match.py`, `tournament.py`, `metrics.py`, `pgn_exporter.py`)
  * `chess_ai/experiments/ablation_mcts.py`, `chess_ai/ui/desktop_pygame.py`
  * Bộ 61 bài kiểm thử tự động trong `tests/` (`TC01` – `TC61`), `scripts/reproduce_experiment.py`, `notebooks/analysis.ipynb`
* **Báo cáo & Ấn phẩm:** Chương 1, tổng hợp Chương 4 & 5, biên tập `report/`, thiết kế Slide thuyết trình và quay Video Demo.

