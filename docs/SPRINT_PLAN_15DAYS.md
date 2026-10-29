# KẾ HOẠCH SPRINT 15 NGÀY & PHÂN CÔNG NHIỆM VỤ (SPRINT_PLAN_15DAYS.md)

**Mã tài liệu:** `SPRINT_PLAN_2026`  
**Dự án:** Multi-Level Chess AI Engine ($8 \times 8$ FIDE) *(Đã được giảng viên phê duyệt)*  
**Môn học:** Trí tuệ Nhân tạo Nâng cao  
**Thời gian thực thi:** 15 Ngày liên tục (Chia thành 5 Pha chiến lược)  
**Quy mô nhóm:** 3 thành viên (`TV1`, `TV2`, `TV3`) | Cơ cấu đóng góp: $33.3\% - 33.3\% - 33.4\%$  
**Nguồn sự thật cấu hình:** `configs/` & `docs/DECISIONS.md`  

---

## PHẦN 1: MA TRẬN PHÂN CÔNG TRÁCH NHIỆM ĐÃ TÁI CÂN BẰNG (RACI MATRIX - ADR-08)

Để triệt tiêu tình trạng quá tải đường găng ở một thành viên và loại bỏ hoàn toàn xung đột gộp nhánh (Merge Conflict), hệ thống phân định ranh giới module theo mô hình RACI đã tái cân bằng:
* **R (Responsible):** Thành viên chịu trách nhiệm chính viết mã nguồn.
* **A (Accountable):** Thành viên chịu trách nhiệm nghiệm thu chất lượng kỹ thuật.
* **C (Consulted):** Thành viên phối hợp viết test, phản biện thiết kế hoặc đồng phát triển.
* **I (Informed):** Thành viên được thông báo trạng thái cập nhật để đồng bộ giao diện.

| Phân hệ / Module Mã Nguồn | TV1 (Core, Search, UCT & RL Lead) | TV2 (DL, ETL, PUCT & Self-Play Lead) | TV3 (UI, Arena, QA & Analytics Lead) | Tỷ lệ Đóng góp Module |
| :--- | :---: | :---: | :---: | :--- |
| **Môi trường & Luật cờ FIDE (`chess_ai/core/game_env`, `chess_env`)** | **R / A** | I | C (Test `TC01–25`) | TV1: $80\%$, TV3: $20\%$ |
| **Biểu diễn Tensor 18 Kênh (`chess_ai/core/board_tensor.py`)** | I | **R / A** | C (Test `TC26–35`) | TV2: $75\%$, TV3: $25\%$ |
| **Tìm kiếm Cổ điển (`chess_ai/search/alphabeta`, `quiescence`, Cấp 1–3)** | **R / A** | I | C (Test & Benchmark) | TV1: $85\%$, TV3: $15\%$ |
| **Đường ống Lichess Elite, Dedup & Memmap (`chess_ai/etl/`)** | I | **R / A** | I | TV2: $100\%$ |
| **Mạng Dual-Head ResNet & Training (`chess_ai/models/`, Cấp 5)** | I | **R / A** | C (Config & Log) | TV2: $85\%$, TV3: $15\%$ |
| **Vanilla MCTS UCT (`chess_ai/search/mcts_node`, `mcts_uct`, Cấp 4)** | **R / A** | C (Hợp đồng Node) | I | TV1: $85\%$, TV2: $15\%$ |
| **Học Tăng Cường Cờ Tàn KRK (`level6_reinforcement.py`)** | **R / A** | C (Hàm thưởng) | I | TV1: $85\%$, TV2: $15\%$ |
| **Guided MCTS PUCT (`chess_ai/search/mcts_puct`, Cấp 7)** | C (`mcts_node`) | **R / A** | I | TV2: $85\%$, TV1: $15\%$ |
| **Vòng lặp Self-Play AlphaZero (`level8_alphazero.py`)** | C (Arena Gate) | **R / A** | I | TV2: $85\%$, TV1: $15\%$ |
| **Giao diện Pygame Desktop Đa tiến trình (`chess_ai/ui/`)** | C (API Agent) | I | **R / A** | TV3: $90\%$, TV1: $10\%$ |
| **Sân đấu Arena 50 Khai cuộc, Wilson & Bootstrap (`chess_ai/arena/`)** | C | C | **R / A** (Kèm `TC52–61`) | TV3: $80\%$, TV1: $10\%$, TV2: $10\%$ |
| **Thực nghiệm 02 bài Ablation & Notebook (`chess_ai/experiments/`, `notebooks/`)** | R (Ablation 1) | C (Hỗ trợ L5/L7) | **A / R** (Ablation 2, Notebook) | TV1: $35\%$, TV2: $25\%$, TV3: $40\%$ |
| **Báo cáo Khoa học 5 Chương, Slide, Poster & Demo Video** | R (Ch. 2, 3, 4) | R (Ch. 3, 4, 5) | **A / R** (Ch. 1, 4, 5, Tổng hợp) | TV1: $33.3\%$, TV2: $33.3\%$, TV3: $33.4\%$ |
| **TỔNG KẾT TỶ LỆ ĐÓNG GÓP TOÀN ĐỀ TÀI** | **$33.3\%$** | **$33.3\%$** | **$33.4\%$** | **$100.0\%$** |

---

## PHẦN 2: LỘ TRÌNH TRIỂN KHAI 15 NGÀY CHI TIẾT (DAY-BY-DAY TIMELINE)

```text
  [PHA 1: Ngày 1-4] ───> [PHA 2: Ngày 5-7] ───> [PHA 3: Ngày 8-9] ───> [PHA 4: Ngày 10-12] ───> [PHA 5: Ngày 13-15]
   Nền tảng Core,         ETL Lichess Elite,     MCTS Tách biệt         Giải đấu 100 ván/cặp     Báo cáo 5 Chương,
  Baseline Cấp 1-3 &     Tối ưu Cấp 3 (TT/QS)   (L4 UCT & L7 PUCT)     trên 50 Khai cuộc &       Notebook, Slide &
   Pilot Đo Hiệu Năng     & Huấn luyện ResNet    + Mở rộng L6 & L8      02 Ablation Studies       Đóng gói Nghiệm thu
```

---

### PHA 1: NỀN TẢNG MÔI TRƯỜNG, BIỂU DIỄN 18 KÊNH & CỘT MỐC CỔ ĐIỂN (NGÀY 1 – 4)
*Mục tiêu pha:* Xác lập hợp đồng giao diện Clean Architecture, hoàn thiện `ChessEnv` chuẩn FIDE, mã hóa Tensor 18 kênh góc nhìn bên đi, giao diện Pygame khung xương, tác tử Cấp 1–3 và chạy đo kiểm Pilot.

#### Ngày 1: Khởi tạo Hạ tầng, Hợp đồng Môi trường & Biểu diễn Tensor 18 Kênh
* **TV1 (Core & Search Lead):**
  * Cài đặt `chess_ai/core/game_env.py`: Hiện thực hóa Abstract Base Class với đầy đủ các phương thức chuẩn đề bài (`reset`, `current_player`, `legal_actions`, `step`, `is_terminal`, `result`, `render`, `clone`, `undo`).
  * Xây dựng `chess_ai/core/chess_env.py` bọc `python-chess`: Bảo đảm `clone()` giữ nguyên lịch sử nước đi (`stack=True`) để phát hiện lặp thế cờ 3 lần chính xác; bổ sung `is_terminal_fast()` cho nút trong của cây tìm kiếm.
  * Xây dựng `chess_ai/agents/base_agent.py`.
* **TV2 (DL & ETL Lead):**
  * Thiết lập module `chess_ai/core/board_tensor.py` theo đúng **ADR-02 & ADR-03** (`configs/model_resnet.yaml`):
    * Hiện thực hàm `board_to_tensor()` kích thước chuẩn PyTorch CHW `(18, 8, 8)` với phép lật bàn cờ chuẩn hóa góc nhìn bên đi (`sq ^ 56` khi Đen đi): Kênh 0–5 (Active), 6–11 (Opponent), 12–15 (4 mặt phẳng nhị phân Castling riêng biệt), 16 (En-Passant target), 17 (Halfmove clock chuẩn hóa).
    * Hiện thực hàm mã hóa/giải mã không gian hành động 4096 chiều ($u_{\text{canon}} \times 64 + v_{\text{canon}}$), quy tắc mặc định phong Hậu và Legal Action Masking.
* **TV3 (UI, Arena & QA Lead):**
  * Chuẩn hóa cấu trúc repository Git, `.gitignore`, `requirements.txt`, và bộ nạp cấu hình tập trung từ `configs/` (`default_config.yaml`, `model_resnet.yaml`, `search_params.yaml`).
  * Khởi tạo bộ 50 thế cờ khai cuộc cân bằng `resources/openings_50.epd` (sau $6 \div 8$ plies) và rà soát bộ 50 thế cờ chiến thuật `resources/benchmark_positions.epd` (`WAC.001` – `WAC.050`).
  * Dựng khung cửa sổ đồ họa `chess_ai/ui/desktop_pygame.py`: Vẽ bàn cờ 64 ô và nạp tài nguyên quân cờ.

#### Ngày 2: Lượng giá Tĩnh Michniewski, Lọc Header Lichess Elite & Kiểm thử FIDE
* **TV1:**
  * Cài đặt `chess_ai/core/evaluation.py`: Hiện thực hàm lượng giá vị trí từ góc nhìn bên đi kết hợp hệ số lực lượng centipawn (`P=100, N=320, B=330, R=500, Q=900, K=20000`) và bảng PST Michniewski (Khai/Trung cuộc & Tàn cuộc).
  * Viết tác tử `chess_ai/agents/level1_random.py` (Uniform Random Baseline).
* **TV2:**
  * Xây dựng `chess_ai/etl/pgn_etl.py` đọc tệp **Lichess Elite Database** từ `data/raw/`:
    * Áp dụng lọc nhanh ở mức Header (`chess.pgn.read_headers()`) trước khi parse cây nước đi: $\text{ELO} \ge 2100$, thời gian ước lượng $T_{\text{base}} + 40 \times T_{\text{inc}} \ge 180\text{s}$ (loại trừ Bullet), `Termination` hợp lệ, độ dài $\ge 30$ plies, loại bỏ hòa sớm trước 60 plies.
* **TV3:**
  * Soạn thảo và chạy tự động bộ kiểm thử FIDE `tests/test_env_fide.py` (25 test cases: `TC01` – `TC25`).
  * Bổ sung xử lý sự kiện chuột chọn ô cờ trong `chess_ai/ui/desktop_pygame.py`.

#### Ngày 3: Tác tử Heuristic 3 Tầng, Khử Trùng Lặp Zobrist & Kiểm thử Tensor
* **TV1:**
  * Xây dựng `chess_ai/agents/level2_heuristic.py`: Hiện thực cấu trúc ưu tiên 3 tầng (Think 1: Chiếu hết ngay `mate-in-1`; Think 2: Phòng thủ đe dọa trực tiếp & Bắt quân có lợi theo MVV-LVA; Think 3: Tối ưu hàm lượng giá tĩnh kèm phạt nhẹ lặp thế cờ khi đang có ưu thế lực lượng để tránh hòa lặp vô nghĩa).
* **TV2:**
  * Hoàn thiện cơ chế Subsampling ($k \in \{3, 4\}$ plies, bỏ qua 12 plies khai cuộc), loại bỏ mẫu phong cấp dưới (`drop_underpromotions: true`), **khử trùng lặp thế cờ bằng khóa băm Zobrist 64-bit** (`deduplicate_zobrist: true`), và phân chia cấp độ ván đấu tất định bằng băm MD5 của `Game ID` (80% Train, 10% Val, 10% Test).
* **TV3:**
  * Viết và chạy bộ kiểm thử `tests/test_representation.py` (`TC26` – `TC35`), bao gồm kiểm tra tính bất biến đối xứng gương màu quân (`TC34`).
  * Hoàn thiện kéo thả quân cờ (Drag & Drop) và hiển thị gợi ý nước đi hợp lệ trên `chess_ai/ui/desktop_pygame.py`.

#### Ngày 4: Iterative Deepening Alpha-Beta, Pilot Đo Hiệu Năng & Milestone 1
* **TV1:**
  * Xây dựng thuật toán Negamax Alpha-Beta tích hợp **Iterative Deepening (ID)** và kiểm soát thời gian hợp tác (Cooperative Time Management) tại `chess_ai/search/alphabeta.py`.
  * Đóng gói tác tử `chess_ai/agents/level3_alphabeta.py`.
* **TV2:**
  * Chạy thử nghiệm trích xuất $5.000$ ván cờ đầu tiên từ Lichess Elite DB ra `np.memmap` (`uint8` cho states, bảo toàn chính xác kênh 17) và viết `chess_ai/etl/memmap_dataset.py`.
* **TV3:**
  * Xây dựng khung điều phối giải đấu `chess_ai/arena/tournament.py` nạp 50 thế cờ khai cuộc từ `resources/openings_50.epd` (đấu 2 ván đổi màu/khai cuộc = 100 ván/cặp).
  * Thiết lập tiến trình/luồng phụ (Worker Process/Thread) trong Pygame để giao diện không bị đơ do GIL khi AI suy nghĩ.
  * **Chạy Pilot Benchmark:** Đo tốc độ duyệt thực tế (NPS của Cấp 3) trên máy trạm mục tiêu để ghi nhận mốc cơ sở thực nghiệm.
* **CỘT MỐC NGHIỆM THU MILESTONE 1 (Cuối Ngày 4):**
  * $100\%$ bài kiểm thử `TC01` – `TC35` vượt qua thành công.
  * Tác tử Cấp 2 đạt tỷ lệ điểm số $S \ge 0.90$ trước Cấp 1 trong loạt 100 ván (50 khai cuộc $\times 2$ màu).
  * Giao diện Pygame chơi được hoàn chỉnh chế độ Người đấu Máy và Máy đấu Máy với Cấp 1, Cấp 2, Cấp 3.

---

### PHA 2: ETL TOÀN DIỆN, TỐI ƯU HÓA CẤP 3 & HUẤN LUYỆN RESNET (NGÀY 5 – 7)
*Mục tiêu pha:* Trích xuất đầy đủ $\approx 1.000.000$ mẫu không trùng lặp vào `np.memmap`, hoàn thiện Cấp 3 (Zobrist TT + Move Ordering + Quiescence Search), huấn luyện mạng Dual-Head ResNet Cấp 5 và hoàn tất kiểm thử Arena (`TC52–TC61`).

#### Ngày 5: Bảng Băm Zobrist, Move Ordering & Trích Xuất 1M Mẫu Memmap
* **TV1:**
  * Xây dựng `chess_ai/core/zobrist.py`: Tạo bảng băm 64-bit và cấu trúc Transposition Table lưu trữ cờ `EXACT`, `LOWERBOUND`, `UPPERBOUND`, `best_move`, `depth`.
  * Cài đặt `chess_ai/search/move_ordering.py`: Sắp xếp nước đi ưu tiên: PV-move từ TT $\rightarrow$ MVV-LVA $\rightarrow$ Killer Moves (2 slots/ply) $\rightarrow$ History Heuristic.
* **TV2:**
  * Chạy trích xuất ngoại tuyến trọn vẹn từ Lichess Elite Database ($\ge 50.000$ ván hợp lệ $\to \approx 1.000.000$ mẫu đã khử trùng Zobrist) vào `data/processed/{train,val,test}_{states,actions,values}.dat`.
  * Xuất tệp `dataset_metadata.json` ghi nhận số ván, số mẫu mỗi tập, số mẫu trùng lặp đã loại bỏ, và thời gian chạy ETL; giữ RAM $< 1.5\text{GB}$.
* **TV3:**
  * Bổ sung đồng hồ thi đấu đếm ngược (Chess Clocks) và bảng biên bản nước đi chuẩn đại số (SAN) trên `chess_ai/ui/desktop_pygame.py`.
  * Xây dựng `chess_ai/arena/pgn_exporter.py` xuất biên bản PGN chuẩn quốc tế và ghi vết `matches.jsonl`.

#### Ngày 6: Quiescence Search, Kiến Trúc Dual-Head ResNet & Bộ Test Arena
* **TV1:**
  * Xây dựng `chess_ai/search/quiescence.py`: Mở rộng tìm kiếm tĩnh cho chuỗi bắt quân và nước giải chiếu (`max_plies: 8`) kèm kỹ thuật cắt tỉa Delta Pruning.
  * Tích hợp đầy đủ Iterative Deepening + Move Ordering + Transposition Table + Quiescence Search vào `chess_ai/agents/level3_alphabeta.py`.
* **TV2:**
  * Xây dựng `chess_ai/models/dual_resnet.py` khớp `configs/model_resnet.yaml`: Đầu vào `(18, 8, 8)`, thân 6 Residual Blocks (64 filters), Policy Head (Conv $1\times 1$ với **16 filters** $\to$ BN $\to$ ReLU $\to$ Linear 4096), Value Head (Conv $1\times 1$ với 1 filter $\to$ Linear 64 $\to$ Linear 1 $\to$ Tanh).
  * Xây dựng `chess_ai/models/loss.py`: Kết hợp Cross-Entropy và MSE (sử dụng `AdamW` với `weight_decay: 1e-4`, đặt `l2_weight: 0.0` để không phạt L2 hai lần).
* **TV3:**
  * Xây dựng `chess_ai/arena/metrics.py`: Hiện thực hóa công thức **Khoảng tin cậy Wilson Score 95%** và **Paired-Opening Bootstrap 95%** ($10.000$ resamples), đo độ trễ $p95$, NPS/SPS, và các chỉ số phong cách (Move Agreement Rate, Castling Rate).
  * Viết và chạy bộ kiểm thử `tests/test_arena_metrics.py` (`TC52` – `TC61`, bao gồm `TC61` kiểm định toàn bộ 50 thế cờ trong `resources/benchmark_positions.epd`).

#### Ngày 7: Huấn Luyện Mạng ResNet Cấp 5 & Đo Kiểm Chiến Thuật WAC-50
* **TV1:**
  * Chạy kiểm định tác tử Cấp 3 trên bộ 50 thế cờ chiến thuật chuẩn `resources/benchmark_positions.epd` (`WAC.001` – `WAC.050`).
  * Đối chiếu thực nghiệm giữa 5 cấu hình (`Config_A` $\to$ `Config_D`) để xác nhận hiệu quả cắt giảm nút duyệt của Move Ordering + TT và mức tăng độ chính xác chiến thuật của Quiescence Search.
* **TV2:**
  * Vận hành `chess_ai/training/train_supervised.py` với `torch.amp.autocast("cuda", dtype=torch.float16)` và `GradScaler`:
    * Lưu trữ checkpoints chuẩn (`best_model.pt`, `latest.pt`) có đầy đủ `scheduler_state_dict`, `scaler_state_dict`, `model_config` và ghi nhận tổng thời gian huấn luyện, đỉnh RAM/VRAM vào `results/training/training_metrics.json`.
  * Đóng gói `chess_ai/agents/level5_supervised.py` (Greedy $\arg\max_{a \in \text{legal}} P(s, a)$ có áp Legal Action Masking).
* **TV3:**
  * Tích hợp menu chọn cấp độ AI trực quan và hiển thị thanh đánh giá thế trận (Evaluation Bar) trên `chess_ai/ui/desktop_pygame.py`.
* **CỘT MỐC NGHIỆM THU MILESTONE 2 (Cuối Ngày 7):**
  * Dữ liệu `np.memmap` hoàn tất ($0\%$ trùng lặp Zobrist giữa Train/Val/Test), bộ test `TC52` – `TC61` vượt qua $100\%$.
  * Tác tử Cấp 3 chứng minh giảm $\ge 30\%$ số nút duyệt nhờ TT + Move Ordering và cải thiện rõ rệt tỷ lệ giải đúng WAC-50 khi bật Quiescence Search.
  * Mạng Dual-Head ResNet Cấp 5 đạt Top-1 Policy Accuracy $\ge 35\%$ và Value MSE thấp hơn rõ rệt so với baseline dự đoán hằng số $0$ trên tập Test độc lập.

---

### PHA 3: ĐỘNG CƠ MCTS TÁCH BIỆT (L4 & L7) & TÁC TỬ MỞ RỘNG L6, L8 (NGÀY 8 – 9)
*Mục tiêu pha:* Triển khai độc lập không xung đột hai động cơ MCTS (`mcts_uct.py` cho Cấp 4 do TV1 viết và `mcts_puct.py` cho Cấp 7 do TV2 viết), hoàn thiện tác tử mở rộng Cấp 6 (KRK RL) và Cấp 8 (AlphaZero Self-play), khép kín toàn bộ 61 bài kiểm thử.

#### Ngày 8: Vanilla MCTS UCT (Cấp 4 - TV1) & Guided MCTS PUCT (Cấp 7 - TV2)
* **TV1:**
  * Cài đặt cấu trúc nút cơ sở `chess_ai/search/mcts_node.py` (hỗ trợ tái sử dụng cây con `subtree_reuse`).
  * Cài đặt `chess_ai/search/mcts_uct.py` và đóng gói `chess_ai/agents/level4_mcts.py`:
    * Thực hiện 4 bước Selection (UCB1), Expansion, Cut-off Simulation (`max_rollout_depth: 60` plies, định giá điểm cắt bằng $\tanh(\text{eval}_{\text{cp}} / 600.0)$), và Backpropagation bằng `push()`/`pop()` trực tiếp.
* **TV2:**
  * Cài đặt `chess_ai/search/mcts_puct.py` và đóng gói `chess_ai/agents/level7_guided_mcts.py`:
    * Kết hợp xác suất tiên nghiệm $P(s, a)$ từ Policy Head và giá trị lá $V(s)$ từ Value Head của Cấp 5 theo công thức PUCT.
    * **Tách bạch nghiêm ngặt 2 chế độ vận hành:**
      * `eval_mode` (dùng khi thi đấu Arena & Ablation): Tắt nhiễu Dirichlet ($\epsilon = 0$), chọn nước đi tất định $\arg\max_a N(s, a)$ ($\tau = 0$).
      * `selfplay_mode` (dùng cho Cấp 8): Bật nhiễu Dirichlet $\text{Dir}(\alpha=0.3), \epsilon=0.25$ tại nút gốc, lấy mẫu theo nhiệt độ $\tau = 1.0$ trong 20 plies đầu.
* **TV3:**
  * Viết bộ kiểm thử tác tử `tests/test_agents_sanity.py` (`TC36` – `TC43`, `TC45`, `TC47` – `TC51`).
  * Kiểm thử tương thích Cấp 1–5 và Cấp 7 trên giao diện Pygame.

#### Ngày 9: Tác tử Học Tăng Cường KRK (Cấp 6 - TV1) & Vòng Lặp Self-Play (Cấp 8 - TV2)
* **TV1:**
  * Triển khai `chess_ai/agents/level6_reinforcement.py`: Cài đặt thuật toán Tabular Q-Learning (kèm đối xứng 8 chiều của thế cờ 3 quân $K_W, R_W, K_B$) và REINFORCE trên bài toán cờ tàn chuẩn FIDE **KRK** sử dụng trực tiếp `ChessEnv`.
  * Huấn luyện $20.000$ episodes, lưu bảng Q / trọng số và xuất dữ liệu đường cong học tập (Learning Curve: tỷ lệ chiếu hết thành công & số bước trung bình theo episode).
* **TV2:**
  * Triển khai `chess_ai/agents/level8_alphazero.py`: Thiết lập vòng lặp tự chơi (Self-play Loop bằng `mcts_puct` ở `selfplay_mode`), bộ đệm `ReplayBuffer` lưu $(s_t, \mathbf{\pi}_t, z_t)$, cập nhật mạng và cơ chế **Checkpoint Arena Gating** (chỉ cập nhật mô hình khi thắng $\ge 55\%$ trong loạt đấu đối đầu 40 ván).
  * Chạy thực nghiệm Self-play Fine-tuning khởi tạo từ trọng số Cấp 5 (kèm kiểm nghiệm hội tụ trên thế cờ tàn thu gọn).
* **TV3:**
  * Hoàn thiện `TC44` (Cấp 6) và `TC46` (Cấp 8) trong `tests/test_agents_sanity.py`, đạt trọn vẹn **61/61 test cases xanh**.
  * Xây dựng kịch bản `scripts/reproduce_experiment.py` (hỗ trợ `--from-logs` và `--full`) và khung notebook `notebooks/analysis.ipynb`.
* **CỘT MỐC NGHIỆM THU MILESTONE 3 (Cuối Ngày 9):**
  * Toàn bộ **61/61 bài kiểm thử tự động (`pytest`)** vượt qua $100\%$.
  * Toàn bộ 8 cấp độ tác tử (Cấp 1 $\to$ Cấp 8) hoạt động ổn định, nạp được vào giao diện Pygame và sẵn sàng bước vào giải đấu.

---

### PHA 4: VẬN HÀNH GIẢI ĐẤU 100 VÁN/CẶP (50 KHAI CUỘC) & 02 ABLATION STUDIES (NGÀY 10 – 12)
*Mục tiêu pha:* Vận hành giải đấu Round-Robin 100 ván/cặp trên 50 thế cờ khai cuộc cân bằng, hoàn thành 02 bài Ablation Studies mở rộng, thu thập log PGN/JSONL và tính toán khoảng tin cậy Wilson 95% + Paired Bootstrap 95%.

#### Ngày 10: Vận Hành Ablation 1 (5 Cấu Hình + Quét Độ Sâu) & Giải Đấu Chế Độ Workload
* **TV1 (Chủ trì Ablation 1):**
  * Thực thi `chess_ai/experiments/ablation_search.py` trên bộ 50 thế cờ chiến thuật chuẩn `resources/benchmark_positions.epd` (`WAC.001` – `WAC.050`):
    * **Phần 1A:** So sánh 5 cấu hình lũy tiến ở $d=3$: `Config_A` (Minimax) $\to$ `Config_B` (Alpha-Beta) $\to$ `Config_C1` (+Move Ordering) $\to$ `Config_C2` (+Zobrist TT) $\to$ `Config_D` (+Quiescence Search).
    * **Phần 1B:** Khảo sát độ nhạy theo độ sâu tìm kiếm $d \in \{2, 3, 4\}$.
* **TV2 & TV3:**
  * Vận hành giải đấu Round-Robin **100 ván/cặp** (50 khai cuộc từ `resources/openings_50.epd` $\times 2$ màu Trắng/Đen, điều phối qua 10 Random Seeds chuẩn) ở **Chế độ 1 (Workload-Control: $10.000$ nodes cho Cấp 3, $400$ sims cho Cấp 4 & Cấp 7)** sử dụng đa tiến trình song song.
  * Ghi vết đầy đủ vào `results/round_robin/matches.jsonl` và `results/round_robin/games.pgn`.

#### Ngày 11: Vận Hành Ablation 2 (Khảo Sát $N_{\text{sim}}$ Cấp 4 & Cấp 7) & Giải Đấu Time-Control
* **TV3 & TV2 (Chủ trì Ablation 2):**
  * Thực thi `chess_ai/experiments/ablation_mcts.py`:
    * **Phần 2A (Khảo sát độ nhạy $N_{\text{sim}}$ cho cả Cấp 4 và Cấp 7):** Chạy cả **Cấp 4 (Vanilla MCTS UCT)** và **Cấp 7 (Guided MCTS PUCT)** qua các mức ngân sách $N_{\text{sim}} \in \{50, 100, 200, 400, 800, 1000\}$ (50 ván/mức trên 25 khai cuộc $\times 2$ màu).
    * **Phần 2B (Bóc tách Policy Prior & Value Head):** Đấu đối kháng 100 ván/cặp giữa 3 biến thể: Cấp 5 (Policy-only Greedy) vs Value-only MCTS (Uniform Prior + Value Head) vs Cấp 7 đầy đủ (Policy Prior + Value Head).
* **TV1 & TV3:**
  * Vận hành giải đấu đối kháng ở **Chế độ 2 (Time-Control: $1.0\text{s}$/nước đi)** theo cơ chế **tuần tự đơn luồng (Single-instance Sequential)** để đo đạc chính xác sức cờ thời gian thực và độ trễ $p95$ mà không bị nhiễu tranh chấp CPU/GPU.

#### Ngày 12: Tổng Hợp Thống Kê Wilson/Bootstrap, Notebook Phân Tích & Giải Thích Chiến Thuật
* **TV1:**
  * Trích xuất từ biên bản PGN thực tế (`results/round_robin/games.pgn`) các tình huống chiến thuật và sai lầm điển hình (ví dụ: bẫy bắt quân vượt chân trời khi tắt Quiescence Search, tình huống Cấp 4 bị cắt ngưỡng rollout, và ván đấu Cấp 7 vượt qua Cấp 3/Cấp 5 nhờ tầm nhìn PUCT).
* **TV2:**
  * Tổng hợp biểu đồ huấn luyện của Cấp 5 (Loss, Top-1/Top-3 Accuracy, Value MSE so với baseline $0$), đường cong học tập KRK của Cấp 6, và kết quả Arena Gating của Cấp 8.
* **TV3:**
  * Hoàn thiện `notebooks/analysis.ipynb` và chạy `python scripts/reproduce_experiment.py --from-logs`:
    * Xuất ma trận đối đầu đầy đủ ($W/D/L$, điểm $S$, **Wilson Score 95% CI**, **Paired-Opening Bootstrap 95% CI**, phân tích $S_{\text{White}}$ vs $S_{\text{Black}}$ và độ lệch chuẩn theo 10 seeds).
    * Xuất bảng chi phí tính toán (NPS/SPS, độ trễ trung bình, $p95$ latency, RAM/VRAM, thời gian train) và bảng chỉ số phong cách (Move Agreement Rate, Castling Rate cho RQ4).
* **CỘT MỐC NGHIỆM THU MILESTONE 4 (Cuối Ngày 12):**
  * Hoàn thành $100\%$ các cặp đấu Round-Robin ($100$ ván/cặp trên 50 khai cuộc cân bằng) và cả 2 bài Ablation mở rộng.
  * Chạy `python scripts/reproduce_experiment.py --from-logs` và `notebooks/analysis.ipynb` tái tạo thành công toàn bộ bảng số liệu và biểu đồ trong $< 30$ giây.

---

### PHA 5: BÁO CÁO KHOA HỌC 5 CHƯƠNG, SLIDE, POSTER & ĐÓNG GÓI (NGÀY 13 – 15)
*Mục tiêu pha:* Soạn thảo báo cáo khoa học 5 chương chuẩn mực, trả lời định lượng 4 câu hỏi nghiên cứu (RQ1–RQ4), hoàn thiện Slide, Poster, Video Demo và đóng gói tệp ZIP đúng quy cách đề bài.

#### Ngày 13: Soạn Thảo Báo Cáo Khoa Học (Chương 1, 2, 3)
* **TV1:**
  * Viết Chương 2 (Cơ sở Lý thuyết): Trình bày chính xác độ phức tạp không gian trạng thái ($\approx 10^{44}\text{--}10^{47}$) và độ phức tạp cây trò chơi Shannon ($\approx 10^{120}$); toán học Negamax, trường hợp tốt nhất $\mathcal{O}(b^{d/2})$ của Alpha-Beta, Move Ordering, Zobrist TT, Quiescence Search, và UCT.
  * Viết phần thiết kế Cấp 1–4 và Cấp 6 trong Chương 3.
* **TV2:**
  * Viết phần Học sâu, Cấp 5, Cấp 7 và Cấp 8 trong Chương 3: Mã hóa Tensor 18 kênh góc nhìn bên đi (`[18, 8, 8]`), mạng Dual-Head ResNet, bảng đối chiếu thành phần Đã/Chưa hiện thực của Cấp 7 so với AlphaGo gốc, pipeline Lichess Elite DB và khử trùng Zobrist.
* **TV3:**
  * Viết Chương 1 (Giới thiệu): Ghi nhận phê duyệt đề tài từ giảng viên, xác lập 4 câu hỏi nghiên cứu (RQ1–RQ4), phạm vi 8 cấp độ và danh mục sản phẩm bàn giao.
  * Xuất toàn bộ biểu đồ vector (`.pdf`/`.png` 300 DPI) từ `notebooks/analysis.ipynb` vào `results/figures/`.

#### Ngày 14: Soạn Thảo Báo Cáo Khoa Học (Chương 4, 5) & Kiểm Chéo Số Liệu
* **TV1, TV2 & TV3:**
  * Viết Chương 4 (Thực nghiệm, Kết quả & Thảo luận): Trình bày ma trận đối đầu 100 ván/cặp kèm khoảng tin cậy Wilson & Paired Bootstrap 95%, bảng chi phí tài nguyên & thời gian huấn luyện, đường cong học tập Cấp 5/6/8, kết quả 2 bài Ablation Studies, phân tích định lượng phong cách (RQ4) và phân tích ván đấu thực tế trích từ log.
  * Viết Chương 5 (Kết luận & Hướng phát triển): Trả lời trực diện RQ1–RQ4 bằng số liệu thực nghiệm; phân tích hạn chế (nhiễu nhãn kết cục ván cờ đối với Value Head, giới hạn phần cứng đối với AlphaZero Tabula Rasa trên bàn cờ $8 \times 8$) và hướng mở rộng.
  * Cập nhật hoàn chỉnh `README.md` (hướng dẫn cài đặt, train, evaluate, reproduce, demo).

#### Ngày 15: Slide Thuyết Trình, Poster, Video Demo & Đóng Gói Nộp Bài
* **TV1:**
  * Rà soát liêm chính học thuật, trích dẫn chuẩn (bao gồm sách *Mark Liu, AlphaGo Simplified, CRC Press*) và kiểm tra chạy sạch `pytest` + `reproduce_experiment.py` trên môi trường mới.
* **TV2:**
  * Thiết kế Poster tóm tắt kết quả nghiên cứu (khổ A0) trình bày trực quan kiến trúc Clean Architecture, Tensor 18 kênh, ma trận đối đầu và 2 bài Ablation.
* **TV3:**
  * Thiết kế bộ Slide thuyết trình (15–20 slides) và quay Video Demo ($3 \div 5$ phút) thể hiện 2 chế độ Người đấu Máy và Máy đấu Máy trên Pygame Desktop.
  * Đóng gói toàn bộ sản phẩm thành tệp ZIP đúng quy định tại Mục 3.7 của đề bài:  
    `NhomXX_Chess_TTNTNC_HoTen_MaSV.zip` (chứa mã nguồn, `configs/`, `tests/`, `notebooks/`, `checkpoints/`, `logs/`, `docs/`, `README.md`).
* **CỘT MỐC NGHIỆM THU MILESTONE 5 (Cuối Ngày 15):**
  * Hồ sơ đề tài hoàn chỉnh $100\%$, sẵn sàng nộp bài và bảo vệ trước hội đồng giảng viên.

---

## PHẦN 3: KẾ HOẠCH QUẢN TRỊ RỦI RO & PHÒNG NGỪA (RISK CONTINGENCY)

| Rủi Ro Kỹ Thuật Tiềm Tàng | Mức Độ | Biện Pháp Phòng Ngừa Kỹ Thuật Đã Thiết Lập | Kế Hoạch Dự Phòng (Fallback) |
| :--- | :---: | :--- | :--- |
| **Giao diện Pygame bị đơ do GIL khi AI tính toán bằng Python** | Cao | Tách tác vụ `select_move()` sang Worker Process (`multiprocessing`) hoặc kiểm tra nhường luồng định kỳ, giao tiếp qua Queue. | Ở chế độ Time-Control, cơ chế Cooperative Time Check đảm bảo AI luôn trả kết quả đúng hạn. |
| **Các tác tử tất định đấu lặp lại cùng 1 ván cờ** | Cao | Sử dụng bộ 50 thế cờ khai cuộc cân bằng (`resources/openings_50.epd`), mỗi khai cuộc đấu 2 ván đổi màu Trắng/Đen. | Kiểm tra tự động tính đa dạng biên bản ván đấu trong `TC53`. |
| **Nhiễu thời gian khi chạy giải đấu song song** | Trung bình | Chạy song song đa tiến trình cho chế độ Workload-Control; chạy tuần tự đơn luồng cho chế độ Time-Control. | Ưu tiên hoàn tất ma trận toàn diện ở Workload-Control và các cặp trọng tâm ở Time-Control. |
| **Mạng ResNet bị gián đoạn phiên huấn luyện** | Trung bình | Tích hợp `CheckpointManager` lưu định kỳ cả `optimizer`, `scheduler`, `scaler` (`--resume checkpoints/latest.pt`). | Huấn luyện trên Google Colab T4 từ tệp nén `np.memmap` và tải checkpoint `.pt` về máy trạm. |
| **Xung đột Git giữa TV1 và TV2 trên module MCTS** | Trung bình | Tách hoàn toàn thành 3 tệp độc lập: `mcts_node.py`, `mcts_uct.py` (TV1) và `mcts_puct.py` (TV2). | Mọi thay đổi trên `mcts_node.py` phải đóng băng từ cuối Ngày 8. |

---

## PHẦN 4: TIÊU CHÍ NGHIỆM THU ĐÓNG GÓP THÀNH VIÊN (CONTRIBUTION RUBRIC)

Điểm số đóng góp của từng thành viên ($33.3\% - 33.3\% - 33.4\%$) được xác định minh bạch dựa trên 4 minh chứng định lượng nộp kèm báo cáo:
1. **Lịch sử Commit Git cá nhân:** Tần suất commit đều đặn trên nhánh phụ trách (`feature/tv1-*`, `feature/tv2-*`, `feature/tv3-*`).
2. **Hoàn thành Module & Unit Tests:** Vượt qua $100\%$ các bài kiểm thử tự động (`TC01` – `TC61`) và Quality Gates của phân hệ phụ trách.
3. **Đóng góp Thực nghiệm & Báo cáo:** Trực tiếp vận hành kịch bản thực nghiệm, đóng góp vào `notebooks/analysis.ipynb`, báo cáo 5 chương, slide và poster.
4. **Kỷ luật Tiến độ:** Bàn giao đúng hạn tại 5 cột mốc Milestone của Sprint 15 ngày.
