# HỒ SƠ QUYẾT ĐỊNH KIẾN TRÚC & KỸ THUẬT (DECISIONS.md)

**Mã tài liệu:** `ADR_LOG_2026`  
**Dự án:** Multi-Level Chess AI Engine ($8 \times 8$ FIDE)  
**Môn học:** Trí tuệ Nhân tạo Nâng cao  
**Trạng thái:** Đã phê duyệt & Đóng băng (Frozen Baseline)  
**Nguồn sự thật cấu hình (Single Source of Truth):** Thư mục `configs/` (`default_config.yaml`, `model_resnet.yaml`, `search_params.yaml`)

---

## MỤC ĐÍCH TÀI LIỆU

Tài liệu này ghi nhận toàn bộ các quyết định kiến trúc và thiết kế kỹ thuật (Architecture Decision Records - ADR) đã được nhóm thống nhất sau quá trình rà soát chéo với đề bài gốc (`docs/TTNTNC_De_tai_cuoi_ky_Game_AI.pdf`). Mọi tài liệu đặc tả trong `docs/` và mã nguồn triển khai đều bắt buộc tuân thủ các quyết định tại đây, lấy các tệp YAML trong `configs/` làm nguồn sự thật duy nhất về tham số.

---

## ADR-01: Phê Duyệt Đề Tài Cờ Vua $8 \times 8$ FIDE & Phạm Vi 8 Cấp Độ Tác Tử

* **Bối cảnh:** Mục 3.1 và 3.3 của đề bài quy định các trò chơi ngoài danh sách gợi ý (như Cờ vua $8 \times 8$ chuẩn FIDE) phải được giảng viên chấp thuận trước khi triển khai, đồng thời ưu tiên chứng minh tác dụng của từng thành phần hơn là liệt kê thuật toán hình thức.
* **Quyết định:**
  1. Đề tài **Cờ vua chuẩn FIDE $8 \times 8$** cùng phạm vi triển khai **đầy đủ 8 cấp độ tác tử (Cấp 1 đến Cấp 8)** đã được giảng viên phê duyệt chính thức. Thông tin phê duyệt được ghi nhận rõ trong `README.md` và Chương 1 của Báo cáo.
  2. Giữ trọn vẹn 8 cấp độ tác tử, nhưng định hình phạm vi thực nghiệm thực tế cho các cấp mở rộng:
     * **Cấp 1 – 5 & Cấp 7:** Thi đấu đầy đủ trên bàn cờ chuẩn $8 \times 8$ FIDE.
     * **Cấp 6 (Reinforcement Learning):** Huấn luyện Tabular Q-learning & Policy Gradient (REINFORCE) trên bài toán cờ tàn chuẩn FIDE **KRK (Vua + Xe Trắng đấu Vua Đen)** sử dụng trực tiếp `ChessEnv` (khởi tạo từ FEN KRK), đảm bảo hội tụ rõ ràng và xuất được đường cong học tập (Learning Curve) chuẩn mực.
     * **Cấp 8 (AlphaZero-inspired):** Hiện thực đầy đủ vòng lặp Self-play PUCT + Replay Buffer + Checkpoint Arena (ngưỡng thắng $55\%$), vận hành thực nghiệm theo cơ chế **Self-play Fine-tuning khởi tạo từ Cấp 5** (và kiểm thử hội tụ trên thế cờ tàn thu gọn), kết hợp phân tích lý thuyết định lượng ở Chương 5 về rào cản chi phí phần cứng của việc huấn luyện Tabula Rasa từ con số 0 trên bàn cờ $8 \times 8$.
  3. **Làm rõ quy định điểm khuyến khích (+10 điểm):** Ghi nhận đúng theo Rubric đề bài: trần điểm cộng tối đa là $+10$ cho toàn bộ các hạng mục nâng cao (AlphaGo/AlphaZero-inspired có thực nghiệm thuyết phục, tối ưu hóa tìm kiếm, giải thích chiến thuật), không phải điểm cố định riêng của Cấp 7.

---

## ADR-02: Biểu Diễn Trạng Thái Bàn Cờ Chuẩn Hóa 18 Kênh Góc Nhìn Bên Đi (`[18, 8, 8]`)

* **Bối cảnh:** Thiết kế cũ ("Phương án C - 14 kênh không lật bàn cờ") mắc 3 lỗi kỹ thuật nghiêm trọng:
  1. Công thức kênh nhập thành $\frac{1}{4}(1\cdot\text{KS} + 2\cdot\text{QS} + 4\cdot\text{oKS} + 8\cdot\text{oQS})$ cho giá trị tối đa $3.75$ (vượt ngoài $[0, 1]$) và bị cắt cụt mất phần thập phân khi lưu vào mảng `np.int8`.
  2. Gộp 4 quyền nhập thành độc lập vào 1 số vô hướng làm mạng CNN rất khó tách từng quyền.
  3. Thiếu thông tin ô bắt Tốt qua đường (En-passant) và bộ đếm quy tắc 50 nước (Halfmove clock); đồng thời việc không lật bàn cờ buộc mạng phải tự học hai hướng tiến ngược nhau của Tốt qua cờ màu.
* **Quyết định:** Chuyển sang **Biểu diễn Chuẩn hóa Góc nhìn Bên Đi (Canonical Side-to-Move Perspective)** với kích thước chuẩn PyTorch CHW là **`(18, 8, 8)`**:
  * **Quy tắc lật tọa độ chuẩn hóa (`canonical_flip: true`):**
    * Khi tới lượt Trắng (`board.turn == chess.WHITE`): Giữ nguyên tọa độ ô `sq` ($a1 = 0 \dots h8 = 63$, hàng $r = \text{sq} // 8$, cột $c = \text{sq} \% 8$).
    * Khi tới lượt Đen (`board.turn == chess.BLACK`): Lật đối xứng dọc bàn cờ qua phép XOR hàng `sq_canon = sq ^ 56` (hàng $r_{\text{canon}} = 7 - r$, cột $c_{\text{canon}} = c$). Nhờ đó, bên tới lượt đi luôn xuất phát từ phía dưới (hàng $0, 1$) và tiến quân lên phía trên (hàng $6, 7$).
  * **Cấu trúc 18 Kênh Đặc trưng (`(18, 8, 8)`):**
    * **Kênh 0 – 5:** Quân của bên tới lượt đi (`Active`: P, N, B, R, Q, K) — mặt phẳng nhị phân $\{0, 1\}$.
    * **Kênh 6 – 11:** Quân của đối thủ (`Opponent`: P, N, B, R, Q, K) — mặt phẳng nhị phân $\{0, 1\}$.
    * **Kênh 12:** Quyền nhập thành cánh Vua của bên đi (`Active Kingside Castling`) — toàn mặt phẳng là $1$ nếu có quyền, $0$ nếu mất quyền.
    * **Kênh 13:** Quyền nhập thành cánh Hậu của bên đi (`Active Queenside Castling`) — nhị phân $\{0, 1\}$.
    * **Kênh 14:** Quyền nhập thành cánh Vua của đối thủ (`Opponent Kingside Castling`) — nhị phân $\{0, 1\}$.
    * **Kênh 15:** Quyền nhập thành cánh Hậu của đối thủ (`Opponent Queenside Castling`) — nhị phân $\{0, 1\}$.
    * **Kênh 16:** Ô đích bắt Tốt qua đường hợp lệ (`En-passant target square`) — giá trị $1$ tại đúng ô đích EP (đã lật chuẩn hóa theo bên đi) nếu tồn tại nước bắt EP hợp lệ, $0$ tại mọi ô còn lại.
    * **Kênh 17:** Bộ đếm quy tắc 50 nước chuẩn hóa (`Halfmove clock`) — toàn mặt phẳng nhận giá trị thực $\min(\text{halfmove\_clock} / 100.0, 1.0) \in [0.0, 1.0]$. Khi lưu trữ nén trong `np.memmap` kiểu `uint8`, kênh 17 lưu số nguyên $\min(\text{halfmove\_clock}, 100) \in [0, 100]$ và chia $100.0$ khi chuyển sang `float32` trên GPU, bảo toàn chính xác tuyệt đối $100\%$ thông tin mà không mất dữ liệu thập phân.

---

## ADR-03: Không Gian Hành Động Phẳng 4096 Chiều & Quy Tắc Phong Cấp

* **Quyết định:**
  1. **Mã hóa nước đi theo hệ quy chiếu bên đi:**
     * Với nước đi từ ô nguồn $u$ đến ô đích $v$ ($u, v \in [0, 63]$):
       $$u_{\text{canon}} = \begin{cases} u & \text{nếu Trắng đi} \\ u \oplus 56 & \text{nếu Đen đi} \end{cases}, \quad v_{\text{canon}} = \begin{cases} v & \text{nếu Trắng đi} \\ v \oplus 56 & \text{nếu Đen đi} \end{cases}$$
     * Chỉ số hành động phẳng: $\text{action\_idx} = u_{\text{canon}} \times 64 + v_{\text{canon}} \in [0, 4095]$.
  2. **Quy tắc giải mã phong cấp (Promotion Decoding Rule):**
     * Trong giai đoạn trích xuất dữ liệu huấn luyện (ETL): Bật cấu hình `drop_underpromotions: true` để loại bỏ các mẫu nước đi phong cấp dưới hiếm gặp (Mã, Tượng, Xe — chiếm $< 0.5\%$), giữ tính nhất quán đơn ánh cho nhãn phong Hậu.
     * Trong giai đoạn suy luận (Inference / Masking): Khi xây dựng mặt nạ hợp lệ $M(s) \in \{0, 1\}^{4096}$, một chỉ số $a = u_{\text{canon}} \times 64 + v_{\text{canon}}$ được bật $M_a = 1$ nếu tồn tại ít nhất một nước đi hợp lệ từ $u \to v$. Khi giải mã chỉ số $a$ thành `chess.Move`, nếu $(u, v)$ là nước phong cấp và nước phong Hậu `chess.Move(u, v, promotion=chess.QUEEN)` hợp lệ thì chọn phong Hậu; ngược lại fallback chọn nước hợp lệ đầu tiên khớp cặp $(u, v)$ trong `board.legal_moves`.

---

## ADR-04: Nguồn Dữ Liệu Lichess Elite Database & Khử Trùng Lặp (Deduplication)

* **Bối cảnh:** Đọc stream trực tiếp tệp dump Lichess tháng ($\sim 30\text{GB}$ nén) qua HTTP mất nhiều giờ và dễ đứt kết nối vì tỷ lệ ván cờ có cả hai bên $\text{ELO} \ge 2100$ rất thấp. Ngoài ra, đề bài yêu cầu mô tả rõ cách loại trùng dữ liệu (Deduplication).
* **Quyết định:**
  1. Sử dụng **Lichess Elite Database** (`database.nikonoel.fr` — kho dữ liệu chuẩn đã lọc sẵn các ván cờ Lichess giữa kỳ thủ $\text{ELO} \ge 2200/2100$, loại bỏ sẵn Bullet), tải tệp nén `.zip`/`.zst` về thư mục cục bộ `data/raw/` trước khi chạy ETL. Vẫn hỗ trợ đọc luồng `.pgn.zst` cục bộ hoặc HTTP làm tuỳ chọn mở rộng.
  2. **Lọc nhanh ở mức Header (`read_headers`):** Kiểm tra `WhiteElo`, `BlackElo`, `TimeControl` ($\text{base} + 40 \times \text{inc} \ge 180\text{s}$), `Result`, `Termination` (`Normal` hoặc `Time forfeit`) trước khi parse cây nước đi nhằm tăng tốc độ ETL gấp $5 \div 8$ lần.
  3. **Phân chia cấp độ ván đấu tất định (Deterministic Game-Level Split):** Dựa trên giá trị băm `int(hashlib.md5(game_id.encode()).hexdigest(), 16) % 100`: $< 80 \to \text{Train}$, $80..89 \to \text{Val}$, $90..99 \to \text{Test}$. Giải quyết mâu thuẫn giữa streaming và xáo trộn danh sách Game ID.
  4. **Khử trùng lặp thế cờ bằng Zobrist Hash (`deduplicate_zobrist: true`):** Duy trì tập khóa băm Zobrist 64-bit `seen_zobrist_hashes` trong quá trình trích xuất để đảm bảo không có bất kỳ thế cờ nào xuất hiện trùng lặp giữa tập Train, Validation và Test (triệt tiêu hoàn toàn Data Leakage ở cả cấp độ ván đấu lẫn cấp độ thế cờ).

---

## ADR-05: Giao Thức Giải Đấu 100 Ván/Cặp Với Bộ 50 Khai Cuộc Cân Bằng

* **Bối cảnh:** Các tác tử tất định (như Cấp 3 Alpha-Beta ở chế độ cố định độ sâu/nút và Cấp 5 Greedy Policy) không chịu ảnh hưởng của Random Seed. Nếu luôn xuất phát từ bàn cờ gốc ban đầu, 100 ván Trắng và 100 ván Đen sẽ lặp lại đúng 2 ván duy nhất. Bên cạnh đó, chạy 200 ván/cặp cho mọi cặp ở cả 2 chế độ vượt quá quỹ thời gian 2 ngày của Pha 4.
* **Quyết định:**
  1. **Quy mô chuẩn:** Thiết lập cố định **100 ván/cặp đấu** cho toàn bộ các cặp tác tử (đáp ứng chuẩn tối thiểu 100 ván/cặp ở Mục 3.5 của đề bài).
  2. **Bộ 50 thế cờ khai cuộc cân bằng (`resources/openings_50.epd`):** Tuyển chọn 50 thế cờ khai cuộc chuẩn FIDE sau $6 \div 8$ plies (đánh giá cân bằng trong khoảng $[-0.3, +0.3]$ pawn). Với mỗi thế cờ khai cuộc $i \in \{1 \dots 50\}$:
     * Ván $2i - 1$: Tác tử A cầm Trắng, Tác tử B cầm Đen.
     * Ván $2i$: Tác tử B cầm Trắng, Tác tử A cầm Đen.
  3. **Kiểm định thống kê kép:** Tính toán đồng thời **Khoảng tin cậy Wilson Score 95%** trên điểm số quy đổi $S = (W + 0.5D)/N$ và **Khoảng tin cậy Bootstrap 95% theo cặp khai cuộc** ($10.000$ lần lấy mẫu lại các cặp ván đối xứng) để xử lý chính xác các ván hòa ($0.5$) và tương quan nội tại theo từng khai cuộc.

---

## ADR-06: Thiết Kế Tìm Kiếm Cổ Điển, Tách File MCTS & Tách Chế Độ Thi Đấu / Tự Chơi

* **Quyết định:**
  1. **`ChessEnv.clone()` và Kiểm tra Kết thúc Nhanh:**
     * `ChessEnv.clone()` sao chép đầy đủ ngăn xếp lịch sử (`self.board.copy(stack=True)`) để bảo toàn khả năng `undo()` và phát hiện hòa lặp 3 lần (Threefold Repetition).
     * Cung cấp thêm phương thức `is_terminal_fast()` trong tìm kiếm nội bộ (kiểm tra chiếu hết, hòa pat, thiếu lực lượng, 50 nước mà không quét lại toàn bộ lịch sử lặp 3 lần tại mọi nút lá) kết hợp kiểm tra lặp khóa Zobrist trên đường đi hiện tại của cây tìm kiếm.
  2. **Iterative Deepening (ID) & Quản lý Thời gian Hợp tác:**
     * Tác tử Cấp 3 (`chess_ai/search/alphabeta.py`) bắt buộc triển khai **Iterative Deepening** ($d = 1, 2, \dots, d_{\max}$) kết hợp kiểm tra đồng hồ hợp tác (Cooperative Time Check mỗi $256$ nút duyệt). Khi chạm ngưỡng thời gian, thuật toán dừng ngay và trả về nước đi tốt nhất từ độ sâu hoàn chỉnh gần nhất.
  3. **Tách biệt Module MCTS để Tránh Xung đột Git:**
     * `chess_ai/search/mcts_node.py`: Định nghĩa lớp nút cây MCTS dùng chung và tiện ích tái sử dụng cây con (`subtree_reuse`).
     * `chess_ai/search/mcts_uct.py` (TV1 phụ trách - Cấp 4): Cài đặt UCT với giới hạn độ sâu mô phỏng `max_rollout_depth: 60` plies; nếu chạm trần chưa kết thúc ván, trả về giá trị lượng giá tĩnh chuẩn hóa $\tanh(\text{eval}_{\text{cp}} / 600.0) \in [-1, 1]$.
     * `chess_ai/search/mcts_puct.py` (TV2 phụ trách - Cấp 7 & 8): Cài đặt PUCT kết hợp Policy Prior $P(s, a)$ và Value Head $V(s)$.
  4. **Tách chế độ `eval_mode` và `selfplay_mode` trong Guided MCTS:**
     * Khi thi đấu giải hoặc đánh giá (`eval_mode`): Tắt hoàn toàn nhiễu Dirichlet ($\epsilon = 0$) và chọn nước đi tất định theo số lần thăm lớn nhất ($\tau \to 0$, tức $\arg\max_a N(s, a)$).
     * Khi tự chơi sinh dữ liệu ở Cấp 8 (`selfplay_mode`): Bật nhiễu Dirichlet $\text{Dir}(\alpha=0.3), \epsilon=0.25$ tại nút gốc và lấy mẫu theo nhiệt độ $\tau = 1.0$ trong $20$ plies (10 nước) đầu, sau đó giảm về $\tau = 0.1$.

---

## ADR-07: Chuẩn Hóa Bộ Benchmark Chiến Thuật WAC-50 & Mở Rộng Ablation

* **Quyết định:**
  1. Thay thế toàn bộ tệp `resources/benchmark_positions.epd` bằng **50 thế cờ chiến thuật chuẩn `WAC.001` – `WAC.050`** trích từ bộ *Win At Chess* gốc (Arasan `wacnew.epd`), đã kiểm định $100\%$ bằng `python-chess`.
  2. **Mở rộng Ablation 1 (Tầng Tìm kiếm Cổ điển):**
     * Tách thành **5 cấu hình lũy tiến** để cô lập rõ đóng góp của Transposition Table (TT):
       1. `Config_A`: Minimax thuần ($d=3$).
       2. `Config_B`: Minimax + Alpha-Beta ($d=3$).
       3. `Config_C1`: Alpha-Beta + Move Ordering (MVV-LVA, Killer).
       4. `Config_C2`: Alpha-Beta + Move Ordering + Zobrist Transposition Table (kiểm chứng trực tiếp `TC41`).
       5. `Config_D`: Cấu hình Cấp 3 đầy đủ (`Config_C2` + Quiescence Search).
     * Bổ sung khảo sát độ nhạy theo độ sâu tìm kiếm $d \in \{2, 3, 4\}$ theo đúng yêu cầu Mục 3.5 của đề bài.
     * Lưu ý học thuật rõ ràng: `Config_A`, `Config_B`, `Config_C1`, `Config_C2` ở cùng độ sâu cố định trả về cùng giá trị minimax (chỉ khác biệt đột biến về số nút duyệt và độ trễ), trong khi `Config_D` (Quiescence Search) mới tạo ra bước nhảy về độ chính xác giải thế cờ chiến thuật nhờ giảm thiểu hiện tượng mù chân trời (Horizon Effect).
  3. **Mở rộng Ablation 2 (Tầng MCTS & Học sâu):**
     * **Phần A:** Khảo sát độ nhạy số lượt mô phỏng $N_{\text{sim}} \in \{50, 100, 200, 400, 800, 1000\}$ cho **cả Cấp 4 (Vanilla MCTS UCT) lẫn Cấp 7 (Guided MCTS PUCT)** để đáp ứng trọn vẹn yêu cầu khảo sát số mô phỏng của Cấp 4 trong đề bài.
     * **Phần B:** So sánh bóc tách thành phần dẫn hướng nơ-ron (100 ván/cặp): Cấp 5 (Policy-only Greedy) vs Guided MCTS chỉ dùng Value Head (Uniform Prior + Value) vs Cấp 7 đầy đủ (Policy Prior + Value Head).

---

## ADR-08: Tái Cân Bằng Phân Công Nhiệm Vụ (RACI Rebalancing)

* **Bối cảnh:** Trong kế hoạch cũ, TV2 gánh toàn bộ đường găng (Board Tensor, ETL, ResNet, L5, MCTS L4/L7, L6, L8), dẫn tới nguy cơ nghẽn tiến độ ở Pha 2 và Pha 3.
* **Quyết định:** Phân bổ lại khối lượng công việc cân bằng tuyệt đối ($33.3\% - 33.3\% - 33.4\%$):
  * **TV1 (Core, Classical Search, UCT & Tabular RL Lead):** Phụ trách `chess_ai/core/` (trừ `board_tensor.py`), `chess_ai/search/alphabeta.py`, `chess_ai/search/move_ordering.py`, `chess_ai/search/quiescence.py`, `chess_ai/search/mcts_node.py`, `chess_ai/search/mcts_uct.py`, các tác tử `level1_random.py`, `level2_heuristic.py`, `level3_alphabeta.py`, `level4_mcts.py`, `level6_reinforcement.py`, và kịch bản `chess_ai/experiments/ablation_search.py`.
  * **TV2 (Deep Learning, ETL, Guided PUCT & Self-Play Lead):** Phụ trách `chess_ai/core/board_tensor.py`, `chess_ai/etl/` (`pgn_etl.py`, `memmap_dataset.py`), `chess_ai/models/` (`dual_resnet.py`, `loss.py`, `train_supervised.py`), `chess_ai/search/mcts_puct.py`, và các tác tử `level5_supervised.py`, `level7_guided_mcts.py`, `level8_alphazero.py`.
  * **TV3 (UI, Arena, Testing, Configs & Analytics Lead):** Phụ trách `configs/`, `chess_ai/ui/desktop_pygame.py`, `chess_ai/arena/` (`tournament.py`, `metrics.py`, `pgn_exporter.py`), `chess_ai/experiments/ablation_mcts.py`, toàn bộ bộ kiểm thử `tests/` (bao gồm `test_representation.py`), notebook phân tích `notebooks/analysis.ipynb`, và kịch bản `scripts/reproduce_experiment.py`.

