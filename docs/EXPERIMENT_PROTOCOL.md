# ĐỀ CƯƠNG PHƯƠNG PHÁP LUẬN & GIAO THỨC THỰC NGHIỆM (EXPERIMENT_PROTOCOL.md)

**Mã tài liệu:** `EXP_PROTOCOL_2026`  
**Dự án:** Multi-Level Chess AI Engine ($8 \times 8$ FIDE)  
**Môn học:** Trí tuệ Nhân tạo Nâng cao  
**Thành viên phụ trách:** TV3 (UI, Arena & Benchmarking Lead) phối hợp TV1 & TV2  
**Nguồn sự thật cấu hình:** `configs/default_config.yaml` & `configs/search_params.yaml`  
**Mục tiêu cốt lõi:** Thiết lập quy chuẩn đo kiểm thực nghiệm khoa học, khách quan, công bằng và bảo đảm tính tái lập $100\%$ theo đúng Rubric đánh giá môn học.

---

## PHẦN 1: NGUYÊN TẮC CỐT LÕI & LIÊM CHÍNH HỌC THUẬT

1. **Khách quan tuyệt đối trong báo cáo:**
   * Nghiêm cấm việc chỉ chọn lọc các ván thắng đẹp mắt để minh họa kết luận tổng thể.
   * Toàn bộ kết luận khoa học phải dựa trên dữ liệu thống kê tổng hợp của giải đấu Round-Robin đối đầu chéo và các bài Ablation.
2. **Không rò rỉ dữ liệu (No Data Leakage & No Reverse Tuning):**
   * Tuyệt đối không dùng các ván đấu hoặc thế cờ trong bộ kiểm thử cuối cùng (`test_*` hoặc `resources/benchmark_positions.epd`) để tinh chỉnh ngược trọng số mạng nơ-ron hay siêu tham số tìm kiếm.
3. **Lưu vết thực nghiệm toàn diện (Comprehensive Logging):**
   * Mọi ván đấu trong khuôn khổ giải đấu đều được ghi vết tự động ra tệp `results/round_robin/matches.jsonl` (ghi nhận thế cờ khai cuộc, từng nước đi, thời gian suy nghĩ, số nút duyệt / mô phỏng, điểm đánh giá) và tệp biên bản chuẩn `results/round_robin/games.pgn`.
4. **Tái lập thực nghiệm hai chế độ (`scripts/reproduce_experiment.py` & `notebooks/analysis.ipynb`):**
   * Chế độ `--from-logs` (Mặc định khi chấm bài): Đọc dữ liệu thô từ `logs/` để tái tạo toàn bộ bảng thống kê, khoảng tin cậy và biểu đồ trong $< 30$ giây.
   * Chế độ `--full`: Chạy lại từ đầu toàn bộ các bài Ablation và giải đấu Round-Robin.
   * Đi kèm notebook tương tác `notebooks/analysis.ipynb` theo đúng Mục 3.7 của đề bài.

---

## PHẦN 2: THỂ THỨC GIẢI ĐẤU ROUND-ROBIN ĐỐI XỨNG TRÊN 50 KHAI CUỘC CÂN BẰNG

### 2.1. Quy Mô Mẫu & Bộ 50 Thế Cờ Khai Cuộc Cân Bằng (Khử Lặp Ván & First-Player Advantage)
* **Vấn đề của xuất phát điểm cố định:** Các tác tử tất định (như Cấp 2, Cấp 3 ở chế độ cố định nút/độ sâu, và Cấp 5 Greedy Policy) luôn chọn cùng một nước đi tại cùng một trạng thái. Nếu luôn bắt đầu từ thế cờ xuất phát mặc định, 100 ván cầm Trắng sẽ lặp lại đúng 1 ván duy nhất.
* **Giải pháp chuẩn hóa (theo ADR-05):**
  * Thiết lập cố định **100 ván/cặp đấu** (`games_per_pair: 100` trong `configs/default_config.yaml`, đáp ứng chuẩn tối thiểu 100 ván/cặp tại Mục 3.5 của đề bài).
  * Sử dụng **Bộ 50 thế cờ khai cuộc cân bằng** (`resources/openings_50.epd`, gồm 50 thế cờ chuẩn FIDE sau $6 \div 8$ plies có điểm đánh giá cân bằng trong $[-0.3, +0.3]$ pawn).
  * **Ghép cặp đối xứng (Paired-Opening Swap):** Với mỗi thế cờ khai cuộc $k \in \{1, \dots, 50\}$:
    * **Ván $2k - 1$:** Tác tử A cầm Trắng, Tác tử B cầm Đen.
    * **Ván $2k$:** Tác tử B cầm Trắng, Tác tử A cầm Đen (trên chính thế cờ khai cuộc $k$ đó).
  * Cơ chế này vừa đảm bảo 100 ván đấu đa dạng thực chất cho cả các tác tử tất định, vừa khử triệt để lợi thế đi trước (First-Player Advantage) và bất đối xứng của từng khai cuộc.

### 2.2. Kiểm Soát Tính Ngẫu Nhiên & Phân Tích Độ Nhạy Theo Seed
* Cố định bộ danh sách **10 hạt giống ngẫu nhiên độc lập** (đồng bộ 100% với `configs/default_config.yaml`):
  $$\text{SEEDS} = [42, 101, 2024, 777, 12, 99, 555, 1337, 888, 314]$$
* Mỗi seed điều phối đúng 5 thế cờ khai cuộc liên tiếp ($5 \times 2 = 10$ ván đấu) trong loạt 100 ván/cặp.
* Trước mỗi ván đấu, hệ thống đồng bộ hóa `random.seed(s)`, `np.random.seed(s)` và `torch.manual_seed(s)`.
* Báo cáo phân tích thêm độ lệch chuẩn của điểm số $S$ giữa 10 khối seed để đánh giá tính ổn định của các tác tử ngẫu nhiên (Cấp 1, Cấp 4). Note: ở chế độ thi đấu chính thức (`eval_mode`), Cấp 7 tắt nhiễu Dirichlet ($\epsilon = 0$) và chọn $\arg\max_a N(s, a)$ để đạt sức cờ ổn định cao nhất.

### 2.3. Quy Định Kết Thúc Ván Đấu & Xử Lý Hòa Cờ
Động cơ điều phối giải đấu (`chess_ai/arena/tournament.py`) tự động áp dụng các quy tắc phân định kết quả FIDE:
* **Chiếu hết (Checkmate):** Bên chiếu hết thắng cuộc ($1-0$ hoặc $0-1$).
* **Hòa pat (Stalemate):** Vua không bị chiếu nhưng không còn nước đi hợp lệ ($1/2 - 1/2$).
* **Hòa thiếu lực lượng (Insufficient Material):** Vua vs Vua; Vua + Tượng vs Vua; Vua + Mã vs Vua.
* **Quy tắc 50 nước đi (Fifty-move Rule):** Hòa nếu 100 plies liên tiếp không có quân bị bắt và không có Tốt di chuyển.
* **Luật lặp lại thế cờ 3 lần (Threefold Repetition):** Xử hòa nếu cùng một trạng thái Zobrist xuất hiện lần thứ 3.
* **Giới hạn số nước tối đa (`max_plies: 400`):** Nếu ván cờ vượt quá $200$ nước đi ($400$ plies) mà chưa ngã ngũ, hệ thống tự động xử hòa ($0.0$).

---

## PHẦN 3: HAI CHẾ ĐỘ KIỂM SOÁT NGÂN SÁCH MINH BẠCH (FAIR BUDGETING)

Tuân thủ Mục 3.5 của đề bài (*"báo cáo theo ít nhất hai chế độ: ngân sách tính toán được quy định rõ và cấu hình mặc định minh bạch. Không suy diễn công bằng tính toán khi chỉ áp cùng độ sâu tìm kiếm"*), giải đấu được vận hành ở hai chế độ:

### 3.1. Chế Độ 1: Cấu Hình Mặc Định Minh Bạch Theo Khối Lượng Tính Toán (Workload-Control Mode)
* **Mục tiêu:** Đánh giá chất lượng thuật toán dưới ngân sách bước tính toán cố định, không phụ thuộc dao động tải CPU/OS và cho phép chạy song song đa tiến trình (`multiprocessing`) an toàn.
* **Ngân sách mặc định (`configs/search_params.yaml`):**
  * **Cấp 3 (Alpha-Beta):** Giới hạn tối đa $N_{\text{nodes}} = 10.000$ nút duyệt / nước đi (kèm Iterative Deepening trần $d_{\max} = 4$).
  * **Cấp 4 (Vanilla MCTS UCT) & Cấp 7 (Guided MCTS PUCT):** Cố định $N_{\text{sim}} = 400$ simulations / nước đi.
* **Lưu ý học thuật:** Báo cáo nêu rõ 1 nút duyệt Alpha-Beta và 1 simulation MCTS có chi phí đơn vị khác nhau, do đó kết quả ở chế độ này luôn được trình bày song hành với bảng thời gian suy nghĩ trung bình và độ trễ $p95$.

### 3.2. Chế Độ 2: Khống Chế Thời Gian Thực (Time-Control Mode)
* **Mục tiêu:** Đo lường sức cờ thực tế khi mọi tác tử được cấp **cùng một quỹ thời gian suy nghĩ trên mỗi nước đi** ($1.0\text{s}$/nước chuẩn và $0.5\text{s}$/nước cờ nhanh).
* **Quy tắc thực thi chống nhiễu tài nguyên:**
  * Ở chế độ Time-Control, các trận đấu bắt buộc chạy **tuần tự đơn luồng (Single-instance Sequential)** để không xảy ra tranh chấp CPU/GPU giữa các tiến trình làm méo thời gian thực.
* **Cơ chế Kiểm soát Thời gian Hợp tác (Cooperative Time Management):**
  * Thay vì dùng lệnh ngắt tiến trình thô bạo (vốn làm mất toàn bộ kết quả đang tính dở), các vòng lặp tìm kiếm (Iterative Deepening ở Cấp 3 và vòng lặp simulation ở Cấp 4/Cấp 7) kiểm tra đồng hồ `time.perf_counter()` sau mỗi đơn vị công việc nhỏ (mỗi $256$ nút duyệt hoặc mỗi simulation).
  * Khi thời gian trôi qua chạm ngưỡng `time_limit - margin`, thuật toán dừng vòng lặp an toàn và trả về ngay nước đi tốt nhất từ độ sâu hoàn chỉnh gần nhất (Cấp 3) hoặc nút con có số lần thăm $N(s, a)$ lớn nhất hiện tại (Cấp 4/7).
  * Một bộ canh giờ ngoài (`grace_period_ms: 100`) chỉ đóng vai trò chốt chặn an toàn cuối cùng nếu tác tử bị treo bất thường.

---

## PHẦN 4: ĐỘ ĐO KHOA HỌC & KIỂM ĐỊNH THỐNG KÊ KÉP

### 4.1. Tỷ Lệ Điểm Số Quy Đổi (Normalized Score)
Kết quả thi đấu của tác tử trong loạt trận gồm $N = 100$ ván được tính theo công thức chuẩn FIDE:
$$S = \frac{W + 0.5 \times D}{N} \in [0.0, 1.0]$$
Trong đó $W, D, L$ lần lượt là số ván thắng, hòa, thua ($W + D + L = N$). Báo cáo đồng thời trình bày tỷ lệ chi tiết $W / D / L$ và phân tách kết quả khi cầm Trắng ($S_{\text{White}}$) so với khi cầm Đen ($S_{\text{Black}}$).

### 4.2. Khoảng Tin Cậy Thống Kê Kép: Wilson Score 95% & Paired Bootstrap 95%
Theo khuyến nghị tại Mục 3.5 của đề bài (*"khoảng tin cậy (khuyến nghị Wilson/Bootstrap)"*), hệ thống tính toán đồng thời hai độ đo khoảng tin cậy:
1. **Khoảng tin cậy Wilson Score 95% ($z = 1.96$):**
   $$CI_{\text{Wilson}} = \frac{1}{1 + \frac{z^2}{N}} \left( S + \frac{z^2}{2N} \pm z \sqrt{\frac{S(1 - S)}{N} + \frac{z^2}{4N^2}} \right)$$
   Đảm bảo khoảng ước lượng luôn nằm trong $[0, 1]$ ngay cả khi $S \to 0$ hoặc $S \to 1$.
2. **Khoảng tin cậy Bootstrap 95% theo Cặp Khai Cuộc (Paired-Opening Bootstrap):**
   * Vì điểm số mỗi ván nhận 3 giá trị $\{0, 0.5, 1\}$ (không thuần nhị phân Bernoulli) và hai ván trên cùng một thế cờ khai cuộc có tương quan nội tại, module `chess_ai/arena/metrics.py` nhóm kết quả thành $M = 50$ cặp điểm khai cuộc $p_k = \frac{1}{2}(s_{2k-1} + s_{2k}) \in \{0, 0.25, 0.5, 0.75, 1.0\}$.
   * Thực hiện lấy mẫu lại có hoàn lại $B = 10.000$ lần trên $50$ cặp khai cuộc này để trích xuất phân vị $[2.5\%, 97.5\%]$ chính xác.

### 4.3. Chỉ Số Hiệu Năng Hệ Thống & Chỉ Số Định Lượng Phong Cách (Cho RQ4)
* **Hiệu năng & Chi phí tính toán:**
  * Thời gian suy nghĩ trung bình (`avg_latency_ms`) và độ trễ phân vị thứ 95 (`p95_latency_ms`).
  * Tốc độ duyệt nút (`NPS` - Nodes Per Second) cho Cấp 3 và tốc độ mô phỏng (`SPS` - Simulations Per Second) cho Cấp 4, Cấp 7.
  * Đỉnh bộ nhớ RAM (RSS MB), đỉnh VRAM (MB), và tổng thời gian huấn luyện (cho Cấp 5, Cấp 6, Cấp 8).
* **Chỉ số định lượng phong cách thi đấu (Phục vụ trả lời RQ4):**
  * **Move Agreement Rate (%):** Tỷ lệ phần trăm nước đi trùng khớp giữa Cấp 5/Cấp 7 với Cấp 3 và với nước đi chuyên gia trên tập Test.
  * **Average Game Length & Castling Rate:** Độ dài ván đấu trung bình (plies), tần suất nhập thành an toàn Vua trong 15 nước đầu, và tỷ lệ ván thắng bằng chiếu hết so với hòa cờ.

---

## PHẦN 5: THIẾT KẾ CHI TIẾT 02 BÀI NGHIÊN CỨU BÓC TÁCH & ĐỘ NHẠY (ABLATION STUDIES)

### 5.1. Ablation 1: Tầng Tìm Kiếm Cổ Điển, Độ Nhạy Độ Sâu & Giảm Thiểu Horizon Effect
* **Tập dữ liệu kiểm định chuẩn:** Bộ 50 thế cờ chiến thuật chuẩn Win At Chess (`resources/benchmark_positions.epd`, từ `WAC.001` đến `WAC.050`, đã xác thực $100\%$ bằng `python-chess`).
* **Phần 1A — So sánh 5 cấu hình lũy tiến ở cùng độ sâu cơ sở $d = 3$:**
  1. `Config_A (Minimax thuần)`: Duyệt cây vét cạn Negamax ở độ sâu cố định $d = 3$.
  2. `Config_B (Alpha-Beta)`: Tích hợp cắt tỉa $\alpha$-$\beta$ với thứ tự nước đi mặc định.
  3. `Config_C1 (Alpha-Beta + Move Ordering)`: Bổ sung sắp xếp nước đi MVV-LVA, Killer Moves và History Heuristic.
  4. `Config_C2 (Alpha-Beta + Move Ordering + Zobrist TT)`: Tích hợp thêm bảng chuyển vị Zobrist Transposition Table kèm Iterative Deepening ($d = 1 \to 3$), cô lập riêng mức giảm số nút duyệt của TT (kiểm chứng `TC41`).
  5. `Config_D (Cấp 3 Hoàn chỉnh: Config_C2 + Quiescence Search)`: Mở rộng tìm kiếm tĩnh chuỗi bắt quân và thoát chiếu (`max_plies: 8`, kèm Delta Pruning) tại các nút lá.
  > **Kỳ vọng lý thuyết & Kiểm chứng:** `Config_A` $\to$ `Config_C2` bảo toàn giá trị minimax ở độ sâu $d=3$ nên có tỷ lệ giải đúng tương đương nhau nhưng cắt giảm mạnh mẽ số nút duyệt và độ trễ; trong khi `Config_D` tạo ra bước nhảy vọt về **Tactical Accuracy (%)** nhờ hóa giải các bẫy trì hoãn vượt quá chân trời $d=3$.
* **Phần 1B — Khảo sát độ nhạy theo độ sâu tìm kiếm ($d \in \{2, 3, 4\}$):**
  * Đo lường sự thay đổi của Tactical Accuracy (%), số nút duyệt trung bình và độ trễ $p95$ của Cấp 3 khi tăng độ sâu $d = 2 \to 3 \to 4$.

### 5.2. Ablation 2: Tầng MCTS (Cấp 4 & Cấp 7) và Vai Trò Dẫn Hướng Của Mạng Nơ-ron
Gồm 2 phần khảo sát liên hoàn:

#### Phần 2A — Khảo sát độ nhạy số lượt mô phỏng cho cả Cấp 4 (UCT) và Cấp 7 (PUCT):
* **Mục tiêu (đáp ứng yêu cầu Mục 3.2 & 3.5 của đề bài):** Khảo sát tác động của ngân sách mô phỏng:
  $$N_{\text{sim}} \in \{50, 100, 200, 400, 800, 1000\}$$
* **Thiết kế thực nghiệm:**
  * Đo tốc độ giải chiến thuật / tỷ lệ điểm số của **Cấp 4 (Vanilla MCTS UCT)** và **Cấp 7 (Guided MCTS PUCT)** khi đối đầu với mốc chuẩn Cấp 2 / Cấp 3 qua từng mức $N_{\text{sim}}$ (50 ván/mức trên 25 thế cờ khai cuộc cân bằng $\times 2$ màu).
  * Vẽ chung biểu đồ đường cong hiệu năng (Performance Curve) của Cấp 4 và Cấp 7 theo $N_{\text{sim}}$ để kiểm chứng quy luật suy giảm lợi nhuận biên (Diminishing Returns) và chứng minh sự vượt trội của dẫn hướng nơ-ron.

#### Phần 2B — Bóc tách đóng góp của Policy Prior $P(s, a)$ và Value Head $V(s)$:
* Tổ chức đấu đối kháng 100 ván/cặp (50 khai cuộc $\times 2$ màu) giữa 3 biến thể:
  1. **Policy-Only Greedy (Cấp 5):** Chỉ chọn $\arg\max_a P(s, a)$ ($0$ lookahead).
  2. **Value-Only MCTS (Uniform Prior + Value Head):** MCTS dùng $P(s, a)$ đồng đều nhưng định giá lá bằng $V(s)$ ($N_{\text{sim}} = 400$).
  3. **Full Guided MCTS PUCT (Cấp 7):** Kết hợp đầy đủ cả Policy Prior $P(s, a)$ lẫn Value Head $V(s)$ ($N_{\text{sim}} = 400$).
* **Mục tiêu khoa học:** Trả lời trọn vẹn câu hỏi nghiên cứu **RQ1** bằng cách định lượng riêng biệt phần đóng góp của Policy Head, Value Head và cây tìm kiếm PUCT.

---

## PHẦN 6: ĐẶC TẢ TỆP LOG VẾT & KỊCH BẢN TÁI LẬP (`REPRODUCE_EXPERIMENT.PY`)

### 6.1. Cấu Trúc Log JSONL Chuẩn (`results/round_robin/matches.jsonl`)
```json
{
  "match_id": "L3_vs_L7_open012_w",
  "opening_id": "OPEN.012",
  "opening_fen": "r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3",
  "white": "Level7_GuidedMCTS",
  "black": "Level3_AlphaBeta",
  "result": "1-0",
  "termination": "checkmate",
  "total_plies": 68,
  "random_seed": 2024,
  "budget_mode": "workload_control",
  "budget_target": {"white_sims": 400, "black_nodes": 10000},
  "white_avg_latency_ms": 285.4,
  "black_avg_latency_ms": 142.1,
  "white_p95_latency_ms": 410.2,
  "black_p95_latency_ms": 260.8,
  "white_avg_sps": 1401.5,
  "black_avg_nps": 70372.9,
  "moves_uci": ["f1b5", "a7a6", "b5a4", "g8f6"]
}
```

### 6.2. Lệnh Thực Thi Kịch Bản Tái Lập Tự Động
```bash
# Chế độ 1 (Nhanh - Mặc định khi nghiệm thu): Tái tạo toàn bộ bảng số liệu, Wilson/Bootstrap CI & biểu đồ từ logs/
python scripts/reproduce_experiment.py \
    --from-logs \
    --config configs/default_config.yaml \
    --output-dir results/figures/

# Chế độ 2 (Chạy lại toàn phần): Thực thi lại Ablation 1, Ablation 2 và Giải đấu Round-Robin 100 ván/cặp
python scripts/reproduce_experiment.py \
    --full \
    --config configs/default_config.yaml \
    --output-dir results/figures/
```
