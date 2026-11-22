# ĐẶC TẢ YÊU CẦU GIAO DIỆN DESKTOP (UI_SPEC.md)

**Mã tài liệu:** `UI_SPEC_DESKTOP_2026`  
**Dự án:** Multi-Level Chess AI Engine ($8 \times 8$ FIDE)  
**Thành viên phụ trách:** TV3 (UI, Arena, QA & Analytics Lead)  
**Tệp mã nguồn chính:** `chess_ai/ui/desktop_pygame.py` (lệnh chạy: `chess-ui` hoặc `python -m chess_ai.ui.desktop_pygame`)  
**Tài nguyên đồ họa:** `chess_ai/ui/assets/piece/<piece_set>/` (mặc định: `cburnett`)

---

## PHẦN 1: MỤC TIÊU & NGUYÊN TẮC KIẾN TRÚC

Giao diện Desktop Pygame đóng vai trò lớp trình diễn (Presentation Layer) phục vụ chơi thử trực tiếp, quan sát quá trình suy nghĩ (Telemetry) của 8 cấp độ AI và quay Video Demo nộp bài cuối kỳ.

### 1.1. Ranh giới Clean Architecture
* **Không chứa luật cờ trong UI:** UI **tuyệt đối không** truy cập trực tiếp `ChessEnv.board` để sửa bàn cờ hay tự tính nước hợp lệ.
* **Giao tiếp chuẩn qua hợp đồng môi trường & tác tử:**
  * Đọc trạng thái qua `env.render("fen")`, `env.legal_actions()`, `env.san(action)`, `env.is_terminal()`, `env.result()`, `env.termination_reason()`.
  * Thực thi thay đổi qua `env.step(action)` và tạo bản sao an toàn bằng `env.clone()`.
  * Khởi tạo tác tử qua `create_agent()` / `list_full_board_agents()` tại `chess_ai/agents/registry.py`, đọc thống kê từ `agent.last_search_stats`.

### 1.2. Nguyên tắc Vận hành Không Đơ (Non-Blocking 60 FPS)
* Vòng lặp vẽ và bắt sự kiện Pygame chạy trên **Main Thread** ở tốc độ mục tiêu **60 FPS**.
* Mọi lệnh gọi `agent.select_move(snapshot, time_limit, max_workload)` phải chạy trên **Worker Thread / Process** tách biệt (`ThreadPoolExecutor` / `ProcessPoolExecutor`) với bản sao `env.clone()`, tuyệt đối không làm treo cửa sổ hay mất phản hồi nút bấm khi AI đang tính toán.

---

## PHẦN 2: BỐ CỤC TỔNG THỂ & HỆ THỐNG TAB CHUYỂN ĐỔI (`1280 × 760 px`)

Kích thước cửa sổ chuẩn khuyến nghị: **`1280 × 760 px`** (tỉ lệ rộng vừa màn hình laptop, không tràn chiều dọc/ngang). Ở trên cùng có **Thanh Tab điều hướng (`44 px`)** cho phép chuyển đổi tức thì giữa 2 không gian làm việc mà không đóng ứng dụng:

```text
+----------------------------------------------------------------------------------------+
|  [ ♟ Thi đấu & Phân tích (Play & Telemetry) ]   [ 📊 Nghiên cứu & Thực nghiệm (Lab) ]   |
+----------------------------------------------------------------------------------------+
```

### 2.1. Tab 1: Thi đấu & Phân tích (`Play & Telemetry` — Bố cục 3 cột)

```text
+-----------------------+-----------------------------------+--------------------------+
|    CONFIGURATION      |  [Đen (Top)   |  Trạng thái  03:45] |        TELEMETRY         |
|                       +-----------------------------------+                          |
|  AGENT SELECTION      |                                   |  EVALUATION   WIN PROB.  |
|  [Level 1 - Random  ] |                                   |   +-------+    ╭─────╮   |
|  [Level 2 - Heurist.] |                                   |   | +0.85 |    │58.4%│   |
|  [Level 3 - AlphaB. ] |          BÀN CỜ FIDE 8×8          |   +-------+    ╰─────╯   |
|  [Level 4 - MCTS UCT] |           (544 × 544 px)          +--------------------------+
|  [Level 5 - ResNet  ] |         (Tọa độ a-h, 1-8)         |  AI SEARCH METRICS       |
|  [Level 6 - KRK (6) ] |                                   |  NPS / SPS:      18,580  |
|  [Level 7 - PUCT    ] |                                   |  Search Depth:       10  |
|  [Level 8 - AlphaZ. ] +-----------------------------------+  TT Hits:         1,403  |
+-----------------------+  [Trắng (Bot) |  Lượt đi     04:12] |  MCTS Sims:         400  |
|  BUDGET & CONTROLS    +-----------------------------------+  Latency (ms):      181  |
|  Mode: Human/AI vs AI |  FEN: rnbqkbnr/pppppppp/8/...     +--------------------------+
|  Time (s):   [===---] |                                   |  MOVE HISTORY (SAN)      |
|  Nodes:      [====--] |                                   |  1. e4   e5              |
|  Simulations:[===---] |                                   |  2. Nf3  Nc6             |
|  [Flip] [Undo] [Side] |                                   |  ...                     |
|  [Đầu hàng] [Cầu hòa*]|                                   |  [|<]  [<]  [>]  [>|]    |
|  [   NEW GAME / GO  ] |                                   |                          |
+-----------------------+-----------------------------------+--------------------------+
|<------- 270 px ------>|<------------- 670 px ------------>|<-------- 340 px -------->|
```

### 2.2. Tab 2: Nghiên cứu & Thực nghiệm (`Research Lab` — Bố cục 2 cột)

```text
+------------------------------+---------------------------------------------------------+
|     EXPERIMENT RUNNER        |   [ Biểu đồ Trực quan ]   [ Bảng Số liệu ]   [ Nhật ký ]|
|                              +---------------------------------------------------------+
| 1. Chọn Kịch bản Thực nghiệm:|                                                         |
|  (•) Đấu số lượng lớn (Arena)|                                                         |
|  ( ) Ablation 1 (WAC-50)     |            KHUNG HIỂN THỊ BIỂU ĐỒ & SỐ LIỆU             |
|  ( ) Ablation 2 (MCTS N_sim) |                                                         |
|  ( ) Nạp từ Log (--from-logs)|     • Biểu đồ tỷ lệ điểm S & Wilson/Bootstrap 95% CI    |
|                              |     • Biểu đồ Ablation 1 (Nodes vs Accuracy, Độ sâu)    |
| 2. Tham số Thực nghiệm:      |     • Biểu đồ Ablation 2 (Đường cong N_sim Cấp 4 & 7)   |
|  Agents: [L1, L2, L3, L4...] |     • Bảng ma trận W/D/L, NPS/SPS, độ trễ p95           |
|  Số ván/cặp: [ 20 / 50 / 100]|                                                         |
|  Ngân sách:  [Workload/Time] |                                                         |
|  Workers:    [ 1 / 4 ]       |                                                         |
|                              |                                                         |
| 3. Tiến độ & Điều khiển:     |                                                         |
|  [██████████░░░░░] 64%       |                                                         |
|  [ Chạy Thực Nghiệm ] [Dừng] |                                                         |
+------------------------------+---------------------------------------------------------+
|<---------- 340 px ---------->|<----------------------- 940 px ------------------------>|
```

### Chi tiết phân bổ 3 cột (`1280 × 720 px`)
1. **Cột Trái — `Configuration` (`270 px`):**
   * **Tầng trên (`AGENT SELECTION`):** Danh sách 8 thẻ cấp độ tác tử (`Level 1` $\to$ `Level 8`), tô viền sáng cho cấp độ đang chọn. Đánh dấu rõ `Level 6` chỉ dùng cho cờ tàn KRK (tự động khóa/báo chú thích khi ở chế độ 8×8 đầy đủ); khi bật chế độ `AI vs AI`, cho phép chọn riêng tác tử Trắng và Đen.
   * **Tầng giữa (`BUDGET & SEARCH`):** Chuyển đổi chế độ `Time-Control` vs `Workload-Control` kèm thanh trượt/nút tăng giảm trực quan cho `Time Limit (s)`, `Max Nodes` (Alpha-Beta), `Simulations` (MCTS).
   * **Tầng dưới (`CONTROLS`):** Các nút thao tác gọn (`Đảo bàn [F]`, `Undo [U]`, `Đổi phe [P]`, `Chế độ [M]`) và 1 nút hành động chính nổi bật ở đáy (`Ván mới [N]`).
2. **Cột Giữa — `Game Arena` (`670 px`):**
   * **Top Player Bar (Ngay phía trên bàn cờ, `~48 px`):** Hiển thị thông tin (Icon, Tên người chơi / Cấp AI, trạng thái `Đang suy nghĩ...`) và **Đồng hồ đếm ngược** của bên ngồi phía trên (mặc định là **Đen**; tự động đổi sang **Trắng** khi bấm `Đảo bàn [F]`), làm sáng viền khi tới lượt bên đó.
   * **Bàn cờ $8 \times 8$ (`544 × 544 px`, mỗi ô `68 × 68 px`):** Căn giữa hoàn toàn, có ký hiệu cột `a–h` và hàng `1–8` (tự đảo theo `flipped`), tô sáng ô xuất phát & ô đích của nước vừa đi (`last_move`) và chấm gợi ý nước đi hợp lệ.
   * **Bottom Player Bar (Ngay phía dưới bàn cờ, `~48 px`):** Hiển thị thông tin và **Đồng hồ đếm ngược** của bên ngồi phía dưới (mặc định là **Trắng**; tự động đổi theo `flipped`), làm sáng viền khi tới lượt bên đó.
   * **FEN & Result Footer (`~32 px`):** Một dòng gọn ở đáy hiển thị chuỗi `FEN` hiện tại và kết quả/lý do kết thúc ván đấu.
3. **Cột Phải — `Telemetry` (`340 px`):**
   * **Tầng trên (`EVALUATION & WIN PROBABILITY`):** Bên trái là **Evaluation Bar** dọc kèm giá trị số (`+0.85` pawn hoặc `[-1, +1]`); bên phải là **Win Probability Gauge** (vòng tròn hiển thị xác suất thắng quy đổi từ `eval`, ví dụ `58.4%`).
   * **Tầng giữa (`AI SEARCH METRICS`):** Bảng thống kê thời gian thực lấy trực tiếp từ `agent.last_search_stats` (`NPS / SPS`, `Search Depth`, `TT Hits`, `MCTS Simulations`, `Latency ms`).
   * **Tầng dưới (`MOVE HISTORY (SAN)` + Replay):** Danh sách cặp nước đi chuẩn đại số (`1. e4 e5`) cuộn gọn trong khung, tô sáng nước đang xem, kèm 4 nút điều hướng Replay (`|<`, `<`, `>`, `>|`) ở đáy cột.

---

## PHẦN 3: CÁC YẾU TỐ BẮT BUỘC & CHỨC NĂNG VẬN HÀNH

### 3.1. Giao diện con: Thiết lập Ván đấu Mới (`New Game Setup Modal`)
Khi người dùng bấm **`Ván mới [N]`**, hệ thống mở một **hộp thoại con (Overlay Modal ~`520 × 480 px` căn giữa cửa sổ)** với nền tối phủ mờ phía sau (`alpha overlay`), cho phép cấu hình đầy đủ trước khi bắt đầu:

```text
+-------------------------------------------------------+
|              THIẾT LẬP VÁN ĐẤU MỚI                    |
+-------------------------------------------------------+
| 1. Chế độ thi đấu:                                    |
|    [ Người vs Người ] [ Người vs Máy ] [ Máy vs Máy ] |
|                                                       |
| 2. Chọn màu quân & Tác tử cho hai bên:                |
|    Bên TRẮNG: [ Người chơi / Level 1..8 (trừ L6) ]    |
|    Bên ĐEN:   [ Người chơi / Level 1..8 (trừ L6) ]    |
|    (Có nút [⇄ Đổi màu Trắng/Đen] nhanh)               |
|                                                       |
| 3. Thời gian ván đấu theo thể loại (Chess Clock base + increment):|
|    • Tự do:      [ Không giới hạn (∞) ]                 |
|    • Siêu chớp:  [ 1+0 ] [ 1+1 ] [ 2+1 ]                |
|    • Cờ chớp:    [ 3+0 ] [ 3+2 ] [ 5+0 ] [ 5+3 ]        |
|    • Cờ nhanh:   [ 10+0 ] [ 10+5 ] [ 15+10 ]            |
|    • Tiêu chuẩn: [ 30+0 ] [ 30+20 ]                     |
|                                                       |
| 4. Ngân sách AI mỗi nước (khi có AI tham gia):        |
|    (•) Time-Control: [ 0.5s ] [ 1.0s ] [ 2.0s ]       |
|    ( ) Workload-Control: Nodes / Simulations          |
|                                                       |
|              [ Hủy ]      [ Bắt đầu ván đấu ]         |
+-------------------------------------------------------+
```

* **3 Chế độ chơi hỗ trợ đầy đủ:**
  * **Người vs Người (`Human vs Human`):** Cả hai bên Trắng và Đen đều điều khiển bằng chuột trên bàn cờ, đồng hồ đếm ngược luân phiên theo lượt, không gọi Worker AI.
  * **Người vs Máy (`Human vs AI`):** Người dùng chọn mình cầm **Trắng** hoặc **Đen** (hoặc bấm `⇄ Đổi màu`), bên còn lại chọn cấp độ AI từ danh sách `list_full_board_agents()`.
  * **Máy vs Máy (`AI vs AI`):** Chọn độc lập **AI bên Trắng** và **AI bên Đen** để trình diễn đối kháng tự động.
* **Thời gian ván đấu theo thể loại cờ (`base + increment`):**
  * **Tự do (`Unlimited`):** `Không giới hạn (∞)` (không đếm ngược, phù hợp phân tích hoặc xem AI sâu).
  * **Siêu chớp (`Bullet`):** `1+0`, `1+1`, `2+1` (phút + giây cộng thêm mỗi nước).
  * **Cờ chớp (`Blitz`):** `3+0`, `3+2`, `5+0`, `5+3`.
  * **Cờ nhanh (`Rapid`):** `10+0`, `10+5`, `15+10`.
  * **Cờ tiêu chuẩn (`Classical`):** `30+0`, `30+20`.
  * Đồng hồ trên hai thanh Top/Bottom Player Bar đếm ngược theo lượt đi, tự động cộng `increment` (giây) ngay sau mỗi nước hợp lệ, và xử thua hết giờ (flag) nếu đồng hồ về `00:00`.
* **An toàn khi mở Modal:** Nếu đang có ván đấu dở hoặc bấm `Hủy` (`Esc`), giữ nguyên ván hiện tại; khi bấm **`Bắt đầu ván đấu`**, mới khởi tạo lại `ChessEnv`, reset `agent.reset()` và đóng Modal.

### 3.2. Tương tác bàn cờ & Luật FIDE
* **Chọn & Đi quân:** Click chọn ô (hoặc kéo thả Drag & Drop) $\rightarrow$ tô sáng ô đang chọn + chấm tròn ô đích hợp lệ $\rightarrow$ click ô đích để thực thi qua `env.step(action)`.
* **Đánh dấu nước vừa đi (`Last Move Highlight`):** Tô màu vàng-xanh nhạt cho cả ô xuất phát và ô đích của nước đi gần nhất để người xem dễ theo dõi.
* **Luật FIDE đầy đủ:** Nhập thành (`O-O`, `O-O-O`), Bắt Tốt qua đường (En-passant), và Phong cấp (mặc định phong Hậu khi click nhanh; hỗ trợ chọn Q/R/B/N).

### 3.3. Kiểm soát ván đấu, Đầu hàng, Cầu hòa, Undo & Replay
* **Ván mới (`[N]`):** Mở hộp thoại thiết lập ván đấu mới (hoặc reset theo cấu hình vừa chọn). Chặn reset khi Worker AI chưa xong để tránh xung đột.
* **Đầu hàng (`Resign`):** Cho phép kết thúc nhanh ván đấu ở mọi chế độ:
  * Ở `Human vs Human`: Bên đang tới lượt đầu hàng $\rightarrow$ bên kia thắng (`resignation`).
  * Ở `Human vs AI`: Người chơi đầu hàng $\rightarrow$ AI thắng ngay lập tức (bỏ qua kết quả Worker AI nếu đang tính).
  * Ở `AI vs AI`: Dừng nhanh ván đấu đang chạy (`resignation` / `aborted`).
* **Cầu hòa (`Offer Draw` — chỉ có ở chế độ có Người chơi):**
  * **Quy tắc hiển thị:** Nút `Cầu hòa` **chỉ xuất hiện ở `Human vs Human` và `Human vs AI`**, **ẩn hoàn toàn ở chế độ `AI vs AI`**.
  * **Cơ chế xử lý:**
    * *`Human vs Human`:* Hiển thị xác nhận `Đồng ý hòa / Từ chối` cho bên đối phương; nếu đồng ý thì kết thúc hòa (`1/2-1/2`, `draw_agreement`).
    * *`Human vs AI`:* AI kiểm tra đánh giá thế trận hiện tại (`eval` từ `last_search_stats`): chỉ chấp nhận hòa nếu AI không nắm ưu thế rõ (ví dụ `eval <= +20 cp` / `+0.20` pawn); ngược lại thông báo AI từ chối và ván đấu tiếp tục.
* **Đi lại (`Undo [U]`):** Lùi 2 plies ở chế độ `Human vs AI`, và lùi 1 ply ở chế độ `Human vs Human` hoặc `AI vs AI`.
* **Xem lại ván đấu (`Replay`: `|< [Home]`, `< [←]`, `> [→]`, `>| [End]`):**
  * Tua lại mọi trạng thái qua `MoveRecord.snapshot`.
  * **Quy tắc an toàn:** Khi `replay_index < len(records)`, **khóa không cho AI tự động suy nghĩ/đi đè**; nếu người chơi đi nước mới tại mốc cũ, cắt bỏ nhánh lịch sử phía sau mốc đó.

### 3.4. Chức năng Tab 2: Nghiên cứu & Thực nghiệm (`Research Lab`)
Tab 2 kết nối trực tiếp tới các module thực nghiệm của hệ thống (`chess_ai/arena/tournament.py`, `chess_ai/experiments/ablation_search.py`, `chess_ai/experiments/ablation_mcts.py`, `scripts/reproduce_experiment.py`), chạy nền không chặn giao diện:
* **Tái hiện & Chạy mới Thực nghiệm:**
  * **Đấu số lượng lớn (`Arena Batch Runner`):** Cho phép chọn tập hợp các Agent (`Level 1–8` hỗ trợ 8×8), số khai cuộc/số ván (`10, 20, 50, 100 ván/cặp` từ `resources/openings_50.epd`), chế độ ngân sách (`Workload-Control` / `Time-Control`) và số luồng `workers`.
  * **Chạy Ablation 1 (`WAC-50`) & Ablation 2 (`MCTS N_sim`):** Khởi chạy trực tiếp từ UI và cập nhật tiến độ theo từng cấu hình/mức ngân sách.
  * **Nạp nhanh từ Log (`--from-logs`):** Đọc ngay kết quả đã lưu trong `results/` (`ablation1_results.json`, `ablation2_results.json`, `tournament_summary.json`) để hiển thị tức thì dưới 2 giây.
* **Trực quan hóa Biểu đồ & Bảng Thống kê:**
  * **Biểu đồ kết quả:** Sinh biểu đồ qua `matplotlib` (tỷ lệ điểm `S` kèm khoảng tin cậy Wilson/Bootstrap 95%, biểu đồ Ablation 1 giảm nút & Tactical Accuracy, đường cong Ablation 2 theo $N_{\text{sim}}$) và nạp lên Pygame Surface sắc nét (có nút chuyển đổi giữa các biểu đồ).
  * **Bảng số liệu chi tiết:** Hiển thị ma trận `W / D / L`, điểm chuẩn hóa `S`, `Wilson 95% CI`, `Bootstrap 95% CI`, `Avg/p95 Latency (ms)`, `NPS/SPS`.
  * **An toàn tiến trình nền:** Trong khi thực nghiệm chạy ở Worker Process/Thread, hiển thị Progress Bar & nút `Dừng`; người dùng vẫn có thể chuyển sang Tab 1 xem lại ván đấu bình thường mà không bị đơ cửa sổ.

---

## PHẦN 4: YÊU CẦU VỀ THẨM MỸ & TRẢI NGHIỆM (AESTHETICS & UX)

### 4.1. Tài nguyên Quân cờ trong `assets/` & Đồ họa
* **Thư mục lưu trữ chuẩn:** `chess_ai/ui/assets/piece/<piece_set>/`, mỗi bộ gồm đúng **12 tệp vector `.svg`** (hoặc `.png` nền trong suốt):
  * Quân Trắng: `wK`, `wQ`, `wR`, `wB`, `wN`, `wP`
  * Quân Đen: `bK`, `bQ`, `bR`, `bB`, `bN`, `bP`
* **44 bộ quân cờ có sẵn trong `chess_ai/ui/assets/piece/`:**
  * Bộ mặc định: **`cburnett`** (chuẩn quốc tế của Lichess/Wikipedia).
  * Các bộ khác: `alpha`, `anarcandy`, `caliente`, `california`, `cardinal`, `celtic`, `chess7`, `chessboart`, `chessnut`, `companion`, `cooke`, `disguised`, `dubrovny`, `fantasy`, `firi`, `fresca`, `gioco`, `governor`, `horsey`, `icpieces`, `kiwen-suwi`, `kosal`, `leipzig`, `letter`, `maestro`, `merida`, `minimal-warmth`, `monarchy`, `mono`, `mpchess`, `papercut`, `pirouetti`, `pixel`, `reillycraig`, `rhosgfx`, `riohacha`, `shahi-ivory-brown`, `shapes`, `spatial`, `staunty`, `tatiana`, `totoy`, `xkcd`.
* **Cách nạp & hiển thị:** Hàm `_load_piece_images()` nạp trực tiếp từ đường dẫn tương đối `Path(__file__).resolve().parent / "assets" / "piece" / piece_set`, phóng mượt bằng `pygame.transform.smoothscale` kích thước $\approx 88\%$ cạnh ô. Có thể đổi nhanh qua tham số `ChessPygameApp(piece_set="...")` (hoặc bộ chọn theme trong `Configuration`).
* **Fallback đa nền tảng:** Nếu thiếu tệp ảnh, tự động vẽ ký tự Unicode (`♔♕♖♗♘♙` / `♚♛♜♝♞♟`) bằng font hệ thống để ứng dụng không bao giờ crash.

### 4.2. Bảng màu & Độ tương phản (Dark Pro Slate Theme)
* **Nền cửa sổ & Thẻ Panel:** Tông xanh than đậm hiện đại (`#171B24` nền chính, `#1E2430` nền thẻ, viền mảnh `#2E3748`).
* **Ô bàn cờ:** Tông xanh gỗ chuẩn Lichess:
  * Ô sáng: `RGB(238, 238, 210)` (`#EEEED2`)
  * Ô tối: `RGB(118, 150, 86)` (`#769656`)
* **Đánh dấu tương tác:**
  * Ô đang chọn / nước vừa đi (`last_move`): Vàng-xanh nhạt (`#BACB44` / `#F6F669` trên ô sáng-tối).
  * Gợi ý nước hợp lệ: Chấm tròn bán trong suốt hoặc viền vàng hổ phách `#F6D252`.
* **Điểm nhấn Telemetry & Nút chính:** Xanh dương dịu `#56A6FF` cho nút chọn cấp độ / nút `New Game`, xanh ngọc `#4ADE80` và cam `#F59E0B` cho Evaluation Bar & Win Probability Gauge.

### 4.3. Hiệu năng hiển thị & Phông chữ
* **Cache Font:** Khởi tạo `pygame.font.SysFont` **một lần duy nhất** khi vào `run()`; nghiêm cấm khởi tạo lại font bên trong vòng lặp 60 FPS.
* **Hỗ trợ tiếng Việt & Căn lề:** Dùng font `Segoe UI` (hoặc sans-serif dự phòng), số liệu Telemetry dùng căn phải/chữ số rõ ràng để không nhảy cột khi giá trị thay đổi liên tục.

---

## PHẦN 5: TRẠNG THÁI TƯƠNG THÍCH HIỆN TẠI (`v1.0` vs `v2.0`) & TIÊU CHÍ NGHIỆM THU

### 5.1. Đối chiếu Tương thích với Mã nguồn Hiện tại (`chess_ai/ui/desktop_pygame.py`)
Để tránh đứt gãy trong quá trình phát triển song song, toàn bộ lớp `ChessPygameApp` hiện tại (`v1.0`, cửa sổ `1000 × 640 px`) vẫn dùng chung hạt nhân API (`ChessEnv`, `MoveRecord`, `ThreadPoolExecutor`, `_load_piece_images` cho bộ `cburnett`, `list_full_board_agents()`) và được nâng cấp dần lên `v2.0` (`1280 × 760 px`) theo bảng sau:

| Thành phần | Trạng thái Hiện tại (`v1.0` đã chạy) | Nâng cấp Mục tiêu (`v2.0`) |
| :--- | :--- | :--- |
| **Thanh Tab trên cùng** | Chưa có (vào thẳng chế độ chơi) | Thêm Tab `Thi đấu & Phân tích` và Tab `Nghiên cứu & Thực nghiệm` |
| **Bố cục màn hình chơi** | 2 vùng (`640px` Bàn cờ + `360px` Panel phải) | 3 cột (`270px` Config + `670px` Arena + `340px` Telemetry) |
| **Đồng hồ & Thông tin 2 bên** | Hiển thị dạng text trong Panel phải | Đưa lên 2 thanh **Top / Bottom Player Bar** ở 2 đầu bàn cờ |
| **Thiết lập Ván đấu mới** | Chuyển nhanh qua nút `Ván mới [N]`, `Chế độ [M]`, `Đổi phe [P]` | Mở hộp thoại con (`Modal`) chọn 3 chế độ, màu quân, đồng hồ, ngân sách |
| **Chế độ chơi** | `Human vs AI` & `AI vs AI` | Bổ sung thêm `Human vs Human` |
| **Quân cờ & Đồ họa** | Đã nạp `12/12` `.svg` bộ `cburnett` + cache `SysFont` | Giữ nguyên, bổ sung tọa độ `a–h`/`1–8` và `last_move` highlight |
| **Telemetry & Research Lab** | Chưa vẽ Eval Bar / Win Prob / Biểu đồ | Bổ sung Eval Bar, `AI Search Metrics`, và Tab chạy/vẽ biểu đồ Arena/Ablation |

### 5.2. Tiêu chí Nghiệm thu Giao diện
1. **Khởi chạy sạch:** Lệnh `chess-ui` hoặc `python -m chess_ai.ui.desktop_pygame` mở cửa sổ đúng bố cục quy định, nạp đủ `12/12` ảnh quân cờ `cburnett` ở 60 FPS mượt mà.
2. **Hiển thị đầy đủ & không tràn khung:** Toàn bộ nút điều khiển, đồng hồ 2 đầu bàn cờ, Telemetry, FEN footer và lịch sử SAN đều nằm gọn trong khung cửa sổ.
3. **Không lỗi vòng đời & Kiểm thử tự động:** Toàn bộ kiểm thử logic điều khiển (headless) trong `tests/test_ui_logic.py` vượt qua $100\%$.
