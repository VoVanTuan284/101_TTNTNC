# QUY CHUẨN KIỂM THỬ & TIÊU CHÍ BÀN GIAO GIAI ĐOẠN (HANDOFF & QA SPECIFICATION)

**Tên tệp:** `HANDOFF_AND_QA.md`  
**Dự án:** Multi-Level Chess AI Engine ($8 \times 8$ FIDE)  
**Quy mô:** Nhóm 3 thành viên (`TV1`, `TV2`, `TV3`) | 15 Ngày Sprint  
**Nguồn sự thật cấu hình:** `configs/` & `docs/DECISIONS.md`  
**Tiêu chuẩn chất lượng:** $100\%$ tuân thủ FIDE, Zero Illegal Moves, Wilson & Paired Bootstrap $95\%$, Clean Architecture.

---

## PHẦN 1: QUY CHUẨN ĐỒNG BỘ & QUẢN TRỊ MÃ NGUỒN (GITFLOW CONTRACT)

Để $3$ thành viên phát triển song hành không xảy ra xung đột mã nguồn (Merge Conflicts) hay phụ thuộc vòng lặp (Circular Dependency), toàn bộ nhóm thống nhất tuân thủ hợp đồng kỹ thuật sau (theo **ADR-08**):

### 1.1. Chiến lược Phân nhánh Git (Branching Strategy)
* **`main`**: Nhánh phát hành ổn định tuyệt đối (Production-ready). Chỉ gộp vào ngày kết thúc mỗi Sprint Milestone sau khi vượt qua $100\%$ các bài kiểm thử `pytest`.
* **`dev`**: Nhánh tích hợp chung (Integration branch). Mọi PR (Pull Request) gộp vào `dev` bắt buộc phải có xác nhận (Review & Approve) của ít nhất $1$ thành viên khác.
* **Nhánh tính năng độc lập (Feature Branches — Tách biệt hoàn toàn theo tệp):**
  * `feature/tv1-core-search-rl`: TV1 phụ trách `chess_ai/core/game_env.py`, `chess_ai/core/chess_env.py`, `chess_ai/core/evaluation.py`, `chess_ai/core/zobrist.py`, `chess_ai/search/alphabeta.py`, `chess_ai/search/move_ordering.py`, `chess_ai/search/quiescence.py`, `chess_ai/search/mcts_node.py`, `chess_ai/search/mcts_uct.py`, `chess_ai/agents/base_agent.py`, `chess_ai/agents/level1..4`, `chess_ai/agents/level6_reinforcement.py`, và `chess_ai/experiments/ablation_search.py`.
  * `feature/tv2-data-models-puct`: TV2 phụ trách `chess_ai/core/board_tensor.py`, `chess_ai/etl/pgn_etl.py`, `chess_ai/etl/memmap_dataset.py`, `chess_ai/models/dual_resnet.py`, `chess_ai/models/loss.py`, `chess_ai/training/train_supervised.py`, `chess_ai/search/mcts_puct.py`, và `chess_ai/agents/level5_supervised.py`, `chess_ai/agents/level7_guided_mcts.py`, `chess_ai/agents/level8_alphazero.py`.
  * `feature/tv3-ui-arena-qa`: TV3 phụ trách `configs/`, `chess_ai/ui/desktop_pygame.py`, `chess_ai/arena/tournament.py`, `chess_ai/arena/metrics.py`, `chess_ai/arena/pgn_exporter.py`, `chess_ai/experiments/ablation_mcts.py`, toàn bộ bộ kiểm thử `tests/`, `notebooks/analysis.ipynb`, và `scripts/reproduce_experiment.py`.

### 1.2. Hợp đồng Mocking để Triệt Tiêu Điểm Nghẽn (No-Blocking Mocking Protocol)
* **TV3 không cần chờ TV1 & TV2**: Module `chess_ai/arena/tournament.py` và `chess_ai/ui/desktop_pygame.py` được kiểm thử ngay từ Ngày 2–4 bằng `chess_ai/agents/level1_random.py` hoặc `MockAgent`. Bộ kiểm thử `tests/test_representation.py` được TV3 viết song song dựa trên hợp đồng hàm trong `chess_ai/core/board_tensor.py`.
* **TV2 không xung đột với TV1 trên MCTS**: Cây MCTS cổ điển (`chess_ai/search/mcts_uct.py` của TV1) và cây MCTS dẫn hướng nơ-ron (`chess_ai/search/mcts_puct.py` của TV2) nằm ở hai tệp tách biệt, chỉ kế thừa cấu trúc nút cơ sở từ `chess_ai/search/mcts_node.py`.
* **TV1 độc lập hoàn toàn với GPU/Lichess**: Toàn bộ các tác tử Cấp 1–4 và Cấp 6 (KRK RL) của TV1 vận hành trực tiếp trên `ChessEnv` bằng CPU, không phụ thuộc tiến độ huấn luyện mạng ResNet của TV2.

---

## PHẦN 2: QUY ƯỚC CHUẨN VỀ DẤU LƯỢNG GIÁ & TÍN HIỆU THƯỞNG

Để tránh sai lệch cực trị (Max/Min inversion) giữa các tầng thuật toán, toàn bộ hệ thống thống nhất quy ước dấu như sau:

### 2.1. Tín hiệu Kết quả Môi trường (`ChessEnv.result()`)
* **Góc nhìn Tuyệt đối (Absolute - White Perspective)**:
  * $+1.0$: Bên Trắng thắng cờ (`1-0`).
  * $0.0$: Hòa cờ (`1/2-1/2`).
  * $-1.0$: Bên Đen thắng cờ (`0-1`).
  * `None`: Trận đấu chưa kết thúc.

### 2.2. Giá trị Lượng giá Tĩnh & Hàm Đánh giá Lá (`Evaluation & Value Head`)
* **Góc nhìn Tương đối (Relative Perspective - Side-to-Move / Active Player)**:
  * Cả hàm heuristic tĩnh (`chess_ai/core/evaluation.py`) lẫn đầu ra mạng nơ-ron Value Head (`chess_ai/models/dual_resnet.py`) **luôn luôn trả về giá trị từ góc nhìn của bên tới lượt đi hiện tại (`env.current_player()`)**:
    * $+\infty$ (hoặc $+1.0$): Bên tới lượt đi nắm ưu thế thắng tuyệt đối / chiếu hết đối thủ.
    * $0.0$: Thế cờ cân bằng hoặc hòa cờ.
    * $-\infty$ (hoặc $-1.0$): Bên tới lượt đi bị chiếu hết / rơi vào thế thua.
* **Quy tắc Negamax trong Tìm kiếm Cổ điển & MCTS Backpropagation**:
  * Khi gọi đệ quy tại độ sâu kế tiếp hoặc lan truyền ngược từ nút con lên nút cha (nơi lượt đi đổi bên):
    $$\text{score} = -\text{alphabeta}(\text{child}, d - 1, -\beta, -\alpha), \quad v_{\text{parent}} = -v_{\text{child}}$$

---

## PHẦN 3: ĐẶC TẢ ĐỊNH DẠNG CHECKPOINT MÔ HÌNH HỌC SÂU (`.pt`)

Mô hình PyTorch huấn luyện cho Cấp 5, Cấp 7 và Cấp 8 bắt buộc đóng gói theo đúng `checkpoint_schema` trong `configs/model_resnet.yaml` (đồng bộ 100% với `docs/DATASET_SPEC.md`):

```python
checkpoint_payload = {
    "epoch": int,                         # Số epoch đã hoàn thành
    "global_step": int,                   # Tổng số bước cập nhật gradient lũy kế
    "model_state_dict": dict,             # Trọng số mạng Dual-Head ResNet
    "optimizer_state_dict": dict,         # Trạng thái bộ tối ưu hóa AdamW
    "scheduler_state_dict": dict,         # Trạng thái CosineAnnealingLR scheduler
    "scaler_state_dict": dict,            # Trạng thái GradScaler (AMP FP16)
    "train_loss": float,                  # Mất mát huấn luyện tổng hợp
    "val_policy_acc": float,              # Độ chính xác Top-1 Policy trên tập Val
    "val_value_mse": float,               # Sai số MSE Value trên tập Val
    "best_val_loss": float,               # Mất mát Validation thấp nhất đạt được
    "model_config": {
        "input_shape": [18, 8, 8],        # Tensor 18 kênh góc nhìn bên đi
        "canonical_flip": True,           # Lật bàn cờ sq ^ 56 khi Đen đi
        "num_res_blocks": 6,              # 6 Residual Blocks
        "filters": 64,                    # 64 Kênh đặc trưng thân mạng
        "policy_conv_filters": 16,        # 16 Kênh tích chập 1x1 ở Policy Head
        "value_conv_filters": 1,          # 1 Kênh tích chập 1x1 ở Value Head
        "value_hidden_units": 64,         # 64 Nút ẩn FC ở Value Head
        "action_space": 4096,             # Kích thước không gian hành động phẳng
    },
    "random_seed": int,                   # Seed đảm bảo tái lập
    "torch_version": str,                 # Phiên bản PyTorch sử dụng
}
```

* **Yêu cầu nạp trọng số an toàn (Device-Agnostic & PyTorch 2.6+ Compatible)**:
  ```python
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  checkpoint = torch.load(filepath, map_location=device, weights_only=False)
  model.load_state_dict(checkpoint["model_state_dict"])
  ```
* Bắt buộc có cơ chế fallback: Nếu máy chạy kiểm thử không có GPU CUDA, tác tử tự động chuyển sang CPU; nếu chạy unit test khi chưa có tệp `.pt` huấn luyện, cho phép khởi tạo trọng số ngẫu nhiên có seed cố định khi bật cờ `allow_untrained_fallback=True` (chỉ dùng cho unit test kỹ thuật, tuyệt đối không dùng cho giải đấu chính thức theo Mục 3.3 đề bài).

---

## PHẦN 4: BỘ 61+ TEST CASES TỰ ĐỘNG (AUTOMATED TEST SUITE)

Hệ thống kiểm thử tự động sử dụng `pytest` tại thư mục `tests/` (do TV3 chủ trì viết và vận hành xuyên suốt từ Ngày 2 đến Ngày 9) với 61 kịch bản kiểm thử chuẩn:

```text
tests/
├── test_env_fide.py         # [25 Tests] Kiểm tra độ chuẩn xác của luật cờ FIDE & ChessEnv
├── test_representation.py   # [10 Tests] Biểu diễn Tensor (18, 8, 8), lật bàn cờ & Action 4096
├── test_agents_sanity.py    # [16 Tests] Hợp đồng BaseAgent, 8 cấp tác tử & kiểm soát ngân sách
└── test_arena_metrics.py    # [10 Tests] Vận hành Arena 50 khai cuộc, Wilson & Paired Bootstrap CI
```

### Nhóm 1: Kiểm thử Môi trường & Luật cờ FIDE (`test_env_fide.py` - 25 Tests)
1. **TC01**: Khởi tạo bàn cờ mặc định, kiểm tra đủ $32$ quân, đúng vị trí và `current_player() == +1` (Trắng).
2. **TC02**: Sinh danh sách `legal_actions()` tại thế cờ xuất phát chuẩn, khẳng định đúng chính xác $20$ nước đi.
3. **TC03**: Nước đi bắt Tốt qua đường (*En Passant*) của Trắng được ghi nhận hợp lệ đúng thời điểm.
4. **TC04**: Nước đi bắt Tốt qua đường (*En Passant*) của Đen được ghi nhận hợp lệ đúng thời điểm.
5. **TC05**: Nước bắt Tốt qua đường tự động hết hiệu lực nếu không thực hiện ngay tại nước kế tiếp.
6. **TC06**: Nhập thành cánh Vua (Kingside Castling - $O\text{-}O$) của Trắng hợp lệ khi đường thông thoáng.
7. **TC07**: Nhập thành cánh Hậu (Queenside Castling - $O\text{-}O\text{-}O$) của Trắng hợp lệ khi đường thông thoáng.
8. **TC08**: Nhập thành cánh Vua của Đen hợp lệ.
9. **TC09**: Nhập thành cánh Hậu của Đen hợp lệ.
10. **TC10**: Bác bỏ quyền nhập thành nếu Vua đang bị chiếu (*In Check*).
11. **TC11**: Bác bỏ quyền nhập thành nếu ô Vua đi qua bị đối thủ kiểm soát (*Crossed Check*).
12. **TC12**: Bác bỏ quyền nhập thành nếu Vua hoặc Xe tương ứng đã từng di chuyển.
13. **TC13**: Phong cấp chuẩn thành Hậu khi Tốt tiến đến hàng $8$ (Trắng) hoặc hàng $1$ (Đen).
14. **TC14**: Hỗ trợ phong cấp dưới thành Mã (*Underpromotion to Knight*).
15. **TC15**: Hỗ trợ phong cấp dưới thành Xe (*Underpromotion to Rook*).
16. **TC16**: Hỗ trợ phong cấp dưới thành Tượng (*Underpromotion to Bishop*).
17. **TC17**: Nhận diện thế cờ Chiếu hết Fool's Mate sau 2 nước đi, `is_terminal() == True`, `result() == -1.0`.
18. **TC18**: Nhận diện thế cờ Chiếu hết Scholar's Mate sau 4 nước đi, `result() == 1.0`.
19. **TC19**: Nhận diện thế cờ Hòa pat (*Stalemate*), Vua không bị chiếu nhưng không còn nước đi hợp lệ, `result() == 0.0`.
20. **TC20**: Nhận diện hòa do thiếu lực lượng bắt buộc (Vua đơn độc vs Vua + Tượng).
21. **TC21**: Nhận diện hòa do thiếu lực lượng bắt buộc (Vua đơn độc vs Vua + Mã).
22. **TC22**: Nhận diện hòa theo quy tắc 50 nước đi không ăn quân và không đi Tốt (`halfmove_clock >= 100`).
23. **TC23**: Nhận diện hòa do lặp lại thế cờ 3 lần (*Threefold Repetition*) ngay cả trên bản `clone()`.
24. **TC24**: Kiểm tra phương thức `ChessEnv.undo()`, xác nhận trạng thái bàn cờ, FEN và move stack khôi phục hoàn hảo.
25. **TC25**: Kiểm tra phương thức `ChessEnv.clone()` (`stack=True`), bảo đảm sửa đổi hoặc `undo()` trên bản clone độc lập với bản gốc.

### Nhóm 2: Biểu diễn Tensor 18 Kênh & Không Gian Hành Động (`test_representation.py` - 10 Tests)
26. **TC26**: Khởi tạo Tensor trạng thái kích thước chuẩn PyTorch CHW `(18, 8, 8)`, kiểu `float32`, mọi phần tử nằm trong $[0.0, 1.0]$.
27. **TC27**: Khi tới lượt Trắng: Kênh 0–5 chứa quân Trắng ở hàng $0..1$, Kênh 6–11 chứa quân Đen ở hàng $6..7$ (không lật).
28. **TC28**: Khi tới lượt Đen: Phép lật chuẩn hóa `sq ^ 56` đưa quân Đen (Active) về Kênh 0–5 ở hàng $0..1$, và quân Trắng (Opponent) về Kênh 6–11 ở hàng $6..7$.
29. **TC29**: Kênh 12–15 biểu diễn chính xác 4 mặt phẳng nhị phân riêng biệt cho quyền nhập thành (`Active_KS`, `Active_QS`, `Opp_KS`, `Opp_QS`).
30. **TC30**: Kênh 16 đánh dấu đúng giá trị `1.0` tại ô đích En-Passant hợp lệ (đã quy chiếu lật theo lượt đi) và Kênh 17 phản ánh chính xác $\min(\text{halfmove\_clock} / 100.0, 1.0)$.
31. **TC31**: Ánh xạ nước đi chuẩn từ ô $u \to v$ ra chỉ số nguyên phẳng $u_{\text{canon}} \times 64 + v_{\text{canon}} \in [0, 4095]$ cho cả lượt Trắng và lượt Đen.
32. **TC32**: Giải mã ngược từ chỉ số $a \in [0, 4095]$ về đúng `chess.Move` hợp lệ (tự động gắn `promotion=chess.QUEEN` khi Tốt tiến tới hàng cuối).
33. **TC33**: Kiểm tra Legal Action Masking: gán chính xác $-\infty$ (hoặc `0` trên mask) cho mọi chỉ số không hợp lệ, tổng xác suất Softmax của các nước hợp lệ bằng $1.0$.
34. **TC34**: Kiểm tra tính bất biến đối xứng gương màu quân (Color-Mirror Invariance): Một thế cờ của Trắng và thế cờ đối xứng dọc đổi màu của Đen phải sinh ra hai Tensor `(18, 8, 8)` giống hệt nhau $100\%$.
35. **TC35**: Kiểm tra ghi và đọc mảng nhị phân `np.memmap` (`uint8` cho states, khôi phục kênh 17 chia `100.0`) không mất mát độ chính xác.

### Nhóm 3: Kiểm thử Tác tử & An toàn Nước đi (`test_agents_sanity.py` - 16 Tests)
36. **TC36**: Tác tử Cấp 1 (Random) thực hiện $100$ nước đi liên tiếp không bao giờ sinh nước cờ phạm luật.
37. **TC37**: Tác tử Cấp 2 (Heuristic) tìm thấy nước chiếu hết ngay trong 1 nước (Think 1) trên bộ thế cờ Mate-in-1.
38. **TC38**: Tác tử Cấp 2 tự động chặn nguy cơ bị đối thủ chiếu hết ở nước kế tiếp (Think 2).
39. **TC39**: Tác tử Cấp 2 chọn nước bắt quân có lợi nhất theo MVV-LVA thay vì đổi quân lỗ.
40. **TC40**: Tác tử Cấp 3 (Alpha-Beta) với Quiescence Search vượt qua thế cờ bẫy Horizon Effect mà cấu hình tắt Quiescence mắc phải.
41. **TC41**: Bảng chuyển vị Zobrist Transposition Table + Move Ordering giúp Cấp 3 giảm tối thiểu $30\%$ số nút duyệt ở cùng độ sâu $d=3$ trên tập mẫu WAC.
42. **TC42**: Tác tử Cấp 4 (Vanilla MCTS UCT) hoàn thành tìm kiếm với `max_rollout_depth=60` và trả về nước cờ hợp lệ trong ngân sách $N_{\text{sim}}=100$.
43. **TC43**: Tác tử Cấp 5 (Supervised ResNet) nạp checkpoint thành công và chọn đúng nước đi hợp lệ có logit cao nhất.
44. **TC44**: Tác tử Cấp 6 (RL KRK) cập nhật bảng Q / Policy Gradient thành công và chiếu hết được Vua đơn độc từ thế cờ KRK chuẩn.
45. **TC45**: Tác tử Cấp 7 (Guided MCTS PUCT) ở chế độ `eval_mode` tắt nhiễu Dirichlet ($\epsilon=0$) và kết hợp đúng $P(s, a), V(s)$.
46. **TC46**: Tác tử Cấp 8 (Self-play) ở chế độ `selfplay_mode` bật nhiễu Dirichlet, hoàn thành 1 ván tự chơi và lưu bộ ba $(s, \mathbf{\pi}, z)$ vào Replay Buffer.
47. **TC47**: Kiểm soát thời gian hợp tác (Time-Control): Tác tử dừng Iterative Deepening / MCTS đúng hạn ($1.0\text{s} + 0.1\text{s}$ dung sai) và trả về nước đi hợp lệ.
48. **TC48**: Kiểm soát khối lượng (Workload-Control): Tác tử dừng chính xác khi chạm trần `max_nodes` hoặc `max_simulations`.
49. **TC49**: Tác tử xử lý tức thời khi rơi vào thế cờ chỉ có duy nhất 1 nước đi hợp lệ (*Forced Move*).
50. **TC50**: Tác tử xử lý an toàn khi được gọi trên trạng thái bàn cờ đã kết thúc.
51. **TC51**: Phương thức `agent.reset()` xóa sạch trạng thái đệm giữa hai ván đấu độc lập.

### Nhóm 4: Động cơ Thi đấu Arena & Thống kê (`test_arena_metrics.py` - 10 Tests)
52. **TC52**: Vận hành giải đấu mini $10$ ván (5 thế cờ khai cuộc $\times 2$ màu) giữa Cấp 1 và Cấp 2, ghi nhận kết quả tự động.
53. **TC53**: Kiểm tra ghép cặp khai cuộc đối xứng: Hai tác tử tất định đấu 10 ván trên 5 khai cuộc sinh ra đúng 10 biên bản ván đấu theo đúng 5 thế cờ mở màn đã nạp.
54. **TC54**: Tính toán chính xác tỷ lệ điểm số quy đổi $S = (W + 0.5 \times D) / N$.
55. **TC55**: Kiểm tra công thức Khoảng tin cậy Wilson Score 95%, cận dưới và cận trên luôn thuộc $[0, 1]$.
56. **TC56**: Kiểm tra Khoảng tin cậy Paired-Opening Bootstrap 95% ($10.000$ resamples) hoạt động chính xác khi có nhiều ván hòa ($0.5$).
57. **TC57**: Đo đạc chính xác độ trễ phản hồi phân vị 95 (`p95_latency_ms`) và tốc độ duyệt `NPS`/`SPS`.
58. **TC58**: Xuất biên bản ván đấu ra tệp PGN chuẩn quốc tế (`chess_ai/arena/pgn_exporter.py`), đọc lại thành công bằng `chess.pgn.read_game()`.
59. **TC59**: Ghi vết chi tiết từng trận đấu ra tệp `matches.jsonl` đúng schema quy định.
60. **TC60**: Cơ chế xử lý hòa tự động khi trận đấu chạm ngưỡng `max_plies = 400` ($200$ nước đi).
61. **TC61**: Kiểm định toàn bộ 50 dòng trong `resources/benchmark_positions.epd` đều có FEN hợp lệ và nước đi `bm` thuộc `board.legal_moves`.

---

## PHẦN 5: TIÊU CHÍ NGHIỆM THU ĐỘC LẬP GIỮA CÁC THÀNH VIÊN (QUALITY GATES)

Mỗi thành viên khi bàn giao mã nguồn phải vượt qua các Cổng Kiểm Soát Chất Lượng (Quality Gates) thực tế và có kiểm chứng định lượng:

```text
[Thành viên Bàn Giao] ────> [Chạy Bộ Test Pytest 61/61] ────> [Reviewer Xác Nhận] ────> [Gộp vào nhánh dev]
```

### 5.1. Tiêu chí Nghiệm thu của TV1 (Core, Classical Search, UCT & Tabular RL Lead)
1. Lớp `ChessEnv` vượt qua trọn vẹn $25$ bài test FIDE (`TC01` – `TC25`).
2. Tác tử Cấp 2 đạt tỷ lệ điểm số $S \ge 0.90$ trước Cấp 1 trong loạt đấu $100$ ván (50 khai cuộc $\times 2$ màu) nhờ tích hợp phạt lặp thế cờ khi đang ưu thế.
3. Tác tử Cấp 3 (ID + Alpha-Beta + Move Ordering + TT + Quiescence) chứng minh mức giảm $\ge 30\%$ số nút duyệt khi bật Move Ordering + TT (`Config_C2` so với `Config_B`) và tăng rõ rệt tỷ lệ giải đúng trên bộ chuẩn `resources/benchmark_positions.epd` (WAC-50) khi bật Quiescence Search (`Config_D` so với `Config_C2`).
4. Tác tử Cấp 4 (UCT với `max_rollout_depth: 60`) và Cấp 6 (KRK RL có đường cong học tập hội tụ) vượt qua toàn bộ unit test liên quan.

### 5.2. Tiêu chí Nghiệm thu của TV2 (Deep Learning, ETL, Guided PUCT & Self-Play Lead)
1. Module `chess_ai/core/board_tensor.py` vượt qua toàn bộ $10$ bài test (`TC26` – `TC35`), bảo đảm tính bất biến đối xứng màu quân (`TC34`).
2. Đường ống ETL trích xuất thành công từ **Lichess Elite Database** vào `np.memmap` với RAM duy trì $< 1.5\text{GB}$, có báo cáo số lượng mẫu khử trùng Zobrist và đảm bảo $0\%$ rò rỉ thế cờ giữa Train/Val/Test.
3. Mạng Dual-Head ResNet (Cấp 5) huấn luyện hội tụ ổn định: đạt **Top-1 Policy Accuracy $\ge 35\%$** (cao gấp $>10$ lần baseline đoán ngẫu nhiên $\sim 2.8\%$) và **Value MSE thấp hơn có ý nghĩa thống kê so với baseline dự đoán hằng số $0$** trên tập Test độc lập.
4. Tác tử Cấp 7 (Guided MCTS PUCT ở `eval_mode`) đạt tỷ lệ thắng vượt trội($S > 0.60$) khi đối đầu trực tiếp với Cấp 5 (Policy-only Greedy) trong bài Ablation 2B.
5. Vòng lặp Cấp 8 (AlphaZero-inspired) thực thi trọn vẹn chu trình Self-play $\to$ Training $\to$ Arena Gating.

### 5.3. Tiêu chí Nghiệm thu của TV3 (UI, Arena, Testing, Configs & Analytics Lead)
1. Bộ 61 bài kiểm thử tự động (`TC01` – `TC61`) được hoàn thiện đúng tiến độ và chạy xanh $100\%$ (`pytest`).
2. Giao diện `chess_ai/ui/desktop_pygame.py` tách tiến trình/luồng tính toán AI độc lập, không bị đơ giật giao diện, hỗ trợ chọn đủ 8 cấp độ AI ở cả hai chế độ Human-vs-AI và AI-vs-AI.
3. Động cơ `chess_ai/arena/tournament.py` vận hành tự động loạt đấu 100 ván/cặp trên 50 thế cờ khai cuộc cân bằng (`resources/openings_50.epd`), xuất đầy đủ `matches.jsonl`, `games.pgn`, Wilson Score 95% và Paired Bootstrap 95%.
4. Hoàn thiện kịch bản `scripts/reproduce_experiment.py` (hỗ trợ `--from-logs` và `--full`) cùng notebook `notebooks/analysis.ipynb` sinh toàn bộ bảng biểu cho báo cáo.
