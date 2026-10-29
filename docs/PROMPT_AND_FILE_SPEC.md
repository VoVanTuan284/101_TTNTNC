# TÀI LIỆU ĐẶC TẢ KỸ THUẬT & KIẾN TRÚC HỆ THỐNG (PROMPT_AND_FILE_SPEC.md)

**Mã dự án:** `CHESS_AI_ADVANCED_2026`  
**Đề tài:** Xây Dựng Và Đánh Giá Hệ Thống Game AI Cờ Vua Đa Cấp Độ ($8 \times 8$ FIDE) _(Đã được giảng viên phê duyệt)_  
**Môn học:** Trí tuệ Nhân tạo Nâng cao  
**Kiến trúc chuẩn:** Clean / Hexagonal Architecture  
**Nguồn sự thật cấu hình:** `configs/` (`default_config.yaml`, `model_resnet.yaml`, `search_params.yaml`) & `docs/DECISIONS.md`  
**Phân tầng tác tử:** Đầy đủ 8 Cấp độ Tác tử (Cấp 1 $\rightarrow$ Cấp 8) | Sprint 15 ngày

---

## PHẦN 1: TỔNG QUAN HỆ THỐNG & CẤU TRÚC THƯ MỤC CHUẨN

Hệ thống được thiết kế theo nguyên lý kiến trúc sạch (Clean Architecture), bảo đảm sự phân tách độc lập và ranh giới rõ ràng giữa các phân hệ: Môi trường lõi (`chess_ai/core/`), Thuật toán tìm kiếm (`chess_ai/search/`), Toàn bộ 8 cấp độ tác tử (`chess_ai/agents/`), Mạng học sâu & Huấn luyện (`chess_ai/models/`), Đường ống dữ liệu (`chess_ai/etl/`), Động cơ kiểm định đối kháng (`chess_ai/arena/`), Kịch bản thực nghiệm bóc tách (`chess_ai/experiments/`), Giao diện tương tác (`chess_ai/ui/`), Bộ kiểm thử tự động (`tests/`), và Notebook phân tích (`notebooks/`).

```text
101_TTNTNC/
├── pyproject.toml                           # Cấu hình package, pytest, Python >= 3.11 & CLI entry points
├── .python-version                          # Ghim phiên bản Python 3.11
├── README.md                                # Hướng dẫn cài đặt, huấn luyện, đánh giá & demo
├── requirements.txt                         # Danh mục thư viện cố định phiên bản
├── .gitignore                               # Giữ lại checkpoints/best_model.pt & results/ để nộp bài
├── configs/                                 # Nguồn sự thật duy nhất (Single Source of Truth)
│   ├── default_config.yaml                  # Đường dẫn, 10 seeds chuẩn, cấu hình ETL, Arena & Logging
│   ├── model_resnet.yaml                    # Tensor [18, 8, 8], kiến trúc Dual-Head ResNet & Checkpoint schema
│   └── search_params.yaml                   # Siêu tham số PST, Alpha-Beta (ID), UCT, PUCT, RL & Self-play
├── chess_ai/                                # Gói mã nguồn trung tâm (Clean Architecture)
│   ├── __init__.py
│   ├── config.py                            # Bộ nạp cấu hình YAML tập trung
│   ├── utils/                               # Tiện ích dùng chung toàn hệ thống (seeding, budget, epd)
│   ├── core/                                # Lõi môi trường (game_env, chess_env, board_tensor, evaluation, zobrist)
│   ├── search/                              # Tìm kiếm (alphabeta, move_ordering, quiescence, transposition, mcts_*)
│   ├── agents/                              # Toàn bộ 8 cấp độ tác tử AI + base_agent + registry
│   ├── models/                              # Kiến trúc mạng Dual-Head ResNet (dual_resnet, loss)
│   ├── training/                            # Kịch bản huấn luyện (train_supervised, train_rl_krk, selfplay, replay_buffer)
│   ├── etl/                                 # Đường ống dữ liệu (pgn_etl, memmap_dataset)
│   ├── arena/                               # Giải đấu & thống kê (match, tournament, metrics, pgn_exporter)
│   ├── experiments/                         # Thực nghiệm bóc tách (ablation_search, ablation_mcts)
│   └── ui/                                  # Giao diện Pygame Desktop (desktop_pygame, assets/)
├── resources/                               # Thế cờ chuẩn đầu vào (benchmark_positions.epd, openings_50.epd)
├── data/                                    # Dữ liệu huấn luyện dung lượng lớn (raw/, processed/ - gitignored)
├── checkpoints/                             # Trọng số mô hình nộp kèm (commit best_model.pt & l6_krk_qtable.pkl)
├── results/                                 # Kết quả thực nghiệm & biểu đồ nộp bài (round_robin, ablation1/2, training, figures)
├── logs/                                    # Nhật ký debug lúc chạy (chess.log - gitignored)
├── scripts/                                 # Kịch bản tái lập tự động (reproduce_experiment.py)
├── notebooks/                               # Notebook phân tích & tạo bảng/hình (analysis.ipynb)
├── tests/                                   # Bộ 61+ kịch bản kiểm thử tự động (conftest, fixtures/, 4 tệp test_*)
├── docs/                                    # Tài liệu đặc tả, đề bài, DECISIONS.md & CONTRIBUTIONS.md
└── report/                                  # Báo cáo 5 chương, Slide, Poster & Link Video Demo nộp bài
```

---

## PHẦN 2: RANH GIỚI TRÁCH NHIỆM & LIÊM CHÍNH HỌC THUẬT

### 2.1. Phân định Ranh giới Kế thừa và Tự phát triển (Academic Integrity Boundary)

Theo quy định tại Mục 2 và Mục 5 của đề bài môn học, nhóm phân định minh bạch ranh giới giữa thư viện nền tảng và phần tự cài đặt:

- **Kế thừa hợp lệ (`python-chess`)**:
  - Biểu diễn bàn cờ nội bộ dưới dạng Bitboard $64$-bit nguyên bản.
  - Sinh danh sách toàn bộ các nước đi hợp lệ (`legal_moves`).
  - Cơ chế đẩy nước đi vào ngăn xếp (`push()`) và hoàn tác (`pop()`).
  - Kiểm tra các điều kiện chiếu, chiếu hết, và hòa cờ theo luật FIDE: hòa pat (stalemate), hòa do thiếu lực lượng (insufficient material), hòa theo quy tắc 50 nước đi (fifty-move rule), và hòa do lặp lại thế cờ 3 lần (threefold repetition).
  - Đọc/ghi cấu trúc chuỗi thế cờ FEN/EPD và biên bản ván cờ PGN.
- **Tự thiết kế và cài đặt $100\%$ ($100\%$ Self-implemented)**:
  - Lớp bọc môi trường chuẩn hoá `GameEnv` và bộ chuyển đổi `ChessEnv` tuân thủ đầy đủ API đề bài (`reset`, `legal_actions`, `step`, `is_terminal`, `result`, `render`, `clone`, `undo`, `current_player`).
  - Toàn bộ thuật toán lượng giá tĩnh: Cân bằng lực lượng (Material Balance), Bảng vị trí quân (Piece-Square Tables Michniewski), An toàn Vua (King Safety), và Tính cơ động (Mobility).
  - Bảng băm Zobrist ($64$-bit keys) và bộ nhớ chuyển vị Transposition Table độc lập.
  - Tầng tìm kiếm cổ điển: Iterative Deepening Minimax, Cắt tỉa Alpha-Beta, Chiến lược sắp xếp nước đi Move Ordering (PV-move, MVV-LVA, Killer Moves, History), và Quiescence Search kèm Delta Pruning.
  - Tầng tìm kiếm ngẫu nhiên: Vanilla MCTS (UCT) với cơ chế Rollout cắt ngưỡng có định giá tĩnh.
  - Tầng học sâu: Biểu diễn trạng thái Tensor $\mathbb{R}^{18 \times 8 \times 8}$ chuẩn hóa góc nhìn bên đi (Canonical Side-to-Move), Kiến trúc mạng Dual-Head ResNet, hàm mất mát đa nhiệm và trình quản lý checkpoint.
  - Tầng học tăng cường & lai: Tác tử RL Cấp 6 trên cờ tàn KRK, Tác tử Guided MCTS (PUCT tích hợp Policy/Value) Cấp 7, và Vòng lặp Self-play Cấp 8.
  - Đường ống dữ liệu ETL: Lọc Lichess Elite Database, khử trùng lặp thế cờ bằng Zobrist Hash, phân tách Game-level split tất định, và lưu trữ mảng nhị phân `np.memmap`.
  - Khung đấu Arena trên 50 thế cờ khai cuộc cân bằng, đo kiểm thống kê Wilson Score $95\%$ và Paired Bootstrap $95\%$, độ trễ $p95$, NPS/SPS.
  - Giao diện đồ họa Pygame đa tiến trình/luồng hỗ trợ Người đấu Máy và Máy đấu Máy.

---

## PHẦN 3: ĐẶC TẢ CHI TIẾT 8 CẤP ĐỘ TÁC TỬ (LEVEL 1 ĐẾN LEVEL 8)

> **Lưu ý từ đề bài:** Các cấp độ từ 1 đến 8 thể hiện mức độ phức tạp của phương pháp tiếp cận thuật toán, không mặc nhiên tương ứng tuyến tính với sức mạnh thi đấu thực tế. Điểm khuyến khích (tối đa $+10$ điểm trên toàn bài) được xét chung cho chất lượng thực nghiệm thuyết phục của các cấp nâng cao (Cấp 7/8), tối ưu hóa tìm kiếm và khả năng giải thích chiến thuật.

### Cấp 1: Uniform Random [BẮT BUỘC]

- **Mục tiêu & Bản chất:** Lấy mẫu ngẫu nhiên đồng đều (Uniform Sampling) từ danh sách các nước đi hợp lệ (`legal_actions()`).
- **Vai trò khoa học:** Đóng vai trò là mốc đối chứng sàn (Lower Bound Baseline) để kiểm chứng tính đúng đắn của các tầng thuật toán cao hơn.

### Cấp 2: Heuristic / Rule-based [BẮT BUỘC]

- **Mục tiêu & Bản chất:** Tác tử 1-ply kết hợp hệ thống luật ưu tiên phân cấp 3 tầng (Think 1–3) nhằm giải quyết rõ thứ tự ưu tiên và triệt tiêu xung đột luật:
  - _Tầng 1 (Think 1 - Quyết định thắng ngay):_ Duyệt toàn bộ `legal_actions()`, nếu có nước cờ dẫn tới chiếu hết đối thủ (`is_checkmate()`) thì thực hiện ngay lập tức.
  - _Tầng 2 (Think 2 - Phòng thủ sống còn & Bắt quân có lợi):_
    - Kiểm tra nếu đối thủ có nước đe dọa chiếu hết hoặc bắt quân lớn không có bảo vệ, ưu tiên phòng thủ/chạy quân.
    - Thực hiện các nước bắt quân có lợi thực sự (Static Exchange / MVV-LVA dương).
  - _Tầng 3 (Think 3 - Định vị vị trí & Thế trận):_ Tính điểm lượng giá trạng thái tĩnh sau 1 nước đi theo hàm mục tiêu tổng hợp:
    $$\text{Score} = w_{\text{mat}} \cdot \Delta \text{Material} + w_{\text{pst}} \cdot \Delta \text{PST} + w_{\text{safe}} \cdot \Delta \text{KingSafety} + w_{\text{mob}} \cdot \Delta \text{Mobility} - \text{RepetitionPenalty}$$
- **Hệ số trọng số (Centipawns - khai báo tại `configs/search_params.yaml`):** Tốt = $100$, Mã = $320$, Tượng = $330$, Xe = $500$, Hậu = $900$, Vua = $20000$. Bổ sung phạt nhẹ khi lặp lại thế cờ khi đang có ưu thế chất để tránh hòa vô nghĩa trước Cấp 1.

### Cấp 3: Iterative Deepening Minimax & Alpha-Beta Search [BẮT BUỘC]

- **Mục tiêu & Bản chất:** Thuật toán duyệt cây đối kháng theo khuôn mẫu Negamax kết hợp cắt tỉa Alpha-Beta ($\alpha$-$\beta$ pruning) và sâu dần lặp (**Iterative Deepening**):
  - _Iterative Deepening (ID) & Quản lý thời gian hợp tác:_ Duyệt tuần tự độ sâu $d = 1, 2, \dots, d_{\max}$. Kiểm tra ngân sách thời gian/nút định kỳ mỗi $256$ nút; nếu hết giờ, trả về ngay nước đi tốt nhất từ độ sâu hoàn tất gần nhất.
  - _Transposition Table (TT):_ Bảng băm Zobrist 64-bit lưu trữ trạng thái đã duyệt, cờ đánh dấu (`EXACT`, `LOWERBOUND`, `UPPERBOUND`), nước đi tốt nhất (`best_move`) và độ sâu tìm kiếm.
  - _Move Ordering:_ Sắp xếp thứ tự duyệt nước đi tại mỗi nút: ưu tiên PV-move từ TT $\rightarrow$ các nước bắt quân theo MVV-LVA $\rightarrow$ Killer Moves (2 slots/ply) $\rightarrow$ History Heuristic. Giúp hệ số phân nhánh hiệu dụng tiến gần trường hợp tốt nhất lý thuyết $\mathcal{O}(b^{d/2})$.
  - _Quiescence Search (Tìm kiếm tĩnh):_ Tại các nút lá ($d=0$), tiếp tục mở rộng giới hạn (`max_plies: 8`) cho các chuỗi bắt quân và thoát chiếu (kèm Delta Pruning) nhằm giảm thiểu đáng kể hiện tượng mù chân trời (Horizon Effect).

### Cấp 4: Vanilla MCTS (UCT) [BẮT BUỘC]

- **Mục tiêu & Bản chất:** Thuật toán tìm kiếm cây Monte Carlo nguyên bản (`chess_ai/search/mcts_uct.py`) vận hành qua 4 giai đoạn:
  1. _Selection:_ Duyệt từ nút gốc xuống nút lá bằng công thức UCB1 (UCT):
     $$\text{UCT}(s, a) = Q(s, a) + c_{\text{uct}} \cdot \sqrt{\frac{\ln N(s)}{N(s, a)}} \quad \text{với } c_{\text{uct}} = 1.414$$
  2. _Expansion:_ Mở rộng một nút con hợp lệ chưa được khám phá.
  3. _Simulation (Cut-off Rollout):_ Mô phỏng nhanh (ưu tiên nhẹ nước bắt quân/ngẫu nhiên) tối đa tới độ sâu cắt `max_rollout_depth = 60` plies. Nếu ván cờ chưa kết thúc tại ngưỡng cắt, ước lượng giá trị lá bằng hàm lượng giá tĩnh chuẩn hóa $\tanh(\text{eval}_{\text{cp}} / 600.0) \in [-1, 1]$.
  4. _Backpropagation:_ Cập nhật số lần thăm $N$ và giá trị trung bình $Q$ (đảo dấu theo lượt đi Negamax) ngược lên gốc.
- **Khảo sát bắt buộc theo đề bài:** Thực hiện khảo sát độ nhạy theo ngân sách mô phỏng $N_{\text{sim}} \in \{50, 100, 200, 400, 800, 1000\}$.

### Cấp 5: Supervised Neural Network (Dual-Head ResNet) [BẮT BUỘC]

- **Mục tiêu & Bản chất:** Huấn luyện mạng nơ-ron học sâu Dual-Head ResNet từ tập dữ liệu **Lichess Elite Database** ($\ge 50.000$ ván, $\text{ELO} \ge 2100$):
  - _Đầu vào:_ Tensor chuẩn hóa góc nhìn bên đi $\mathbb{R}^{18 \times 8 \times 8}$ (`canonical_flip: true`).
  - _Cấu trúc:_ Khối thân chung gồm 6 Residual Blocks (64 filters), phân nhánh thành Policy Head (Conv $1 \times 1$ với 16 filters $\rightarrow$ Linear 4096 chiều) và Value Head (Conv $1 \times 1$ với 1 filter $\rightarrow$ Linear 64 $\rightarrow$ Linear 1 $\rightarrow$ Tanh trong $[-1, 1]$).
  - _Cơ chế ra quyết định:_ Chọn nước đi hợp lệ có xác suất dự đoán cao nhất ($\arg\max_{a \in \text{legal}} P(s, a)$) sau khi áp dụng Legal Action Masking.
  - _Phân chia dữ liệu & Khử trùng:_ Chia tách ở cấp độ ván đấu (Game-level split: 80% Train, 10% Val, 10% Test) và khử trùng lặp thế cờ bằng khóa băm Zobrist 64-bit.

### Cấp 6: Reinforcement Learning [MỞ RỘNG]

- **Mục tiêu & Bản chất:** Học tăng cường trực tiếp từ tương tác môi trường và tín hiệu phần thưởng trên bài toán cờ tàn chuẩn FIDE **KRK (Vua + Xe Trắng đấu Vua Đen)**:
  - _Phương pháp:_ Cài đặt Tabular Q-Learning (trên không gian trạng thái rút gọn quy chiếu đối xứng 8 lần của bộ 3 quân $K_W, R_W, K_B$) và Policy Gradient (REINFORCE), chạy trực tiếp trên `ChessEnv` khởi tạo từ FEN KRK.
  - _Cấu trúc hàm thưởng:_ Chiếu hết đối thủ: $+1.0$; để mất Xe hoặc hòa pat/50 nước: $-0.2$; phạt mỗi bước đi: $-0.01$ (với hệ số chiết khấu $\gamma = 0.99$) nhằm buộc tác tử học đường chiếu hết ngắn nhất.
  - _Sản phẩm thực nghiệm:_ Xuất đường cong học tập (Learning Curve: tỷ lệ chiếu hết thành công & độ dài ván trung bình theo số episode huấn luyện).

### Cấp 7: Guided MCTS (AlphaGo-inspired / PUCT) [MỞ RỘNG]

- **Mục tiêu & Bản chất:** Kết hợp mạng Dual-Head ResNet (đã huấn luyện có giám sát ở Cấp 5) với cây tìm kiếm MCTS theo công thức PUCT (`chess_ai/search/mcts_puct.py`):
  - _Selection theo công thức PUCT:_
    $$a_t = \arg\max_{a \in \text{legal}(s)} \left[ Q(s, a) + c_{\text{puct}} \cdot P(s, a) \cdot \frac{\sqrt{\sum_b N(s, b)}}{1 + N(s, a)} \right]$$
  - _Evaluation tại lá:_ Đánh giá trực tiếp trạng thái lá bằng Value Head $V(s) \in [-1, 1]$ (thay cho rollout ngẫu nhiên).
  - _Tách biệt Chế độ Thi đấu (`eval_mode`) và Tự chơi (`selfplay_mode`):_ Khi thi đấu giải (`eval_mode`), tắt nhiễu Dirichlet ($\epsilon = 0$) và chọn nước đi theo $\arg\max_a N(s, a)$ để tối đa hóa sức cờ ổn định.
- **Bảng đối chiếu các thành phần Đã hiện thực / Chưa hiện thực so với AlphaGo nguyên bản (theo yêu cầu Mục 3.2 đề bài):**

| Thành phần Kiến trúc            | AlphaGo (Silver et al., 2016)                                                    | Cấp 7 (Hệ thống Hiện thực)                                                         | Ghi chú Kỹ thuật                                                                 |
| :------------------------------ | :------------------------------------------------------------------------------- | :--------------------------------------------------------------------------------- | :------------------------------------------------------------------------------- |
| **SL Policy Network**           | Mạng CNN 13 lớp học từ KGS                                                       | **Đã hiện thực** (Nhánh Policy của Dual-Head ResNet 6 blocks học từ Lichess Elite) | Dùng làm xác suất tiên nghiệm $P(s, a)$ trong PUCT                               |
| **Value Network**               | Huấn luyện riêng trên dữ liệu tự chơi của RL Policy                              | **Đã hiện thực (Hợp nhất)** (Nhánh Value chung thân ResNet huấn luyện đa nhiệm)    | Tiết kiệm $50\%$ chi phí suy luận GPU nhờ kiến trúc Dual-Head                    |
| **RL Policy Network**           | Tinh chỉnh SL Policy bằng REINFORCE self-play                                    | **Giản lược ở Cấp 7** (Chuyển sang khảo sát ở Cấp 6 & Cấp 8)                       | Cấp 7 dùng trực tiếp trọng số SL để cô lập tác động của cây PUCT                 |
| **Fast Rollout Policy**         | Trộn $\lambda \cdot V(s_L) + (1-\lambda) \cdot z_L$ với rollout tuyến tính nhanh | **Không dùng Rollout ($\lambda = 1.0$)**                                           | Định giá lá thuần túy bằng Value Head $V(s)$ theo chuẩn AlphaGo Zero / AlphaZero |
| **Asynchronous Multi-GPU MCTS** | Duyệt cây bất đồng bộ kèm Virtual Loss trên cụm phân tán                         | **Đơn luồng / Subtree Reuse** trên 1 GPU/CPU cá nhân                               | Phù hợp với ngân sách phần cứng máy trạm sinh viên                               |

### Cấp 8: AlphaZero-inspired [MỞ RỘNG]

- **Mục tiêu & Bản chất:** Thiết lập kiến trúc vòng lặp tự chơi (Self-play Loop) khép kín (`chess_ai/agents/level8_alphazero.py`):
  - _Vòng lặp tự học:_ Tác tử tự đấu bằng thuật toán PUCT ở chế độ `selfplay_mode` (bật nhiễu Dirichlet $\text{Dir}(\alpha=0.3), \epsilon=0.25$ tại gốc; nhiệt độ $\tau = 1.0$ trong 20 plies đầu), thu thập bộ ba $(s_t, \mathbf{\pi}_t, z_t)$ vào `ReplayBuffer`.
  - _Cập nhật lặp & Đánh giá Checkpoint (Arena Gating):_ Huấn luyện mạng trên mẫu từ Replay Buffer; định kỳ cho mô hình mới đấu $40$ ván với mô hình tốt nhất hiện tại, chỉ cập nhật trọng số khi tỷ lệ điểm đạt $\ge 55\%$.
  - _Phạm vi thực thi thực tế:_ Vận hành thực nghiệm theo chế độ **Self-play Fine-tuning khởi tạo từ Cấp 5** (và kiểm nghiệm hội tụ từ đầu trên thế cờ tàn thu gọn), đồng thời phân tích định lượng rào cản tài nguyên của Tabula Rasa trên bàn cờ $8 \times 8$ trong Chương 5.

---

## PHẦN 4: HỢP ĐỒNG GIAO DIỆN LỚP TRỪU TƯỢNG (INTERFACES)

### 4.1. Hợp đồng Môi trường Trò chơi: `chess_ai/core/game_env.py`

Tuân thủ đầy đủ các phương thức API theo Mục 3.1 của đề bài:

```python
from abc import ABC, abstractmethod
from typing import List, Tuple, Any, Optional

class GameEnv(ABC):
    """
    Abstract Base Class xác lập hợp đồng API thống nhất cho môi trường game đối kháng.
    """

    @abstractmethod
    def reset(self) -> Any:
        """Khởi tạo lại trạng thái ván đấu ban đầu. Trả về trạng thái quan sát."""
        pass

    @abstractmethod
    def current_player(self) -> int:
        """Trả về người chơi đến lượt: +1 (Trắng / Tiên) hoặc -1 (Đen / Hậu)."""
        pass

    @abstractmethod
    def legal_actions(self) -> List[Any]:
        """Trả về danh sách toàn bộ hành động hợp lệ tại trạng thái hiện tại."""
        pass

    @abstractmethod
    def step(self, action: Any) -> Tuple[Any, float, bool, dict]:
        """
        Thực thi một hành động hợp lệ trên môi trường. Cấm tuyệt đối nước đi bất hợp lệ.
        Trả về tuple: (state_observation, reward, done, info).
        """
        pass

    @abstractmethod
    def is_terminal(self) -> bool:
        """Kiểm tra xem trạng thái hiện tại đã kết thúc (chiếu hết, hòa cờ) hay chưa."""
        pass

    @abstractmethod
    def result(self) -> Optional[float]:
        """
        Trả về kết quả ván đấu theo góc nhìn tuyệt đối của bên Trắng (White perspective):
        +1.0: Trắng thắng | -1.0: Đen thắng | 0.0: Hòa | None: Ván cờ chưa kết thúc.
        """
        pass

    @abstractmethod
    def render(self, mode: str = "unicode") -> Optional[str]:
        """Hiển thị trực quan trạng thái bàn cờ (unicode, ascii, hoặc fen)."""
        pass

    @abstractmethod
    def clone(self) -> 'GameEnv':
        """Tạo một bản sao độc lập (bao gồm lịch sử nước đi) phục vụ phân nhánh tìm kiếm."""
        pass

    @abstractmethod
    def undo(self) -> None:
        """Hoàn tác nước cờ gần nhất trên ngăn xếp."""
        pass
```

### 4.2. Bộ Chuyển Đổi Môi Trường Cờ Vua: `chess_ai/core/chess_env.py`

```python
import chess
from typing import List, Tuple, Optional
from chess_ai.core.game_env import GameEnv

class ChessEnv(GameEnv):
    """
    Hiện thực hóa GameEnv cho trò chơi Cờ vua FIDE 8x8 dựa trên python-chess Bitboard.
    """
    def __init__(self, fen: Optional[str] = None):
        self.initial_fen = fen
        self.board = chess.Board(fen) if fen else chess.Board()

    def reset(self, fen: Optional[str] = None) -> chess.Board:
        target_fen = fen if fen is not None else self.initial_fen
        self.board = chess.Board(target_fen) if target_fen else chess.Board()
        return self.board

    def current_player(self) -> int:
        return 1 if self.board.turn == chess.WHITE else -1

    def legal_actions(self) -> List[chess.Move]:
        return list(self.board.legal_moves)

    def step(self, action: chess.Move) -> Tuple[chess.Board, float, bool, dict]:
        if action not in self.board.legal_moves:
            raise ValueError(f"Nước cờ bất hợp lệ: {action} tại thế cờ {self.board.fen()}")

        moving_color = self.board.turn
        self.board.push(action)
        done = self.is_terminal()
        res = self.result()

        # Phần thưởng tính theo góc nhìn tương đối của người chơi vừa thực hiện nước đi
        reward = 0.0
        if done and res is not None:
            if res == 1.0:
                reward = 1.0 if moving_color == chess.WHITE else -1.0
            elif res == -1.0:
                reward = 1.0 if moving_color == chess.BLACK else -1.0

        info = {
            "turn": self.board.turn,
            "is_check": self.board.is_check(),
            "fen": self.board.fen(),
            "halfmove_clock": self.board.halfmove_clock,
            "move_count": len(self.board.move_stack),
        }
        return self.board, reward, done, info

    def is_terminal(self) -> bool:
        return self.board.is_game_over(claim_draw=True)

    def is_terminal_fast(self) -> bool:
        """
        Kiểm tra kết thúc nhanh tại các nút bên trong cây tìm kiếm (không quét lại toàn bộ
        move_stack cho threefold repetition; việc lặp thế cờ trong cây được kiểm tra qua Zobrist path).
        """
        return (
            self.board.is_checkmate()
            or self.board.is_stalemate()
            or self.board.is_insufficient_material()
            or self.board.is_seventyfive_moves()
            or self.board.halfmove_clock >= 100
        )

    def result(self) -> Optional[float]:
        if not self.is_terminal():
            return None
        outcome = self.board.outcome(claim_draw=True)
        if outcome is None or outcome.winner is None:
            return 0.0
        return 1.0 if outcome.winner == chess.WHITE else -1.0

    def render(self, mode: str = "unicode") -> Optional[str]:
        if mode == "unicode":
            return self.board.unicode()
        if mode == "fen":
            return self.board.fen()
        return str(self.board)

    def clone(self) -> 'ChessEnv':
        new_env = ChessEnv.__new__(ChessEnv)
        new_env.initial_fen = self.initial_fen
        # Bảo toàn move_stack (stack=True) để hỗ trợ undo() và phát hiện lặp thế cờ 3 lần chính xác
        new_env.board = self.board.copy(stack=True)
        return new_env

    def undo(self) -> None:
        if len(self.board.move_stack) > 0:
            self.board.pop()
```

### 4.3. Hợp đồng Tác tử Cờ vua Căn bản: `chess_ai/agents/base_agent.py`

```python
from abc import ABC, abstractmethod
import chess
from typing import Optional, Dict, Any
from chess_ai.core.chess_env import ChessEnv

class BaseAgent(ABC):
    """
    Lớp cơ sở trừu tượng cho toàn bộ 8 cấp độ tác tử AI trong hệ thống.
    """
    def __init__(self, name: str):
        self.name = name
        self.last_search_stats: Dict[str, Any] = {}

    @abstractmethod
    def select_move(
        self,
        env: ChessEnv,
        time_limit: Optional[float] = None,
        max_workload: Optional[int] = None
    ) -> chess.Move:
        """
        Lựa chọn nước đi hợp lệ tốt nhất theo hợp đồng ngân sách kiểm soát kép:
        - time_limit: Giới hạn thời gian suy nghĩ tối đa (giây) cho chế độ Time-Control.
        - max_workload: Giới hạn số nút duyệt (Alpha-Beta) hoặc số mô phỏng (MCTS) cho chế độ Workload-Control.
        Đồng thời ghi nhận thống kê (nodes, simulations, depth, eval) vào self.last_search_stats.
        """
        pass

    def reset(self) -> None:
        """Tái thiết lập trạng thái nội tại (Transposition Table, cây MCTS) khi sang ván mới."""
        self.last_search_stats.clear()
```

---

## PHẦN 5: ĐẶC TẢ BIỂU DIỄN TENSOR 18 KÊNH GÓC NHÌN BÊN ĐI (`[18, 8, 8]`)

Theo quyết định kiến trúc **ADR-02** và **ADR-03** (`docs/DECISIONS.md`), toàn bộ mô hình học sâu và đường ống dữ liệu thống nhất sử dụng **Biểu diễn Chuẩn hóa Góc nhìn Bên Đi (Canonical Side-to-Move Perspective)** với kích thước chuẩn PyTorch CHW là $\mathbb{R}^{18 \times 8 \times 8}$:

### 5.1. Phép Lật Tọa Độ Chuẩn Hóa (`canonical_flip = True`)

- Khi tới lượt Trắng (`board.turn == chess.WHITE`): Giữ nguyên chỉ số ô `sq` $\in [0, 63]$ (hàng $r = \text{sq} // 8$, cột $c = \text{sq} \% 8$).
- Khi tới lượt Đen (`board.turn == chess.BLACK`): Lật đối xứng dọc bàn cờ qua phép XOR hàng `sq_canon = sq ^ 56` (tức $r_{\text{canon}} = 7 - r, c_{\text{canon}} = c$).
- **Ưu điểm học thuật:** Bên tới lượt đi luôn nằm ở nửa dưới bàn cờ và tiến Tốt theo hướng từ hàng $0 \to 7$, giúp các bộ lọc tích chập $3 \times 3$ học bất biến hình học hiệu quả hơn mà không bị nhầm lẫn hướng tiến quân.

### 5.2. Cấu Trúc 18 Kênh Đặc Trưng (`(18, 8, 8)`)

- **Kênh 0 – 5 (Quân của Bên tới lượt đi - Active Player):** Tốt (0), Mã (1), Tượng (2), Xe (3), Hậu (4), Vua (5) — mặt phẳng nhị phân $\{0, 1\}$.
- **Kênh 6 – 11 (Quân của Đối thủ - Opponent Player):** Tốt (6), Mã (7), Tượng (8), Xe (9), Hậu (10), Vua (11) — mặt phẳng nhị phân $\{0, 1\}$.
- **Kênh 12 – 15 (4 Mặt phẳng Quyền Nhập Thành Nhị phân Riêng biệt):**
  - Kênh 12: Quyền nhập thành cánh Vua của bên đi (`Active_KS`) $\in \{0, 1\}$.
  - Kênh 13: Quyền nhập thành cánh Hậu của bên đi (`Active_QS`) $\in \{0, 1\}$.
  - Kênh 14: Quyền nhập thành cánh Vua của đối thủ (`Opp_KS`) $\in \{0, 1\}$.
  - Kênh 15: Quyền nhập thành cánh Hậu của đối thủ (`Opp_QS`) $\in \{0, 1\}$.
- **Kênh 16 (Mặt phẳng Bắt Tốt Qua Đường - En-Passant):** Nhận giá trị $1$ tại đúng ô đích bắt Tốt qua đường hợp lệ (đã quy chiếu theo góc nhìn bên đi), $0$ tại các ô còn lại.
- **Kênh 17 (Mặt phẳng Bộ đếm 50 Nước - Halfmove Clock):** Toàn mặt phẳng nhận giá trị thực $\min(\text{halfmove\_clock} / 100.0, 1.0) \in [0.0, 1.0]$ (khi lưu đĩa dưới dạng `uint8`, lưu số nguyên $\min(\text{halfmove\_clock}, 100)$ rồi chia $100.0$ trên GPU, bảo toàn chính xác $100\%$ thông tin).

### 5.3. Không Gian Hành Động 4096 Chiều & Quy Tắc Phong Cấp

- Ánh xạ nước đi từ ô nguồn $u$ đến ô đích $v$ qua phép chuẩn hóa bên đi:
  $$u_{\text{canon}} = u \oplus (56 \text{ nếu Đen đi}), \quad v_{\text{canon}} = v \oplus (56 \text{ nếu Đen đi})$$
  $$\text{Action Index} = u_{\text{canon}} \times 64 + v_{\text{canon}} \in [0, 4095]$$
- **Quy ước Phong cấp:** Mặc định giải mã nước Tốt đi tới hàng cuối thành phong Hậu (`chess.QUEEN`); nếu nước phong Hậu không hợp lệ nhưng tồn tại nước phong cấp dưới hợp lệ cùng cặp $(u, v)$, hệ thống fallback chọn nước hợp lệ đầu tiên khớp $(u, v)$. Trong ETL huấn luyện, các mẫu phong cấp dưới hiếm gặp ($< 0.5\%$) được loại bỏ (`drop_underpromotions: true`).
- **Legal Action Masking:** Gán $-\infty$ cho toàn bộ chỉ số không thuộc `legal_actions` trước khi tính Softmax.

---

## PHẦN 6: TÓM TẮT GIAO THỨC THỰC NGHIỆM CHUẨN KHOA HỌC

Chi tiết đầy đủ được quy định tại `docs/EXPERIMENT_PROTOCOL.md` và `configs/default_config.yaml`:

1. **Giải đấu Round-Robin Đối Xứng (100 Ván/Cặp trên 50 Thế cờ Khai cuộc Cân bằng):**
   - Mỗi cặp tác tử thi đấu **100 ván** xuất phát từ **50 thế cờ khai cuộc cân bằng** (`resources/openings_50.epd`, sau 6–8 plies), mỗi khai cuộc đấu 2 ván đổi màu Trắng/Đen. Giải quyết triệt để hiện tượng lặp ván ở các tác tử tất định và khử hoàn toàn First-Player Advantage.
   - Cố định bộ 10 Random Seeds (`agent_seeds = [42, 101, 2024, 777, 12, 99, 555, 1337, 888, 314]`).
2. **Kiểm định Thống kê Kép:**
   - Tính điểm chuẩn hóa $S = \frac{W + 0.5 \times D}{N}$ kèm **Khoảng tin cậy Wilson Score 95%** và **Khoảng tin cậy Bootstrap 95% theo cặp khai cuộc** ($10.000$ resamples).
3. **Hai Bài Ablation / Sensitivity Studies Trọng Tâm:**
   - _Ablation 1 (Tầng Cổ điển trên bộ WAC-50 chuẩn):_ Đánh giá 5 cấu hình lũy tiến (`Minimax` $\rightarrow$ `Alpha-Beta` $\rightarrow$ `+Move Ordering` $\rightarrow$ `+Transposition Table` $\rightarrow$ `+Quiescence Search`) và khảo sát độ nhạy theo độ sâu $d \in \{2, 3, 4\}$.
   - _Ablation 2 (Tầng MCTS & Học sâu):_ Khảo sát độ nhạy số lượt mô phỏng $N_{\text{sim}} \in \{50, 100, 200, 400, 800, 1000\}$ cho **cả Cấp 4 (UCT) và Cấp 7 (PUCT)**; đồng thời so sánh đối đầu trực tiếp giữa Policy-only (Cấp 5), Value-guided MCTS, và Full Guided MCTS (Cấp 7).
