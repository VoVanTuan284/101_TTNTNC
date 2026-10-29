# KHUNG BÁO CÁO KHOA HỌC & TÀI LIỆU TRÍCH DẪN (REPORT_OUTLINE.md)

**Mã tài liệu:** `REPORT_OUTLINE_2026`  
**Dự án:** Multi-Level Chess AI Engine ($8 \times 8$ FIDE) *(Đã được giảng viên phê duyệt)*  
**Môn học:** Trí tuệ Nhân tạo Nâng cao  
**Định dạng chuẩn:** IEEE Conference / Journal Paper Format (Hai cột, cấu trúc 5 chương)  
**Nguồn sự thật cấu hình:** `configs/` & `docs/DECISIONS.md`  
**Quy mô báo cáo:** $15 \div 25$ trang nội dung học thuật chuyên sâu kèm phụ lục thực nghiệm  

---

## PHẦN 1: BỐN CÂU HỎI NGHIÊN CỨU CỐT LÕI (RESEARCH QUESTIONS - RQ1 ĐẾN RQ4)

Báo cáo khoa học tập trung giải quyết và đưa ra luận cứ định lượng cho 4 câu hỏi nghiên cứu trung tâm:

* **RQ1 (Tác động bóc tách của Policy Prior $P(s, a)$ và Value Head $V(s)$ trong cây tìm kiếm PUCT):**  
  *Câu hỏi:* Khi tích hợp mạng Dual-Head ResNet vào thuật toán tìm kiếm cây Monte Carlo (Cấp 7), từng thành phần xác suất tiên nghiệm $P(s, a)$ và hàm định giá trạng thái lá $V(s)$ đóng góp riêng biệt bao nhiêu phần trăm vào mức gia tăng tỷ lệ điểm số $S$ so với việc chỉ dùng mạng Policy đơn độc (Cấp 5), chỉ dùng Value-guided MCTS (Uniform Prior + $V(s)$), hoặc dùng thuật toán Alpha-Beta cổ điển (Cấp 3)?
* **RQ2 (Hiệu quả cắt giảm nút duyệt và giảm thiểu Horizon Effect trong tìm kiếm cổ điển):**  
  *Câu hỏi:* Trong tầng tìm kiếm đối kháng cổ điển (Cấp 3), chiến lược sắp xếp nước đi (Move Ordering) kết hợp bảng chuyển vị Zobrist (Transposition Table) cắt giảm bao nhiêu phần trăm số nút duyệt ở cùng độ sâu $d \in \{2, 3, 4\}$, và Quiescence Search cải thiện tỷ lệ giải đúng (Tactical Accuracy %) như thế nào thông qua việc giảm thiểu hiện tượng "mù chân trời" (Horizon Effect) trên bộ 50 thế cờ chiến thuật chuẩn Win At Chess (`WAC.001` – `WAC.050`)?
* **RQ3 (Quy luật suy giảm lợi nhuận biên theo ngân sách mô phỏng MCTS ở Cấp 4 và Cấp 7):**  
  *Câu hỏi:* Khi tăng ngân sách mô phỏng $N_{\text{sim}} \in \{50, 100, 200, 400, 800, 1000\}$, đường cong hiệu năng của cả **Vanilla MCTS UCT (Cấp 4)** lẫn **Guided MCTS PUCT (Cấp 7)** có tuân theo quy luật suy giảm lợi nhuận biên (Diminishing Returns) hay không, và đâu là điểm cân bằng Pareto tối ưu giữa sức cờ và độ trễ suy luận $p95$?
* **RQ4 (Định lượng sự khác biệt hành vi giữa Heuristic thủ công và Đặc trưng Học sâu):**  
  *Câu hỏi:* Sự chuyển dịch từ hàm lượng giá tĩnh thủ công (Piece-Square Tables Michniewski ở Cấp 2, Cấp 3) sang đặc trưng tích chập 18 kênh góc nhìn bên đi (Dual-Head ResNet ở Cấp 5, Cấp 7) làm thay đổi các chỉ số định lượng hành vi thi đấu — bao gồm tỷ lệ đồng thuận nước đi (Move Agreement Rate), tần suất nhập thành sớm (Castling Rate), độ dài ván đấu trung bình và tỷ lệ thắng chiếu hết — như thế nào?

---

## PHẦN 2: DÀN Ý CHI TIẾT 5 CHƯƠNG BÁO CÁO KHOA HỌC

```text
BÁO CÁO TỔNG KẾT ĐỀ TÀI CUỐI KỲ:
XÂY DỰNG VÀ ĐÁNH GIÁ HỆ THỐNG GAME AI CỜ VUA ĐA CẤP ĐỘ (8x8 FIDE)
├── TÓM TẮT (ABSTRACT) & TỪ KHÓA (KEYWORDS)
├── CHƯƠNG 1: GIỚI THIỆU & ĐỊNH HÌNH BÀI TOÁN
├── CHƯƠNG 2: CƠ SỞ LÝ THUYẾT & CÔNG TRÌNH LIÊN QUAN
├── CHƯƠNG 3: PHƯƠNG PHÁP & THIẾT KẾ HỆ THỐNG
├── CHƯƠNG 4: THỰC NGHIỆM, KẾT QUẢ & THẢO LUẬN
├── CHƯƠNG 5: KẾT LUẬN & HƯỚNG PHÁT TRIỂN
├── TÀI LIỆU THAM KHẢO (REFERENCES)
└── PHỤ LỤC (APPENDIX)
```

---

### TÓM TẮT (ABSTRACT) & TỪ KHÓA
* **Nội dung:** Tóm tắt trong $250 \div 300$ từ: Bối cảnh bài toán cờ vua chuẩn FIDE $8 \times 8$, hệ thống 8 cấp độ tác tử AI, biểu diễn Tensor chuẩn hóa góc nhìn bên đi $\mathbb{R}^{18 \times 8 \times 8}$, quy trình ETL từ Lichess Elite Database có khử trùng Zobrist ($\approx 1.000.000$ mẫu), thể thức giải đấu Round-Robin 100 ván/cặp trên 50 thế cờ khai cuộc cân bằng với khoảng tin cậy kép (Wilson Score 95% & Paired Bootstrap 95%), kết quả 2 bài Ablation Studies và đóng góp thực nghiệm của nhóm.
* **Từ khóa (Keywords):** Game AI, Computer Chess, Iterative Deepening Alpha-Beta, Quiescence Search, Monte Carlo Tree Search (UCT/PUCT), Dual-Head ResNet, Wilson & Bootstrap Confidence Intervals.

---

### CHƯƠNG 1: GIỚI THIỆU & ĐỊNH HÌNH BÀI TOÁN (INTRODUCTION)
* **1.1. Bối cảnh, Luật Cờ Vua FIDE & Độ Phức Tạp Bài Toán:**
  * Ghi nhận phê duyệt chính thức của giảng viên đối với đề tài Cờ vua chuẩn FIDE $8 \times 8$.
  * Đặc trưng môi trường trò chơi: Trò chơi đối kháng 2 người chơi, tổng bằng không (Zero-sum), thông tin hoàn hảo (Perfect Information), tất định (Deterministic).
  * Phân biệt chuẩn xác hai khái niệm độ phức tạp: **Độ phức tạp không gian trạng thái hợp lệ (State-Space Complexity $\approx 10^{44} \div 10^{47}$)** và **Độ phức tạp cây trò chơi (Game-Tree Complexity / Số Shannon $\approx 10^{120}$)** với hệ số phân nhánh trung bình $b \approx 35$, độ dài trung bình $d \approx 70\text{--}80$ plies.
* **1.2. Mục tiêu Nghiên cứu & Phạm vi 8 Cấp độ Tác tử:**
  * Xây dựng đầy đủ 8 cấp độ tác tử (Cấp 1–5 và Cấp 7 trên bàn cờ $8 \times 8$ toàn phần; Cấp 6 học tăng cường trên thế cờ tàn chuẩn KRK; Cấp 8 thực nghiệm vòng lặp Self-play Fine-tuning kèm Checkpoint Arena).
* **1.3. Bốn Câu hỏi Nghiên cứu (RQ1 – RQ4):** Trình bày rõ giả thuyết và độ đo kiểm định cho từng câu hỏi.
* **1.4. Đóng góp Kỹ thuật & Thực nghiệm Chính:**
  1. Kiến trúc Clean Architecture 46 tệp mã nguồn tách biệt rõ môi trường, tìm kiếm, mạng nơ-ron, đấu trường và giao diện.
  2. Biểu diễn Tensor 18 kênh chuẩn hóa góc nhìn bên đi (`[18, 8, 8]`, lật bàn cờ `sq ^ 56`) bảo toàn trọn vẹn quyền nhập thành 4 mặt phẳng, ô En-Passant và bộ đếm 50 nước mà không mất mát độ chính xác khi lưu `uint8`.
  3. Đường ống ETL ngoại tuyến từ Lichess Elite Database tích hợp lọc Header nhanh, phân chia Game-level split tất định và khử trùng lặp thế cờ bằng khóa băm Zobrist 64-bit.
  4. Giao thức thực nghiệm đối xứng 100 ván/cặp trên 50 thế cờ khai cuộc cân bằng, kiểm định thống kê kép (Wilson 95% & Paired Bootstrap 95%) cùng 2 bài Ablation chuyên sâu.
* **1.5. Danh mục Sản phẩm Bàn giao:** Mã nguồn, cấu hình YAML, bộ test 61 kịch bản, checkpoint `.pt`, log thô PGN/JSONL, notebook `notebooks/analysis.ipynb`, báo cáo, slide, poster và video demo.

---

### CHƯƠNG 2: CƠ SỞ LÝ THUYẾT & CÔNG TRÌNH LIÊN QUAN (THEORETICAL BACKGROUND)
* **2.1. Tìm kiếm Đối kháng & Khuôn mẫu Negamax:**
  * Định lý Minimax của Von Neumann và công thức rút gọn Negamax theo góc nhìn bên đi.
* **2.2. Tối ưu hóa Tầng Tìm kiếm Cổ điển:**
  * Cắt tỉa Alpha-Beta ($\alpha$-$\beta$ Pruning) và phân tích độ phức tạp trường hợp tốt nhất (Best-case Asymptotic Complexity) $\mathcal{O}(b^{d/2})$ khi thứ tự duyệt nước đi đạt tối ưu.
  * Iterative Deepening (ID), bảng chuyển vị Zobrist Transposition Table (cờ `EXACT`, `LOWERBOUND`, `UPPERBOUND`) và chiến lược Move Ordering (PV-move, MVV-LVA, Killer Moves, History Heuristic).
  * Hiện tượng mù chân trời (Horizon Effect) và cơ chế giảm thiểu bằng Quiescence Search kết hợp Delta Pruning.
* **2.3. Tìm kiếm Cây Monte Carlo (MCTS / UCT):**
  * Bài toán Multi-armed Bandit, công thức UCB1/UCT và chu trình 4 bước.
  * Lý do cần cắt ngưỡng độ sâu mô phỏng (`max_rollout_depth`) kết hợp định giá tĩnh trong cờ vua.
* **2.4. Mạng Nơ-ron Đa Nhiệm, Học Tăng Cường & Phân Biệt AlphaGo vs AlphaZero:**
  * Kiến trúc Dual-Head ResNet (Shared Residual Trunk, Policy Head, Value Head).
  * Học tăng cường Q-Learning và Policy Gradient (REINFORCE) trên không gian cờ tàn KRK.
  * Công thức chọn nhánh PUCT và sự khác biệt bản chất giữa **AlphaGo-inspired (Cấp 7 - khởi tạo từ học có giám sát trên dữ liệu kỳ thủ)** và **AlphaZero-inspired (Cấp 8 - vòng lặp tự chơi Self-play + Checkpoint Arena)**.

---

### CHƯƠNG 3: PHƯƠNG PHÁP & THIẾT KẾ HỆ THỐNG (METHODOLOGY & SYSTEM DESIGN)
* **3.1. Kiến trúc Hệ thống & Tuyên bố Liêm chính Học thuật:**
  * Sơ đồ khối Clean Architecture và hợp đồng API `GameEnv` (`reset`, `current_player`, `legal_actions`, `step`, `is_terminal`, `result`, `render`, `clone`, `undo`).
  * Bảng phân định rõ phần kế thừa hợp lệ từ `python-chess` và phần nhóm tự phát triển $100\%$.
* **3.2. Biểu diễn Trạng thái Tensor `(18, 8, 8)` & Hành động 4096 Chiều:**
  * Phép lật chuẩn hóa góc nhìn bên đi (`sq ^ 56` khi Đen đi), ý nghĩa từng kênh từ 0 đến 17, quy tắc giải mã phong cấp và Legal Action Masking.
* **3.3. Thiết kế Chi tiết 8 Cấp độ Tác tử:**
  * Trình bày thuật toán, giả mã và siêu tham số (tham chiếu `configs/search_params.yaml`) cho Cấp 1 $\to$ Cấp 8.
  * **Bảng đối chiếu các thành phần Đã hiện thực / Chưa hiện thực của Cấp 7 so với AlphaGo nguyên bản** (theo đúng yêu cầu Mục 3.2 đề bài).
* **3.4. Đường ống Dữ liệu Lichess Elite & Quy trình Huấn luyện:**
  * Bộ lọc 5 tiêu chí ở mức Header, lấy mẫu cách quãng $k \in \{3, 4\}$, phân tách Game-level split tất định bằng MD5, và khử trùng lặp thế cờ bằng khóa băm Zobrist 64-bit.
  * Thảo luận về đặc thù nguồn nhãn chuyên gia và phương sai nhiễu của nhãn kết cục ván đấu $z \in \{-1, 0, +1\}$ ở giai đoạn trung cuộc.
* **3.5. Thiết kế Sân đấu Arena 50 Khai cuộc & Giao diện Pygame Đa tiến trình:**
  * Cơ chế ghép cặp đối xứng 50 thế cờ khai cuộc cân bằng $\times 2$ màu = 100 ván/cặp.
  * Hai chế độ kiểm soát ngân sách (Workload-Control và Time-Control với Cooperative Time Management).

---

### CHƯƠNG 4: THỰC NGHIỆM, KẾT QUẢ & THẢO LUẬN (EXPERIMENTS & RESULTS)
* **4.1. Môi trường Thực nghiệm, Cấu hình Phần cứng & Thời gian Huấn luyện:**
  * Bảng khai báo chi tiết cấu hình CPU, GPU, RAM, phiên bản Python/PyTorch, thời gian chạy ETL và tổng thời gian huấn luyện của Cấp 5, Cấp 6, Cấp 8.
* **4.2. Kết quả Huấn luyện Tác tử Học được (Cấp 5, Cấp 6, Cấp 8):**
  * Đồ thị hàm mất mát (Policy Cross-Entropy, Value MSE), Top-1 / Top-3 Policy Accuracy và so sánh Value MSE với baseline dự đoán hằng số $0$ trên tập Validation/Test.
  * Đường cong học tập (Learning Curve) của Cấp 6 trên cờ tàn KRK và kết quả đối đầu Checkpoint Arena của Cấp 8.
* **4.3. Ma trận Kết quả Giải đấu Round-Robin (100 Ván/Cặp trên 50 Khai cuộc):**
  * Bảng ma trận $W / D / L$ và tỷ lệ điểm chuẩn hóa $S = (W + 0.5D)/N$ kèm **Khoảng tin cậy Wilson Score 95%** và **Paired-Opening Bootstrap 95%** ở cả hai chế độ Workload-Control và Time-Control.
  * Phân tích lợi thế đi trước ($S_{\text{White}}$ vs $S_{\text{Black}}$) và độ ổn định qua 10 Random Seeds.
* **4.4. Đánh giá Chi phí Tính toán & Phân tích Đánh đổi (Trade-off Analysis):**
  * Bảng đối chiếu tốc độ duyệt (NPS/SPS), thời gian trung bình một nước, độ trễ $p95$, và mức tiêu thụ RAM/VRAM.
* **4.5. Kết quả Bài Ablation 1 (Tầng Tìm kiếm Cổ điển & Độ nhạy Độ sâu trên WAC-50):**
  * Bảng và biểu đồ so sánh 5 cấu hình lũy tiến (`Config_A` $\to$ `Config_D`) trên bộ `resources/benchmark_positions.epd` (`WAC.001` – `WAC.050`).
  * Phân tích độ nhạy theo độ sâu tìm kiếm $d \in \{2, 3, 4\}$.
* **4.6. Kết quả Bài Ablation 2 (Độ nhạy $N_{\text{sim}}$ của Cấp 4 & Cấp 7; Bóc tách Policy vs Value):**
  * Đường cong hiệu năng theo $N_{\text{sim}} \in \{50, 100, 200, 400, 800, 1000\}$ của cả **Cấp 4 (UCT)** và **Cấp 7 (PUCT)**.
  * Bảng đối đầu 3 biến thể: Policy-only (Cấp 5) vs Value-only MCTS vs Full Guided MCTS (Cấp 7).
* **4.7. Phân tích Định lượng Phong cách (RQ4) & Các Tình huống Chiến thuật Điển hình:**
  * Bảng chỉ số Move Agreement Rate, Castling Rate, độ dài ván trung bình.
  * Phân tích chi tiết các ván đấu thực tế và tình huống tác tử mắc sai lầm trích từ biên bản PGN.

---

### CHƯƠNG 5: KẾT LUẬN & HƯỚNG PHÁT TRIỂN (CONCLUSION & FUTURE WORK)
* **5.1. Tổng hợp Câu trả lời Định lượng cho 4 Câu hỏi Nghiên cứu (RQ1 – RQ4).**
* **5.2. Phân biệt Sức mạnh Thi đấu Thực nghiệm với Tính Tối ưu Toán học.**
* **5.3. Hạn chế của Hệ thống & Phân tích Rào cản Tính toán của AlphaZero Tabula Rasa trên Bàn cờ $8 \times 8$.**
* **5.4. Khả năng Tái lập Thực nghiệm (`--from-logs` & `--full`) và Hướng Phát triển Tương lai (Syzygy Tablebases, Batched MCTS với Virtual Loss, Nén mô hình ONNX/TensorRT).**

---

## PHẦN 3: BIỂU MẪU PHÂN TÍCH TÌNH HUỐNG CHIẾN THUẬT & SAI LẦM ĐIỂN HÌNH

Nhằm đáp ứng yêu cầu phân tích định tính kết hợp định lượng tại Mục 3.6 của đề cương môn học, mỗi tình huống sai lầm chiến thuật được trình bày chuẩn xác về mặt cơ chế thuật toán theo biểu mẫu sau:

```text
================================================================================
BIỂU MẪU PHÂN TÍCH THẾ CỜ CHIẾN THUẬT / HIỆN TƯỢNG MÙ CHÂN TRỜI (HORIZON EFFECT)
================================================================================
1. Nguồn dữ liệu / Mã kiểm định: Bài thực nghiệm Ablation 1 - Thế cờ WAC.011
   (Đối chứng hành vi của Config_C2 [Tắt Quiescence] vs Config_D [Bật Quiescence])
2. Thế cờ ghi nhận (FEN): r1b1kb1r/3q1ppp/pBp1pn2/8/Np3P2/5B2/PPP3PP/R2Q1RK1 w kq - 0 1
3. Lượt đi & Nước đi tối ưu (bm): Trắng đi nước Bxc6! (Ăn Tốt c6 ghim Hậu Đen d7).
4. Phân tích cơ chế sai lầm khi tắt Quiescence Search (Config_C2 ở độ sâu chẵn):
   - Diễn biến chuỗi bắt quân: Nếu Trắng đi 1. Bxc6 Qxc6 (ply 2) 2. Qxd8# hoặc nếu Đen 
     trì hoãn bằng nước chiếu trung gian / đổi quân tại một nhánh phụ khiến nước ăn lại 
     quân rơi đúng vào ply thứ d + 1 (ngoài chân trời độ sâu cố định d).
   - Sai lầm của Config_C2 (Tắt Quiescence): Khi dừng đột ngột tại nút lá d = 3 ngay sau 
     một nước bắt quân chưa được ăn trả (ví dụ Trắng vừa đưa quân vào ô bị bảo vệ để ăn 
     quân nhỏ), hàm lượng giá tĩnh chỉ đếm số quân hiện có trên bàn cờ và tưởng rằng 
     bên đi vừa lời chất (+300 cp), trong khi ở ngay ply thứ 4 đối thủ có nước bắt lại 
     quân lớn hơn (Recapture).
   - Cơ chế khắc phục của Config_D (Bật Quiescence Search): Tại nút lá d = 0, Quiescence 
     Search tiếp tục mở rộng các nước bắt quân hợp lệ (sắp xếp theo MVV-LVA) cho tới khi 
     không còn nước ăn quân có lợi (Quiet position). Nhờ nhìn thấy nước ăn trả của đối 
     thủ ở ply mở rộng, Config_D đánh giá đúng giá trị sau chuỗi trao đổi quân và chọn 
     chính xác nước đi chiến thuật Bxc6.
5. Bài học kỹ thuật rút ra:
   - Quiescence Search giải quyết hiệu quả các sai lầm do chuỗi trao đổi bắt quân bị cắt 
     ngang tại chân trời độ sâu cố định.
   - Giới hạn trung thực: Vì Quiescence Search chuẩn chỉ mở rộng các nước bắt quân và 
     thoát chiếu để tránh bùng nổ cây tìm kiếm, các đòn phối hợp dựa trên nước đi yên 
     tĩnh (Quiet tactical threats) vượt quá độ sâu d vẫn đòi hỏi phải tăng độ sâu tìm 
     kiếm hoặc kết hợp mạng dẫn hướng Policy/Value như ở Cấp 7.
================================================================================
```

---

## PHẦN 4: DANH MỤC TÀI LIỆU HỌC THUẬT BẮT BUỘC TRÍCH DẪN (REFERENCES)

Danh mục tài liệu tham khảo được định dạng chuẩn IEEE và khớp chính xác với Mục 8 của đề bài:

1. **[Liu 2024]** M. Liu, *AlphaGo Simplified: Rule-Based AI and Deep Learning in Everyday Games*. Boca Raton, FL, USA: CRC Press, 2024, ch. 1–21.
2. **[AIMA 2021]** S. Russell and P. Norvig, *Artificial Intelligence: A Modern Approach*, 4th ed. Hoboken, NJ, USA: Pearson, 2021, ch. 5 (Adversarial Search and Games).
3. **[Sutton 2018]** R. S. Sutton and A. G. Barto, *Reinforcement Learning: An Introduction*, 2nd ed. Cambridge, MA, USA: MIT Press, 2018.
4. **[Silver 2016 - AlphaGo]** D. Silver *et al.*, "Mastering the game of Go with deep neural networks and tree search," *Nature*, vol. 529, no. 7587, pp. 484–489, Jan. 2016.
5. **[Silver 2017 - AlphaZero]** D. Silver *et al.*, "Mastering Chess and Shogi by Self-Play with a General Reinforcement Learning Algorithm," *arXiv preprint arXiv:1712.01815*, Dec. 2017.
6. **[Silver 2018 - AlphaZero Science]** D. Silver *et al.*, "A general reinforcement learning algorithm that masters chess, shogi, and Go through self-play," *Science*, vol. 362, no. 6419, pp. 1140–1144, Dec. 2018.
7. **[Browne 2012 - MCTS Survey]** C. B. Browne *et al.*, "A Survey of Monte Carlo Tree Search Methods," *IEEE Transactions on Computational Intelligence and AI in Games*, vol. 4, no. 1, pp. 1–49, Mar. 2012.
8. **[Wilson 1927]** E. B. Wilson, "Probable Inference, the Law of Succession, and Statistical Inference," *Journal of the American Statistical Association*, vol. 22, no. 158, pp. 209–212, Jun. 1927.
9. **[Efron 1993 - Bootstrap]** B. Efron and R. J. Tibshirani, *An Introduction to the Bootstrap*. New York, NY, USA: Chapman & Hall/CRC, 1993.
10. **[Michniewski]** T. Michniewski, "Simplified Evaluation Function," *Chess Programming Wiki*. [Online]. Available: https://www.chessprogramming.org/Simplified_Evaluation_Function
11. **[Reinfeld 1958 - WAC]** F. Reinfeld, *Win at Chess*. New York, NY, USA: Dover Publications, 1958 (EPD Suite compiled by J. Dart, Arasan Chess).
12. **[FIDE Laws]** World Chess Federation (FIDE), *FIDE Laws of Chess*, FIDE Handbook, E.01, 2023.
