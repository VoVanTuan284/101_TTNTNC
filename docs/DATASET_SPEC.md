# ĐẶC TẢ DỮ LIỆU HUẤN LUYỆN & PIPELINE (DATASET_SPEC.md)

**Mã tài liệu:** `DATASET_SPEC_2026`  
**Dự án:** Multi-Level Chess AI Engine ($8 \times 8$ FIDE)  
**Môn học:** Trí tuệ Nhân tạo Nâng cao  
**Thành viên phụ trách chính:** TV2 (Deep Learning, ETL & Guided Search Lead) | Kiểm thử: TV3  
**Nguồn sự thật cấu hình:** `configs/default_config.yaml` (`data_etl`) & `configs/model_resnet.yaml`  
**Phạm vi tác tử ứng dụng:** Cấp 5 (Supervised Dual-Head ResNet), Cấp 7 (Guided MCTS PUCT) và khởi tạo Cấp 8  

---

## PHẦN 1: MỤC TIÊU & ĐỊNH HƯỚNG DỮ LIỆU

Để huấn luyện mạng nơ-ron đa nhiệm (Dual-Head ResNet) dẫn hướng cho tác tử Cấp 5, Cấp 7 và làm điểm tựa khởi tạo (Warm-start) cho Cấp 8, đường ống dữ liệu (ETL Pipeline) thực hiện hai nhiệm vụ học có giám sát cốt lõi:
1. **Dự đoán phân phối chính sách chuyên gia (Policy Head $P(s, a)$):** Thu hẹp không gian tìm kiếm từ trung bình $\approx 35$ nước đi hợp lệ xuống nhóm nhỏ các nước đi chiến thuật tiềm năng.
2. **Dự đoán giá trị thế trận tương đối (Value Head $V(s)$):** Ước lượng kỳ vọng kết cục ván đấu từ góc nhìn của bên tới lượt đi trong khoảng $[-1.0, 1.0]$.

Tài liệu này quy định quy trình thu thập, lọc header, khử trùng lặp (Deduplication), mã hóa Tensor 18 kênh chuẩn hóa góc nhìn bên đi, lưu trữ nhị phân `np.memmap` và nạp dữ liệu huấn luyện, bảo đảm:
* Triệt tiêu $100\%$ hiện tượng rò rỉ dữ liệu (Data Leakage) ở cả cấp độ ván đấu (Game-level split) lẫn cấp độ thế cờ (Zobrist Deduplication).
* Vận hành ổn định với mức chiếm dụng RAM hệ thống luôn duy trì $< 1.5\text{GB}$.
* Ghi nhận đầy đủ thời gian trích xuất, thời gian huấn luyện trên phần cứng mục tiêu (CPU/GPU) và phân tích hạn chế của nhãn chuyên gia theo đúng Mục 3.4 của đề bài.

---

## PHẦN 2: NGUỒN DỮ LIỆU & BỘ LỌC CHẤT LƯỢNG CAO (CURATION & DEDUPLICATION)

### 2.1. Nguồn Dữ liệu Chuẩn: Lichess Elite Database
* **Kho dữ liệu chính (`configs/default_config.yaml`):** **Lichess Elite Database** (`https://database.nikonoel.fr/`).
* **Lý do lựa chọn (theo ADR-04):** Thay vì đọc luồng trực tiếp từ tệp dump Lichess tháng ($\sim 30\text{GB}$ nén, trong đó tỷ lệ ván có cả hai bên $\text{ELO} \ge 2100$ dưới $2\%$ gây nghẽn mạng và dễ đứt kết nối), Lichess Elite Database đã lọc sẵn các ván đấu chất lượng cao giữa các kỳ thủ có hệ số ELO cao. Tệp nén `.zip` / `.pgn.zst` ($\sim 200 \div 500\text{MB}$) được tải về thư mục cục bộ `data/raw/` trước khi chạy ETL, đảm bảo tốc độ đọc đĩa nhanh và tái lập ngoại tuyến $100\%$.

### 2.2. Bộ Lọc 5 Tiêu Chí Bắt Buộc (Header-First Filtering)
Để tối ưu tốc độ ETL gấp $5 \div 8$ lần, script `chess_ai/etl/pgn_etl.py` sử dụng `chess.pgn.read_headers()` để kiểm tra nhanh siêu dữ liệu trước khi phân tích cú pháp cây nước đi:
1. **Trình độ kỳ thủ (Master Rating Filter):**
   $$\text{ELO}_{\text{White}} \ge 2100 \quad \text{và} \quad \text{ELO}_{\text{Black}} \ge 2100$$
2. **Loại hình kiểm soát thời gian (Continuous Speed Control Filter):**
   * Tính thời gian ước lượng của ván cờ theo công thức chuẩn FIDE/Lichess từ trường `TimeControl` (`base+increment`):
     $$T_{\text{est}} = T_{\text{base}} + 40 \times T_{\text{inc}} \ge 180\text{ giây}$$
   * Giữ lại toàn bộ các ván đấu có $T_{\text{est}} \ge 180\text{s}$ (bao phủ liền mạch `Blitz`, `Rapid`, `Classical` mà không tạo khoảng trống giữa $5\text{--}10$ phút hay $15\text{--}30$ phút), loại bỏ hoàn toàn `Bullet` và `UltraBullet` ($T_{\text{est}} < 180\text{s}$).
3. **Tính hợp lệ của kết cục & Độ dài tối thiểu (`Termination` & `Min Plies`):**
   * Chỉ chấp nhận các ván có header `Termination` là `"Normal"` hoặc `"Time forfeit"` và `Result` thuộc `{"1-0", "0-1", "1/2-1/2"}` (loại bỏ `"Abandoned"`, `"Rules infraction"`, `"Unterminated"`).
   * Tổng số plies của ván cờ phải đạt $\ge 30$ plies ($\ge 15$ nước đi trọn vẹn).
4. **Loại bỏ hòa thỏa thuận sớm (Early Draw Filter):**
   * Bác bỏ các ván có kết quả hòa (`1/2-1/2`) kết thúc trước **60 plies (30 nước đi)** mà tổng số quân trên bàn cờ còn $> 24$ quân (tránh đưa các ván hòa thỏa thuận ngoại giao vào làm nhiễu Value Head).
5. **Chuẩn hóa nhãn kết cục tuyệt đối ($R_{\text{abs}}$):**
   * Trắng thắng (`1-0`): $R_{\text{abs}} = +1.0$ | Đen thắng (`0-1`): $R_{\text{abs}} = -1.0$ | Hòa (`1/2-1/2`): $R_{\text{abs}} = 0.0$.

### 2.3. Chiến Lược Lấy Mẫu Cách Quãng (Subsampling) & Khử Trùng Lặp (Deduplication)
* **Subsampling giảm tương quan chuỗi:**
  * Bỏ qua **12 plies đầu tiên (6 nước khai cuộc)** của mỗi ván đấu (`skip_opening_plies: 12`) để tránh mô hình chỉ học thuộc lòng vài biến khai cuộc phổ biến.
  * Lấy mẫu cách quãng với bước nhảy ngẫu nhiên đồng đều $k \in \{3, 4\}$ plies (`subsample_k_min: 3`, `subsample_k_max: 4` sử dụng bộ sinh số ngẫu nhiên có seed cố định `random_seed: 42`).
* **Khử trùng lặp thế cờ bằng khóa băm Zobrist (`deduplicate_zobrist: true`):**
  * Theo yêu cầu tại Mục 3.4 của đề bài (*"mô tả cách loại trùng"*), hệ thống duy trì một bảng băm `seen_zobrist_keys` (64-bit Zobrist hash từ `chess.polyglot.zobrist_hash(board)`).
  * Nếu một thế cờ $s$ đã xuất hiện trước đó (dù trong cùng tập hay khác tập), mẫu trùng lặp sẽ bị bỏ qua ngay lập tức. Cơ chế này đảm bảo:
    1. Tập dữ liệu không bị lệch phân phối (bias) về các thế cờ trung cuộc xuất hiện lặp đi lặp lại.
    2. **Không tồn tại bất kỳ thế cờ nào trùng lặp giữa tập Train, Validation và Test.**

### 2.4. Phân Tích Nguồn Nhãn & Rủi Ro Học Lại Bias/Nhiễu (Theo Mục 3.4 Đề Bài)
* **Đặc thù nhãn Policy ($a^*$):** Nhãn hành động là nước đi thực tế của kỳ thủ ($\text{ELO} \ge 2100$). Dù chất lượng cao hơn nhiều so với người chơi phổ thông, nước đi trong các ván Blitz vẫn có thể chứa sai sót dưới áp lực thời gian hoặc phản ánh thiên kiến phong cách con người.
* **Đặc thù nhãn Value ($z \in \{-1, 0, +1\}$):** Nhãn giá trị được gán theo **kết cục cuối cùng của ván đấu** chứ không phải điểm đánh giá oracle tại thời điểm $t$. Ở giai đoạn khai/trung cuộc, một thế cờ cân bằng ($V^*(s) \approx 0$) có thể dẫn tới kết cục $+1$ hoặc $-1$ do sai lầm ở 30 nước sau. Do nhãn $z$ có phương sai nhiễu nội tại lớn ($\text{Var}(z) \approx 0.85$), **sai số MSE trên tập Validation/Test không thể tiến về $0$**, mà được kỳ vọng giảm có ý nghĩa thống kê so với mốc đối chứng dự đoán hằng số $0$ ($\text{MSE}_{\text{baseline}} \approx 0.80 \div 0.90$, mục tiêu thực tế $\text{MSE}_{\text{model}} < \text{MSE}_{\text{baseline}}$ rõ rệt và tương quan dương với đánh giá tĩnh).

---

## PHẦN 3: ĐẶC TẢ BIỂU DIỄN TENSOR 18 KÊNH GÓC NHÌN BÊN ĐI (`[18, 8, 8]`)

Theo **ADR-02** (`docs/DECISIONS.md`) và `configs/model_resnet.yaml`, mọi trạng thái bàn cờ được mã hóa thành Tensor $\mathbb{R}^{18 \times 8 \times 8}$ theo góc nhìn chuẩn hóa của bên tới lượt đi (`canonical_flip: true`):

### 3.1. Phép Lật Chuẩn Hóa Theo Lượt Đi
* Với mọi ô `sq` $\in [0, 63]$ trên bàn cờ FIDE:
  $$\text{sq}_{\text{canon}} = \begin{cases} \text{sq} & \text{nếu } \text{board.turn} == \text{WHITE} \\ \text{sq} \oplus 56 & \text{nếu } \text{board.turn} == \text{BLACK} \end{cases}$$
* Tọa độ ma trận: hàng $r = \text{sq}_{\text{canon}} // 8 \in [0, 7]$, cột $c = \text{sq}_{\text{canon}} \% 8 \in [0, 7]$. Bên tới lượt đi luôn ở phía dưới (hàng $0, 1$) và tiến Tốt lên phía trên (hàng $6, 7$).

### 3.2. Chi Tiết 18 Kênh Đặc Trưng (`shape = (18, 8, 8)`)

| Kênh | Nhóm Đặc Trưng | Đối Tượng Đại Diện | Giá Trị Lưu Đĩa (`uint8`) | Giá Trị Khi Đưa Vào Mạng (`float32`) |
| :---: | :--- | :--- | :---: | :---: |
| **0 – 5** | Quân bên đi (`Active`) | P (0), N (1), B (2), R (3), Q (4), K (5) | `0` hoặc `1` | `0.0` hoặc `1.0` |
| **6 – 11** | Quân đối thủ (`Opponent`) | P (6), N (7), B (8), R (9), Q (10), K (11) | `0` hoặc `1` | `0.0` hoặc `1.0` |
| **12** | Nhập thành (`Castling`) | Quyền nhập thành cánh Vua của bên đi (`Active_KS`) | Toàn mặt phẳng `0` / `1` | `0.0` hoặc `1.0` |
| **13** | Nhập thành (`Castling`) | Quyền nhập thành cánh Hậu của bên đi (`Active_QS`) | Toàn mặt phẳng `0` / `1` | `0.0` hoặc `1.0` |
| **14** | Nhập thành (`Castling`) | Quyền nhập thành cánh Vua của đối thủ (`Opp_KS`) | Toàn mặt phẳng `0` / `1` | `0.0` hoặc `1.0` |
| **15** | Nhập thành (`Castling`) | Quyền nhập thành cánh Hậu của đối thủ (`Opp_QS`) | Toàn mặt phẳng `0` / `1` | `0.0` hoặc `1.0` |
| **16** | Bắt Tốt qua đường (`EP`) | Ô đích bắt Tốt qua đường hợp lệ (đã lật chuẩn hóa) | `1` tại ô EP, `0` nơi khác | `0.0` hoặc `1.0` |
| **17** | Bộ đếm 50 nước (`Clock`) | Số plies không ăn quân / đi Tốt (`halfmove_clock`) | Số nguyên $\min(\text{clock}, 100)$ | Chia $100.0 \to [0.0, 1.0]$ |

> **Khắc phục triệt để lỗi mất dữ liệu:** Kênh 0–16 hoàn toàn là nhị phân $\{0, 1\}$, còn Kênh 17 lưu số nguyên $\min(\text{halfmove\_clock}, 100) \in [0, 100]$ trong kiểu `np.uint8` trên đĩa và chỉ thực hiện phép chia `/ 100.0` sau khi ép kiểu sang `torch.float32` trong `Dataset.__getitem__` hoặc trên GPU. Không có bất kỳ phép cắt cụt số thực nào xảy ra.

### 3.3. Mã Hóa Hành Động 4096 Chiều & Nhãn Giá Trị Tương Đối $z$
* **Chỉ số hành động phẳng (`action_idx`):**
  $$\text{action\_idx} = u_{\text{canon}} \times 64 + v_{\text{canon}} \in [0, 4095]$$
  Với cấu hình `drop_underpromotions: true`, các nước đi trong tập huấn luyện thuộc loại phong cấp dưới (Mã, Tượng, Xe — chiếm $< 0.5\%$) được bỏ qua khi trích xuất mẫu để tránh xung đột nhãn với nước phong Hậu trên cùng cặp $(u_{\text{canon}}, v_{\text{canon}})$.
* **Nhãn giá trị tương đối ($z$):**
  $$z = \begin{cases} R_{\text{abs}} & \text{nếu bên tới lượt đi là Trắng} \\ -R_{\text{abs}} & \text{nếu bên tới lượt đi là Đen} \end{cases} \in \{-1.0, 0.0, +1.0\}$$

---

## PHẦN 4: PHÂN CHIA CẤP ĐỘ VÁN ĐẤU TẤT ĐỊNH (DETERMINISTIC GAME-LEVEL SPLIT)

Để vừa xử lý dữ liệu theo luồng (Streaming / Sequential Pass) với bộ nhớ RAM thấp, vừa đảm bảo phân tách cấp độ ván đấu tuyệt đối không rò rỉ dữ liệu:
* **Quy tắc băm định danh ván đấu (`Game ID` / `Site` tag trong PGN):**
  ```python
  bucket = int(hashlib.md5(game_id.encode("utf-8")).hexdigest(), 16) % 100
  if bucket < 80:
      split = "train"      # 80% số ván (~40,000 ván -> ~800,000 mẫu)
  elif bucket < 90:
      split = "val"        # 10% số ván (~5,000 ván  -> ~100,000 mẫu)
  else:
      split = "test"       # 10% số ván (~5,000 ván  -> ~100,000 mẫu)
  ```
* Mọi thế cờ trích xuất từ cùng một ván đấu đều đi vào đúng một tập duy nhất (`train`, `val`, hoặc `test`). Tập `test` được đóng băng hoàn toàn, không bao giờ được sử dụng để chọn siêu tham số hay Early Stopping.

---

## PHẦN 5: CẤU TRÚC LƯU TRỮ NHỊ PHÂN (`NP.MEMMAP`) & QUY MÔ ĐĨA

### 5.1. Danh Mục Tệp Trong `data/processed/`
* `train_states.dat`, `val_states.dat`, `test_states.dat`
* `train_actions.dat`, `val_actions.dat`, `test_actions.dat`
* `train_values.dat`, `val_values.dat`, `test_values.dat`
* `dataset_metadata.json`: Lưu trữ thống kê số ván đã quét, số ván hợp lệ, số mẫu mỗi tập, số mẫu bị loại do trùng khóa Zobrist, thời gian chạy ETL (giây), và cấu hình băm.

### 5.2. Kích Thước Lưu Trữ Cho $1.000.000$ Mẫu

| Tệp Dữ Liệu | Kích Thước Chiều (Shape) | Dtype Trên Đĩa | Dung Lượng ($N = 10^6$ Mẫu) | Cơ Chế Chuyển Đổi Trong PyTorch |
| :--- | :--- | :--- | :--- | :--- |
| **States** | $(N, 18, 8, 8)$ | `np.uint8` | $\approx 1.152\text{ MB}$ | Chuyển sang `float32`; riêng kênh 17 chia `100.0` |
| **Actions** | $(N,)$ | `np.int16` | $\approx 2.0\text{ MB}$ | Chuyển sang `torch.int64` cho `CrossEntropyLoss` |
| **Values** | $(N,)$ | `np.float32` | $\approx 4.0\text{ MB}$ | Giữ nguyên `torch.float32` cho `MSELoss` |
| **Tổng Cộng** | — | — | **$\approx 1.158\text{ MB}$** | **RAM hệ thống luôn duy trì $< 1.5\text{ GB}$** |

---

## PHẦN 6: HẠ TẦNG HUẤN LUYỆN, HÀM MẤT MÁT & CHECKPOINT SCHEMA

### 6.1. Hàm Mất Mát & Tránh Điều Chuẩn L2 Kép
Theo `configs/model_resnet.yaml`:
* Sử dụng bộ tối ưu hóa **AdamW** (`lr = 0.001`, `weight_decay = 1e-4`).
* Vì `AdamW` đã thực hiện điều chuẩn trọng số tách biệt (Decoupled Weight Decay), hàm mất mát trong `chess_ai/models/loss.py` chỉ tính tổng có trọng số của hai đầu ra mà **không cộng lặp số hạng $\|\theta\|_2^2$** (`l2_weight = 0.0`):
  $$\mathcal{L}_{\text{total}} = w_{\text{policy}} \cdot \text{CrossEntropy}(\text{logits}, a^*) + w_{\text{value}} \cdot \text{MSE}(v, z)$$

### 6.2. Khai Thác Hỗn Hợp Độ Chính Xác (AMP) & Ghi Nhận Tài Nguyên
* Sử dụng cú pháp PyTorch 2.x chuẩn: `torch.amp.autocast("cuda", dtype=torch.float16)` kết hợp `torch.amp.GradScaler("cuda")`.
* Script `chess_ai/training/train_supervised.py` tự động ghi nhận vào `results/training/training_metrics.json`:
  * Cấu hình thiết bị thực tế (Tên CPU, số luồng, tên GPU, tổng VRAM).
  * Thời gian huấn luyện mỗi epoch và tổng thời gian huấn luyện (phút/giờ).
  * Đỉnh RAM và đỉnh VRAM tiêu thụ (`torch.cuda.max_memory_allocated()`).
  * Các chỉ số trên tập Train/Val/Test: `policy_loss`, `policy_top1_acc`, `policy_top3_acc`, `value_mse`, và `value_mse_vs_zero_baseline` (đối chiếu trực tiếp với baseline dự đoán $0$).

### 6.3. Cấu Trúc Đóng Gói Checkpoint Thống Nhất (`.pt`)
Tuân thủ nghiêm ngặt `checkpoint_schema` trong `configs/model_resnet.yaml` và `docs/HANDOFF_AND_QA.md`:

```python
checkpoint_payload = {
    "epoch": epoch,                                   # int: Epoch vừa hoàn thành
    "global_step": global_step,                       # int: Tổng số bước cập nhật gradient
    "model_state_dict": model.state_dict(),           # dict: Trọng số mạng Dual-Head ResNet
    "optimizer_state_dict": optimizer.state_dict(),   # dict: Trạng thái bộ tối ưu hóa AdamW
    "scheduler_state_dict": scheduler.state_dict(),   # dict: Trạng thái CosineAnnealingLR
    "scaler_state_dict": scaler.state_dict(),         # dict: Trạng thái GradScaler (AMP)
    "train_loss": float(train_loss),                  # float: Mất mát huấn luyện trung bình
    "val_policy_acc": float(val_policy_acc),          # float: Top-1 Policy Accuracy trên tập Val
    "val_value_mse": float(val_value_mse),            # float: Value MSE trên tập Val
    "best_val_loss": float(best_val_loss),            # float: Mất mát Validation tốt nhất
    "model_config": {
        "input_shape": [18, 8, 8],
        "canonical_flip": True,
        "num_res_blocks": 6,
        "filters": 64,
        "policy_conv_filters": 16,
        "value_conv_filters": 1,
        "value_hidden_units": 64,
        "action_space": 4096,
    },
    "random_seed": 42,
    "torch_version": torch.__version__,
}
```

* **Nạp trọng số an toàn tương thích PyTorch 2.x (`Device-Agnostic Loading`):**
  ```python
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  checkpoint = torch.load(filepath, map_location=device, weights_only=False)
  model.load_state_dict(checkpoint["model_state_dict"])
  ```

---

## PHẦN 7: HƯỚNG DẪN THỰC THI DÒNG LỆNH (CLI EXECUTION GUIDE)

### Bước 1: Trích Xuất Dữ Liệu Từ Lichess Elite Database
```bash
python -m chess_ai.etl.pgn_etl \
    --config configs/default_config.yaml \
    --input_path data/raw/lichess_elite_2025.pgn.zst \
    --output_dir data/processed
```

### Bước 2: Huấn Luyện Mạng Dual-Head ResNet Cấp 5 (Hỗ trợ `--resume`)
```bash
python -m chess_ai.training.train_supervised \
    --config configs/model_resnet.yaml \
    --data_dir data/processed \
    --checkpoint_dir checkpoints
```
Để khôi phục phiên huấn luyện khi bị gián đoạn:
```bash
python -m chess_ai.training.train_supervised \
    --config configs/model_resnet.yaml \
    --resume checkpoints/latest.pt
```
