# Giới Hạn Thực Nghiệm Level 8 Self-Play ($8\times 8$) & Hướng Cải Tiến Tương Lai

**Tài liệu liên quan:** `results/training/selfplay_metrics.json`, `results/round_robin_200/summary_7x7_200.json`, `docs/DECISIONS.md` (`ADR-01`).

---

## 1. Hiện trạng thực nghiệm trên bàn cờ $8\times 8$ toàn phần

- **Kết quả ghi nhận:** Khi khởi tạo ấm (_warm-start_) từ `checkpoints/best_model.pt` (Level 5 đã huấn luyện trên **1.000.000 mẫu Lichess Elite ELO ~2580**) và chạy 6 vòng lặp Self-Play (`20 ván/vòng`, `100 sims/nước`, `batch_size=128`, `40 train_steps/vòng`), điểm Arena Gating của các bản challenger dao động ở mức **`35.0% – 50.0%`** (thấp hơn ngưỡng chấp nhận `55.0%`).
- **Hành vi bảo toàn của Arena Gating:** Toàn bộ 6 vòng đều bị từ chối (`accepted: false`), giữ nguyên `champion_version = 0`. Do đó `checkpoints/l8_alphazero_best.pt` có trọng số trùng khớp với `checkpoints/best_model.pt`, và trong bảng Round-Robin 7×7 (`round_robin_200`, cùng `100 sims`), `Level8_AlphaZero` đạt sức cờ tương đương `Level7_GuidedMCTS` (`74W-52D-74L`, `50.0%`).
- **Ý nghĩa học thuật:** Kết quả này chứng minh cổng kiểm duyệt **Arena Gating hoạt động đúng thiết kế** — tự động ngăn chặn hiện tượng suy thoái mô hình (_catastrophic forgetting_) khi dữ liệu self-play ngân sách nhỏ không vượt được baseline mạnh.

---

## 2. Nguyên nhân gốc rễ

1. **Chênh lệch chất lượng dữ liệu:** Baseline Level 5 học từ 1 triệu nước đi của kỳ thủ ~2580 ELO, trong khi self-play với `100 sims/nước` trên bàn cờ $8\times 8$ tạo ra nhãn policy $\pi_{\text{MCTS}}$ và kết cục $z$ có độ nhiễu cao hơn.
2. **Kích thước batch dữ liệu mỗi vòng nhỏ:** `20 ván/vòng` (~2.000 mẫu/vòng, tổng `12.534 mẫu` sau 6 vòng) chưa đủ đa dạng để cải thiện mạng đã hội tụ trên 1 triệu mẫu.
3. **Phương sai Arena Gating 20 ván lớn:** Với `20 ván`, khoảng tin cậy Wilson 95% quanh mức hòa 50% là `[29.9%, 70.1%]`, khiến việc vượt ổn định ngưỡng `>= 55%` rất khó nếu mức cải thiện thực nhỏ.

---

## 3. Lộ trình nâng cấp trong tương lai (Phương án C)

Nếu có thêm tài nguyên GPU để tiếp tục cải thiện Level 8 trên $8\times 8$ vượt trội rõ rệt so với Level 7/Level 5:

1. **Tăng chất lượng dữ liệu Self-Play:**
   - Nâng ngân sách tìm kiếm self-play từ `100 sims` lên **`400 – 800 sims/nước`** (`mcts_sims_per_move >= 400`).
   - Tăng số ván mỗi vòng lặp từ `20` lên **`100 – 200 ván/iteration`** (tạo ~10.000–20.000 mẫu mới/vòng) và mở rộng `replay_buffer_size` lên `200.000+` với cửa sổ trượt (_sliding window_ giữ các vòng gần nhất).
2. **Tinh chỉnh siêu tham số Fine-Tuning:**
   - Hạ learning rate từ `5e-4` / `1e-3` xuống **`1e-4` (hoặc `5e-5`)** kèm warmup ngắn và `weight_decay = 1e-4` để tránh phá vỡ đặc trưng đã học từ 1M mẫu Lichess.
   - Cân nhắc trộn một phần dữ liệu Supervised (replay mixing ~10–20%) hoặc chỉ fine-tune các tầng head ở những vòng đầu.
3. **Ổn định hóa Arena Gating:**
   - Tăng số ván kiểm định Arena từ `20` lên **`50 – 100 ván`** (`25–50 khai cuộc × 2 màu`) bằng `multiprocessing` / GPU batching để giảm phương sai thống kê.
   - Chỉ cập nhật `round_robin_200` (6 cặp của Level 8) sau khi thu được `champion_version >= 1`.
