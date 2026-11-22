# QUY TRÌNH GITFLOW & QUY ĐỊNH LÀM VIỆC NHÓM (GITFLOW.md)

**Dự án:** Multi-Level Chess AI Engine ($8 \times 8$ FIDE)  
**Môn học:** Trí tuệ Nhân tạo Nâng cao  
**Tài liệu liên quan:** [`docs/CONTRIBUTIONS.md`](CONTRIBUTIONS.md), [`docs/HANDOFF_AND_QA.md`](HANDOFF_AND_QA.md)

---

## 1. Cấu Trúc Nhánh (Branching Model)

* **Nhánh ổn định nộp bài (`main`):**
  * Chỉ chứa mã nguồn và kết quả thực nghiệm đã kiểm chứng sạch.
  * **Nghiêm cấm** `git push --force` hoặc commit trực tiếp các thay đổi lớn chưa qua kiểm tra lên `main`.
* **Nhánh tích hợp nhóm (`dev`):**
  * Nơi gộp code từ các nhánh thành viên để kiểm tra tích hợp liên phân hệ trước khi đưa lên `main`.
* **Nhánh phát triển cá nhân (`feature/*`, `fix/*`, `docs/*`):**
  * `feature/tv1-core-search-rl` _(TV1)_: Core (`ChessEnv`, `evaluation`, `zobrist`), Tìm kiếm cổ điển (`Alpha-Beta + ID`, `TT`, `Move Ordering`, `Quiescence`, `UCT`), Tác tử **Cấp 1 – Cấp 4**, Cấp 6 (KRK) & Ablation 1.
  * `feature/tv2-data-models-puct` _(TV2)_: `board_tensor`, ETL Lichess Elite, Dual-Head ResNet, PUCT và Tác tử Cấp 5, 7, 8.
  * `feature/tv3-ui-arena-qa` _(TV3)_: UI Pygame, Arena Tournament, Ablation 2, bộ kiểm thử `tests/`, `reproduce_experiment.py`, `notebooks/analysis.ipynb`.
  * Có thể tách nhánh phụ ngắn hạn (ví dụ `feature/<tên_tính_năng>`, `fix/<lỗi>`, `docs/<tài_liệu>`) từ `dev` hoặc `main`, xóa nhánh sau khi merge xong.

---

## 2. Quy Chuẩn Thông Điệp Commit (Conventional Commits)

Mọi commit phải tuân theo định dạng chuẩn để dễ truy vết lịch sử đóng góp khi chấm bài (Mục 3.7 đề bài):

```text
<type>(<scope>): <mô tả ngắn gọn bằng tiếng Anh hoặc tiếng Việt không dấu>
```

* **Các `<type>` quy ước:**
  * `feat`: Thêm tính năng hoặc module mới (ví dụ: `feat(search): implement Quiescence Search with Delta Pruning`)
  * `fix`: Sửa lỗi (ví dụ: `fix(etl): use LichessURL header for deterministic split`)
  * `docs`: Cập nhật tài liệu, README, đặc tả (ví dụ: `docs(ui): add UI_SPEC.md for desktop interface`)
  * `test`: Bổ sung hoặc sửa unit test (ví dụ: `test(ui): add headless tests for ChessPygameApp`)
  * `refactor`: Tối ưu/tái cấu trúc code không đổi hành vi
  * `chore`: Cấu hình build, dependencies, `.gitignore`
* **Nguyên tắc chia nhỏ commit:** Chia theo từng cụm file cùng chức năng; không gộp hàng loạt thay đổi không liên quan vào một commit chung chung như `"update"` hay `"fix"`.

---

## 3. Quy Định Pull Request (PR) & Kiểm Tra Trước Khi Merge

1. **Chạy kiểm thử bắt buộc trước khi push/tạo PR:**
   ```bash
   pytest -v
   ```
   Tất cả bài kiểm thử hiện có phải đạt **100% PASS**. Không tạo PR khi test đang lỗi.
2. **Kiểm tra trạng thái tệp trước khi `git add`:**
   * Chạy `git status -sb` để rà soát đúng các tệp thuộc phạm vi công việc; tránh dùng `git add .` bừa bãi làm lọt tệp tạm hoặc dữ liệu lớn.
3. **Review chéo & Tôn trọng ranh giới module (RACI):**
   * PR vào `dev`/`main` nên có ít nhất **1 thành viên khác review** trước khi merge.
   * Không tự ý sửa đổi mã nguồn thuộc module của thành viên khác nếu chưa trao đổi thống nhất (trừ trường hợp sửa lỗi tích hợp khẩn cấp và có ghi chú rõ trong PR).
4. **Giải quyết xung đột (Merge Conflict):**
   * Kéo `main`/`dev` mới nhất về nhánh của mình (`git pull --rebase` hoặc `git merge`), chạy lại `pytest -v` xác nhận xanh toàn bộ rồi mới push.

---

## 4. Quy Định Quản Lý Dữ Liệu Lớn, Môi Trường & `.gitignore`

* **Nghiêm cấm commit các tệp sau lên GitHub:**
  * Môi trường ảo cá nhân: `.venv/`, `venv/`, `env/`.
  * Dữ liệu huấn luyện thô & đã xử lý: `data/raw/*`, `data/processed/*` (vì tệp `.pgn` và `.dat` nặng hàng trăm MB đến hàng GB, vượt giới hạn `100 MB/file` của GitHub).
  * Các checkpoint trung gian (`checkpoints/latest.pt`, `checkpoints/model_epoch_*.pt`), tệp nén tạm (`*.rar`, `*.zip` dữ liệu) và log chạy thử (`logs/*.log`).
* **Các tệp ĐƯỢC PHÉP & CẦN commit để nộp bài:**
  * Mã nguồn `chess_ai/`, `scripts/`, `tests/`, `configs/`, `resources/` (`benchmark_positions.epd`, `openings_50.epd`), `docs/`, `notebooks/`, `report/`.
  * Trọng số cuối cùng: `checkpoints/best_model.pt` ($\sim 5\text{MB}$), `checkpoints/l6_krk_qtable.pkl`.
  * Kết quả thực nghiệm & biểu đồ: `results/` (`matches.jsonl`, `games.pgn`, `ablation1/`, `ablation2/`, `figures/`).
* **Chia sẻ dữ liệu lớn trong nhóm:** Dùng **Google Drive / OneDrive** hoặc **GitHub Releases** (không đẩy vào lịch sử commit).
* **Đồng bộ thư viện:** Khi cài thêm thư viện mới, phải cập nhật đồng bộ cả `requirements.txt` và `pyproject.toml`.

---

## 5. Quy Trình Làm Việc Hàng Ngày (Checklist Nhanh)

```bash
# 1. Cập nhật code mới nhất trước khi làm việc
git checkout <nhanh_cua_ban>
git pull origin main

# 2. Code + kiểm tra test xanh
pytest -v

# 3. Kiểm tra file thay đổi, commit theo từng phần nhỏ và push
git status -sb
git add <cac_tep_lien_quan>
git commit -m "feat(<scope>): <mo ta>"
git push origin <nhanh_cua_ban>
```
