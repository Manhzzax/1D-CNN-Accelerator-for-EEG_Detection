# Hướng dẫn Cấu hình và Tối ưu hóa Máy chủ Huấn luyện (Server Setup & Optimization Guide)

Tài liệu này hướng dẫn cách cấu hình và tối ưu hóa hệ thống máy chủ (GPU Server) để thực thi huấn luyện mô hình 1D-CNN trên bộ dữ liệu CHB-MIT đạt tốc độ cao nhất và tránh thắt nút cổ chai (bottlenecks).

---

## 1. Yêu cầu Hệ thống & Môi trường (Prerequisites)

### Hệ điều hành & Thư viện
- **OS:** Linux (Ubuntu 20.04 LTS hoặc mới hơn được khuyến nghị).
- **Driver GPU:** NVIDIA Driver hỗ trợ CUDA 11.8 hoặc CUDA 12.x.
- **Python:** Phiên bản từ 3.8 đến 3.11.

### Lệnh thiết lập môi trường ảo
```bash
# Tạo môi trường ảo venv
python3 -m venv .venv
source .venv/bin/activate

# Cập nhật pip và cài đặt dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

---

## 2. Tối ưu hóa nạp dữ liệu (PyTorch DataLoader Worker Tuning)

Nạp dữ liệu từ đĩa cứng (ổ đĩa SSD/HDD) lên CPU rồi đẩy sang GPU thường là nút cổ chai lớn nhất trong huấn luyện EEG do dung lượng file lớn.

Các tham số tối ưu hóa được đặt trong `config.yaml` dưới mục `training`:

### A. `num_workers` (Số tiến trình con nạp dữ liệu)
- **Cơ chế:** Khởi tạo các tiến trình con độc lập để đọc file và chuẩn hóa dữ liệu song song trước khi đưa vào hàng đợi huấn luyện.
- **Cách cấu hình:**
  - Thiết lập chuẩn: $\text{num\_workers} = 4 \times \text{Số lượng GPU}$.
  - Hoặc thiết lập dựa trên CPU: $\text{num\_workers} = \text{Số nhân CPU thực tế} - 2$.
  - *Lưu ý:* Đặt quá cao sẽ gây tràn RAM hệ thống hoặc nghẽn I/O đĩa cứng; đặt bằng 0 sẽ chạy đơn luồng trên CPU chính khiến GPU liên tục phải chờ dữ liệu.

### B. `pin_memory: true` (Ghim trang bộ nhớ RAM)
- **Cơ chế:** Khi `pin_memory=True`, PyTorch sẽ khóa các trang bộ nhớ chứa dữ liệu nạp trên RAM vật lý, ngăn hệ điều hành chuyển chúng vào swap. Điều này cho phép truyền dữ liệu trực tiếp bằng cơ chế DMA (Direct Memory Access) từ RAM máy chủ sang VRAM của GPU, giúp tăng đáng kể tốc độ nạp dữ liệu.
- **Khuyến nghị:** Luôn bật (`true`) khi huấn luyện bằng GPU. Tắt (`false`) khi chạy trên CPU.

---

## 3. Tối ưu hóa kích thước Batch (Batch Size) và VRAM GPU

Kích thước Batch ảnh hưởng trực tiếp tới tính ổn định của gradient và lượng bộ nhớ VRAM tiêu thụ.

- **Kích thước Batch khuyến nghị:** `64`, `128` hoặc `256`.
- **Ước lượng VRAM tiêu thụ của mô hình 1D-CNN (~70K params):**
  Do mô hình của chúng ta cực kỳ nhỏ gọn, lượng VRAM tiêu thụ cho trọng số mô hình chỉ khoảng vài trăm KB. Bộ nhớ chủ yếu tiêu thụ cho **kích hoạt chuyển tiếp (Forward Activations)**. Với kích thước batch `128` và dữ liệu $23 \times 256$, tổng lượng VRAM tiêu thụ cho một phiên huấn luyện chỉ dưới **1 GB**.
- **Nếu gặp lỗi Out-Of-Memory (OOM):** 
  Giảm `batch_size` xuống `64` hoặc `32` trong `config.yaml`.

---

## 4. Tự động huấn luyện với độ chính xác hỗn hợp (Automatic Mixed Precision - AMP)

AMP là cơ chế huấn luyện sử dụng đồng thời hai kiểu dữ liệu số thực: **Float16 (bán chính xác)** cho các phép toán tích chập/nhân ma trận để tối ưu hóa nhân Tensor Core của GPU, và **Float32 (chính xác đơn)** cho việc tính toán Loss và cập nhật trọng số nhằm đảm bảo mô hình không bị mất mát hội tụ do tràn số (overflow/underflow).

### Lợi ích:
- Giảm dung lượng VRAM tiêu thụ xuống khoảng **40-50%**.
- Tăng tốc độ huấn luyện trên GPU Tensor Cores gấp **1.5x - 2x**.

### Cấu hình trong `config.yaml`:
Set `use_amp: true`. Trong mã nguồn `scripts/run_train.py`, hệ thống tự động sử dụng `torch.cuda.amp.autocast()` và `torch.cuda.amp.GradScaler()` nếu máy chủ hỗ trợ CUDA.

---

## 5. Giới hạn luồng tính toán CPU (CPU Thread Control)

Mặc định, PyTorch sẽ sử dụng tối đa toàn bộ các luồng CPU khả dụng của máy chủ để tính toán các phép toán ma trận trong quá trình nạp/chuẩn hóa dữ liệu. Điều này có thể làm tê liệt hệ thống máy chủ dùng chung (shared server) và dẫn đến hiện tượng trễ tiến trình (thread thrashing).

### Cách xử lý:
Thiết lập `num_threads: 4` (hoặc giới hạn tùy ý) trong `config.yaml`. Trong code, PyTorch sẽ gọi lệnh:
```python
torch.set_num_threads(config['training']['num_threads'])
```
Lệnh này sẽ giới hạn số luồng tính toán song song trên CPU chính, giữ cho máy chủ hoạt động ổn định và nhường luồng cho các tiến trình khác.

---

## 6. Lệnh chạy nền trên Máy chủ Linux (Background Running)

Khi huấn luyện trên máy chủ từ xa qua SSH, nếu kết nối mạng bị ngắt, tiến trình huấn luyện sẽ bị hủy ngay lập tức. Để ngăn chặn điều này, hãy chạy nền sử dụng các phương pháp sau:

### Phương pháp 1: Sử dụng `nohup` (Đơn giản nhất)
```bash
# Chạy toàn bộ pipeline nạp nền và ghi log ra file train.log
nohup .venv/bin/python main.py --mode all > train.log 2>&1 &

# Kiểm tra tiến trình đang chạy
ps aux | grep main.py

# Theo dõi trực tiếp tiến độ ghi log
tail -f train.log
```

### Phương pháp 2: Sử dụng `tmux` (Khuyên dùng)
`tmux` tạo ra các terminal ảo độc lập hoạt động ngầm ngay cả khi ngắt kết nối SSH.
```bash
# Tạo một session mới tên là 'eeg_train'
tmux new -s eeg_train

# Kích hoạt venv và chạy lệnh train
source .venv/bin/activate
python main.py --mode all

# Thoát tạm thời khỏi session tmux (phím tắt):
# Nhấn giữ Ctrl + B, sau đó nhấn D

# Khi muốn quay lại kiểm tra tiến trình:
tmux attach -t eeg_train
```
