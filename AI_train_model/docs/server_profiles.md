# Hồ sơ Cấu hình 3 Máy chủ GPU Lab (Server Profiles)

Tài liệu này tổng hợp thông số kỹ thuật chi tiết của 3 máy chủ GPU có sẵn và cung cấp cấu hình tối ưu PyTorch tương ứng để thiết lập trong `config.yaml` cho từng máy.

---

## 1. Bảng so sánh tổng quan phần cứng

| Máy chủ / Server | Trạng thái | CPU | RAM | GPU | VRAM | Vai trò |
|---|---|---|---|---|---:|---|
| **Server `.9`** (`CPU-FPGA-GPU`) | **Hoạt động chính (Active)** | Intel Core i7-8700K (6 Cores / 12 Threads) | ~32 GB | NVIDIA GeForce RTX 3090 | **24 GB** | Huấn luyện & Kiểm thử chính |
| **Server `.13`** (3090 Server) | **Dự phòng (Standby)** | CPU model đa nhân (~28 Threads) | ~251 GB | NVIDIA GeForce RTX 3090 | **24 GB** | Chạy các job tải dữ liệu cực lớn |
| **Server `SERVER-02`** (RTX 8000) | **Dự phòng (Standby)** | CPU model đa nhân | ~188 GB | NVIDIA Quadro RTX 8000 | **48 GB** | Phục vụ tác vụ cần VRAM lớn |

---

## 2. Thông số chi tiết & Cấu hình PyTorch tối ưu

### A. Server `.9` (`CPU-FPGA-GPU`) - Máy chủ Active chính
Máy chủ tích hợp card RTX 3090 mạnh mẽ nhưng bộ nhớ RAM và số nhân CPU ở mức vừa phải. Cần giới hạn tài nguyên hợp lý để tránh quá tải RAM hệ thống.

- **Thông số chi tiết:**
  - **Hostname:** `CPU-FPGA-GPU`
  - **CPU:** Intel Core i7-8700K @ 3.70 GHz (6 Cores / 12 Threads)
  - **RAM:** ~32 GB (31 GiB khả dụng cho Linux)
  - **GPU:** 1 × NVIDIA GeForce RTX 3090 (24 GB GDDR6X VRAM, 10,496 CUDA Cores)
  - **Hệ điều hành:** Ubuntu Linux
- **Khuyến nghị cấu hình `config.yaml`:**
  ```yaml
  training:
    batch_size: 128
    num_workers: 4         # i7-8700K có 12 luồng, đặt 4 workers để tránh quá tải luồng CPU
    pin_memory: true       # RTX 3090 cần pin_memory để đẩy nhanh tốc độ truyền DMA
    use_amp: true          # Bật AMP (RTX 3090 hỗ trợ Tensor Cores tối ưu cực tốt cho FP16)
    num_threads: 4         # Khống chế luồng tính toán CPU chính
  ```

---

### B. Server `.13` - Máy chủ RTX 3090 (Dự phòng)
Máy chủ có bộ nhớ RAM hệ thống cực lớn (251 GB) và CPU nhiều nhân. Rất thích hợp cho việc tiền xử lý song song tốc độ cao (EDF slicing) và các batch size trung bình lớn.

- **Thông số chi tiết:**
  - **CPU:** Đa nhân (~28 logical CPUs)
  - **RAM:** ~251 GB
  - **GPU:** 1 × NVIDIA GeForce RTX 3090 (24 GB GDDR6X VRAM, 10,496 CUDA Cores)
- **Khuyến nghị cấu hình `config.yaml`:**
  ```yaml
  training:
    batch_size: 128
    num_workers: 8         # RAM lớn (251G) và 28 luồng CPU cho phép tăng workers lên 8 để nạp cực nhanh
    pin_memory: true
    use_amp: true          # Bật AMP
    num_threads: 8         # Có thể tăng số luồng CPU xử lý lên 8 luồng
  ```

---

### C. Server `SERVER-02` - Máy chủ Quadro RTX 8000 (Dự phòng VRAM lớn)
Máy chủ sở hữu dòng card đồ họa chuyên nghiệp Quadro RTX 8000 có dung lượng VRAM khổng lồ (48 GB). Rất thích hợp khi muốn tăng kích thước Batch Size cực lớn hoặc chạy các mô hình lớn phức tạp hơn mà không lo bị tràn VRAM (OOM).

- **Thông số chi tiết:**
  - **CPU:** Đa nhân (multi-core)
  - **RAM:** ~188 GB
  - **GPU:** 1 × NVIDIA Quadro RTX 8000 (48 GB GDDR6 VRAM, 4,608 CUDA Cores)
- **Khuyến nghị cấu hình `config.yaml`:**
  ```yaml
  training:
    batch_size: 256        # VRAM 48GB cho phép tăng gấp đôi batch_size lên 256 để tăng tốc độ huấn luyện song song
    num_workers: 8         # Đặt 8 workers nhờ bộ nhớ RAM lớn (188G)
    pin_memory: true
    use_amp: true          # Card Turing Quadro RTX 8000 hỗ trợ tốt AMP
    num_threads: 6
  ```
