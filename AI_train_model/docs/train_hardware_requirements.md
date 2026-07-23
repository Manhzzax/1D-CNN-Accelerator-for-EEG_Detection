# Thông số Phần cứng Máy chủ phục vụ Huấn luyện Mô hình EEG 1D-CNN (CHB-MIT)

Tài liệu này xác định các thông số cấu hình phần cứng tối thiểu và khuyến nghị đối với máy chủ (GPU Server hoặc máy trạm Local Workstation) dùng để chạy tiền xử lý tập dữ liệu **CHB-MIT Scalp EEG Database (42.6 GB)** và huấn luyện mạng 1D-CNN.

---

## 1. Bộ xử lý đồ họa (GPU) & Bộ nhớ VRAM

Mạng 1D-CNN của chúng ta có kích thước rất nhỏ gọn (~70K tham số), do đó việc huấn luyện cực kỳ tiết kiệm tài nguyên GPU.

### Ước lượng dung lượng VRAM tiêu thụ
- **Bộ nhớ cho Trọng số Mô hình (Model Weights):**
  $$\text{Parameters} \approx 70,674 \text{ (float32)} \times 4 \text{ bytes} \approx 282.7 \text{ KB}$$
  (Chiếm dung lượng không đáng kể).
- **Bộ nhớ cho Kích hoạt (Forward Activations):**
  Với Batch Size = `128`, dữ liệu đầu vào có kích thước `(128, 23, 256)`. Tổng lượng kích hoạt lưu trữ trong bộ nhớ qua các lớp tích chập và fully connected khi truyền tiếp chỉ khoảng **2 MB** mỗi bước.
- **Tổng dung lượng VRAM thực tế tiêu thụ:**
  Khi chạy huấn luyện với PyTorch (bao gồm bộ nhớ đệm CUDA, Optimizer states của Adam, và Gradients), tổng dung lượng VRAM tiêu thụ thực tế chỉ khoảng **500 MB - 800 MB**.

### Khuyến nghị cấu hình GPU:
- **Cấu hình tối thiểu:** Bất kỳ GPU NVIDIA nào hỗ trợ CUDA với **VRAM từ 2 GB trở lên** (Ví dụ: NVIDIA GeForce GTX 1050, GTX 1650, hoặc card nhúng Jetson).
- **Cấu hình khuyến nghị:** NVIDIA RTX 3060/4060, Tesla T4, A10G, hoặc RTX 3080/4080 có hỗ trợ **Tensor Cores** để kích hoạt chế độ tự động tính toán số thực hỗn hợp **AMP (Automatic Mixed Precision - FP16)** nhằm đẩy nhanh tốc độ chạy.

---

## 2. Bộ vi xử lý (CPU)

Quá trình tiền xử lý dữ liệu EEG của CHB-MIT (đọc và phân tích tiêu đề nhiễu, cắt các đoạn sóng từ 664 file `.edf` lớn liên tục) là tác vụ **phụ thuộc rất lớn vào CPU (CPU-bound)**.

### Yêu cầu đối với CPU:
- **Tốc độ đơn nhân (Single-core speed):** Rất quan trọng cho việc giải mã cấu trúc file nhị phân EDF thông qua thư viện `mne` / `pyedflib`.
- **Số lượng nhân/luồng (Multi-threading):** Do quá trình xử lý có thể song song hóa theo từng thư mục bệnh nhân (sub-folders từ `chb01` đến `chb24`), CPU nhiều nhân sẽ rút ngắn thời gian xử lý từ vài tiếng xuống còn vài phút.
- **Cấu hình tối thiểu:** Intel Core i5 hoặc AMD Ryzen 5 thế hệ mới (6 nhân / 12 luồng).
- **Cấu hình khuyến nghị:** Intel Core i7/i9 hoặc AMD Ryzen 7/9 dòng Desktop, hoặc CPU máy chủ chuyên dụng Intel Xeon / AMD EPYC với **tối thiểu 8 nhân thực (16 luồng)**.

---

## 3. Bộ nhớ hệ thống (RAM)

Mặc dù bộ dữ liệu thô nặng 42.6 GB, cơ chế nạp dữ liệu của thư viện `mne` hỗ trợ đọc tệp dạng **Memory-mapping (lazy loading)** - nghĩa là chỉ nạp phân đoạn tín hiệu cần cắt vào RAM thay vì đọc toàn bộ file EDF vào bộ nhớ cùng lúc.
- Mã nguồn tiền xử lý của chúng ta giới hạn lượng RAM sử dụng tối đa tại một thời điểm dưới **4 GB**.
- Tuy nhiên, hệ điều hành cần RAM đệm lớn để tối ưu hóa bộ nhớ cache của file system (giúp tăng tốc độ đọc liên tục các file EDF).
- **Khuyến nghị dung lượng RAM:** **Tối thiểu 16 GB RAM** (Khuyến nghị **32 GB RAM** nếu chạy đa tiến trình song song trên server).

---

## 4. Thiết bị lưu trữ (Ổ cứng SSD) - Nút cổ chai I/O quan trọng nhất

Đây là yếu tố phần cứng **quyết định** đến thời gian tiền xử lý dữ liệu. Bộ dữ liệu gồm hàng trăm file EDF lớn cần đọc liên tục, việc sử dụng ổ cứng cơ học (HDD) sẽ dẫn đến nghẽn băng thông đọc/ghi dữ liệu (I/O Bottleneck), làm kéo dài thời gian tiền xử lý lên gấp nhiều lần.

### Yêu cầu về ổ đĩa:
- **Bắt buộc:** Phải sử dụng ổ cứng **SSD (Solid State Drive)**, khuyến nghị các dòng ổ cứng SSD chuẩn **NVMe M.2 (PCIe Gen 3 hoặc Gen 4)** có tốc độ đọc tuần tự từ 2500 MB/s trở lên.
- **Tránh sử dụng:** Ổ cứng HDD cơ học truyền thống hoặc ổ cứng mạng (Network Attached Storage - NAS) qua kết nối mạng chậm.
- **Dung lượng trống yêu cầu:** Tối thiểu **100 GB** dung lượng trống, bao gồm:
  - 43 GB chứa bộ dữ liệu thô CHB-MIT gốc.
  - 10 - 20 GB cho file dữ liệu phân đoạn nén (`chbmit_preprocessed.npz`).
  - Bộ nhớ đệm hệ thống và các file trọng số, biểu đồ đầu ra.
