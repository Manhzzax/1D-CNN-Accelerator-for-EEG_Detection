# EEG Seizure Detection 1D-CNN Accelerator Training Pipeline (CHB-MIT version)

Thư mục này chứa quy trình xử lý tín hiệu và huấn luyện mô hình 1D-CNN phân loại cơn động kinh sử dụng **CHB-MIT Scalp EEG Database (42.6 GB)** từ PhysioNet. Mô hình được tối ưu hóa phần cứng có cấu trúc cực kỳ gọn nhẹ (<100K parameters), hỗ trợ gập BatchNorm và xuất trọng số nguyên 16-bit cố định (Q-Dynamic format) sẵn sàng nạp cho phần cứng tăng tốc (FPGA/ASIC).

---

## 1. Cấu trúc thư mục (Project Structure)

```
AI_train_model/
├── config/
│   └── config.yaml          # Quản lý tất cả siêu tham số (Hyperparameters) và đường dẫn dữ liệu EDF
├── src/                     # Mã nguồn cốt lõi (Core Source Code)
│   ├── __init__.py
│   ├── preprocess_chbmit.py # Trích xuất và cắt tín hiệu liên tục từ các file .edf thành các phân đoạn 1s
│   ├── data_loader.py       # Tải các phân đoạn .npz đã tiền xử lý và chuẩn hóa kênh độc lập
│   ├── model.py             # Kiến trúc mạng 1D-CNN (~70K tham số, 23 kênh đầu vào, độ dài 256)
│   ├── utils.py             # Các hàm bổ trợ gập BatchNorm, lưu seed, vẽ biểu đồ
│   └── quantization.py      # Lượng tử hóa Q-bits động và xuất trọng số thành file text
├── scripts/                 # Các script thực thi riêng lẻ từng bước
│   ├── __init__.py
│   ├── run_preprocess.py    # Chạy cắt phân đoạn EDF & dán nhãn
│   ├── run_eda.py           # Phân tích tập dữ liệu phân đoạn & vẽ dạng sóng đa kênh
│   ├── run_train.py         # Huấn luyện mô hình & đánh giá tập Test
│   └── run_quantize.py      # Gập BatchNorm & Lượng tử hóa trọng số
├── main.py                  # Điểm truy cập chính điều phối toàn bộ pipeline
├── requirements.txt         # Danh sách thư viện Python phụ thuộc của project
└── README.md                # Tài liệu hướng dẫn sử dụng (File này)
```

---

## 2. Quản lý cấu hình (`config/config.yaml`)

Tất cả cấu hình được khai báo tập trung tại [config.yaml](file:///d:/Research/1D-CNN-Accelerator-for-EEG_Detection/AI_train_model/config/config.yaml):
- **`data.raw_dir`**: Đường dẫn tới thư mục chứa bộ dữ liệu CHB-MIT gốc của bạn (ví dụ: `D:/Research/chb-mit-scalp-eeg-database-1.0.0/`).
- **`model.input_channels`**: `23` (số lượng kênh điện cực chuẩn dùng của CHB-MIT).
- **`model.input_length`**: `256` (1 giây dữ liệu ghi nhận ở tần số lấy mẫu 256 Hz).

---

## 3. Hướng dẫn chạy chương trình (Execution Guide)

Bạn có thể thực hiện chạy tuần tự từng bước hoặc chạy toàn bộ thông qua file [main.py](file:///d:/Research/1D-CNN-Accelerator-for-EEG_Detection/AI_train_model/main.py):

### Cài đặt môi trường phụ thuộc
```bash
pip install -r requirements.txt
```

### Chạy toàn bộ quy trình (Preprocess -> EDA -> Train -> Quantize)
```bash
python main.py --mode all
```

### Chạy riêng lẻ từng bước
1. **Tiền xử lý & Cắt tín hiệu (Preprocessing):**
   ```bash
   python main.py --mode preprocess
   ```
   *Cơ chế hoạt động:* Script quét qua các thư mục bệnh nhân (ví dụ từ `chb01` đến `chb05`), đọc file tóm tắt annotation để tìm thời điểm bắt đầu và kết thúc cơn co giật. Sau đó, nó cắt các file `.edf` thành các phân đoạn 1 giây (256 mẫu) trên 23 kênh, dán nhãn 1 (Seizure) và 0 (Normal). Để giải quyết mất cân bằng lớp dữ liệu, script sẽ lấy ngẫu nhiên các phân đoạn Normal để cân bằng tỉ lệ 1:1 với Seizure. Dữ liệu nén sẽ lưu tại `data/chbmit_preprocessed.npz` (chỉ khoảng vài chục MB, rất nhẹ và tải cực nhanh).

2. **Phân tích dữ liệu (Exploratory Data Analysis):**
   ```bash
   python main.py --mode eda
   ```
   *Kết quả:* Đọc tập dữ liệu `.npz` đã tiền xử lý, in số lượng mẫu và lưu biểu đồ phân bố cũng như biểu đồ dạng sóng EEG đa kênh của 4 cảm biến tiêu biểu đại diện cho Seizure và Non-Seizure tại thư mục `outputs/`.

3. **Huấn luyện mô hình (Training):**
   ```bash
   python main.py --mode train
   ```
   *Kết quả:* Huấn luyện mô hình 1D-CNN (đầu vào $23 \times 256$), lưu file mô hình tối ưu (`best_model.pth`), vẽ biểu đồ độ chính xác/loss và lưu kết quả đánh giá ma trận nhầm lẫn tại thư mục `outputs/`.

4. **Lượng tử hóa và Xuất Trọng số:**
   ```bash
   python main.py --mode quantize
   ```
   *Kết quả:* 
   - Gập toán học `BatchNorm1d` vào `Conv1d`.
   - Tính toán số bit phần thập phân động (Dynamic Q-bits) cho từng layer độc lập dựa trên khoảng giá trị thực tế của chúng (ví dụ: Q12 cho `conv1` và Q15 cho các layer còn lại) để tránh hiện tượng bão hòa cắt cụm tín hiệu.
   - Kiểm tra chéo độ chính xác mô hình lượng tử hóa dạng số nguyên cố định so với số thực dấu phẩy động (độ lệch gần như bằng 0%).
   - Xuất các file trọng số thành file văn bản phẳng, mỗi số một dòng, sẵn sàng nạp cho phần cứng:
     - `conv1_weight_q16`, `conv1_bias_q16`
     - `conv2_weight_q16`, `conv2_bias_q16`
     - `fc1_weight_q16`, `fc1_bias_q16`
     - `fc2_weight_q16`, `fc2_bias_q16`
     - `fc3_weight_q16`, `fc3_bias_q16`
