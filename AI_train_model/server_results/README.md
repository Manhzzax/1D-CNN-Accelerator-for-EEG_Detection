# Hướng dẫn Lưu trữ & Đồng bộ Kết quả từ Server (Server Results Sync Guide)

Thư mục này được thiết kế để nhận, phân loại và quản lý tất cả kết quả thu được sau khi huấn luyện và lượng tử hóa mô hình trên máy chủ từ xa (Server). Để tránh ghi đè dữ liệu giữa các lần thử nghiệm khác nhau (thay đổi siêu tham số, đổi kiến trúc, thêm kênh, v.v.), tất cả kết quả phải được chia nhỏ thành các thư mục con riêng biệt đại diện cho từng lần chạy.

---

## 1. Cấu trúc lưu trữ khuyến nghị

Mỗi lần đồng bộ hóa kết quả của một phiên huấn luyện từ server về máy local, bạn nên tạo một thư mục con mới (ví dụ: `run_01_baseline`, `run_02_higher_epochs`) với cấu trúc như sau:

```
AI_train_model/server_results/
├── README.md                     # File hướng dẫn này
├── run_01_baseline/              # Ví dụ lần chạy số 1
│   ├── best_model.pth            # Trọng số mô hình tối ưu tải từ server
│   ├── config.yaml               # Bản sao cấu hình đã dùng trên server cho lần chạy này
│   ├── evaluation_report.txt     # Báo cáo độ chính xác trên tập Test
│   ├── quantization_report.txt   # Báo cáo sai số sau lượng tử hóa Q-Dynamic
│   ├── loss_accuracy_curves.png  # Biểu đồ Loss & Accuracy qua các Epochs
│   ├── confusion_matrix.png      # Biểu đồ ma trận nhầm lẫn
│   └── weights_q16/              # Thư mục chứa 10 file trọng số nguyên lượng tử hóa
│       ├── conv1_weight_q16
│       ├── conv1_bias_q16
│       ├── ...
│       └── fc3_bias_q16
└── run_02_experiment/            # Ví dụ lần chạy số 2
    └── ...
```

---

## 2. Hướng dẫn Lệnh Đồng bộ nhanh (Sync Commands)

Thực hiện chạy các lệnh sau trên terminal của **máy local** để tải kết quả từ server về (thay thế `<username>`, `<server_ip>`, `<port>`, và đường dẫn `/path/to/project/` bằng thông tin thực tế của máy chủ):

### Bước 1: Tạo thư mục cho lần chạy mới trên máy local
```bash
# Di chuyển tới thư mục chứa project local
cd /path/to/local/1D-CNN-Accelerator-for-EEG_Detection/

# Tạo thư mục cho lần chạy hiện tại
mkdir -p AI_train_model/server_results/run_01_baseline
```

### Bước 2: Đồng bộ dữ liệu bằng SCP hoặc Rsync

#### Lựa chọn A: Sử dụng lệnh SCP (Đơn giản nhất)
```bash
# 1. Tải toàn bộ file kết quả trong thư mục outputs/ của server về thư mục run tương ứng trên local
scp -P <port> -r <username>@<server_ip>:/path/to/project/AI_train_model/outputs/* AI_train_model/server_results/run_01_baseline/

# 2. Tải bản sao file config.yaml của lần chạy đó để lưu vết tham số
scp -P <port> <username>@<server_ip>:/path/to/project/AI_train_model/config/config.yaml AI_train_model/server_results/run_01_baseline/
```

#### Lựa chọn B: Sử dụng lệnh Rsync (Đồng bộ nhanh, an toàn và tối ưu băng thông)
```bash
# 1. Đồng bộ hóa thư mục outputs của server về local
rsync -avz -e "ssh -p <port>" <username>@<server_ip>:/path/to/project/AI_train_model/outputs/ AI_train_model/server_results/run_01_baseline/

# 2. Tải bản sao file config.yaml về local
scp -P <port> <username>@<server_ip>:/path/to/project/AI_train_model/config/config.yaml AI_train_model/server_results/run_01_baseline/
```

---

## 3. Danh sách kiểm tra đầu ra phần cứng (Hardware Weight Export Checklist)

Khi đồng bộ về local thành công, hãy kiểm tra chắc chắn thư mục `weights_q16/` bên trong thư mục chạy của bạn có đủ **10 tệp tin trọng số phẳng** sau để nạp cho ROM/RAM phần cứng tăng tốc (FPGA/ASIC):
- `conv1_weight_q16` & `conv1_bias_q16`
- `conv2_weight_q16` & `conv2_bias_q16`
- `fc1_weight_q16` & `fc1_bias_q16`
- `fc2_weight_q16` & `fc2_bias_q16`
- `fc3_weight_q16` & `fc3_bias_q16`
