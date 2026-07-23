# Deep Academic Research: EEG Data Analysis Methods for Seizure Detection

Báo cáo này được xây dựng cho **Phase 1 (Học thuật)** của dự án **Thiết kế Bộ tăng tốc phần cứng 1D-CNN cho Phát hiện Động kinh**. Báo cáo đi sâu nghiên cứu các phương pháp phân tích tín hiệu điện não đồ (EEG) truyền thống và lý giải tại sao mạng nơ-ron tích chập 1 chiều (1D-CNN) lại hiệu quả vượt trội khi tự động hóa việc trích xuất các đặc trưng này trực tiếp từ dữ liệu thô.

---

## 1. Bản chất Vật lý và Sinh học của Tín hiệu EEG

Điện não đồ (EEG) ghi lại các dao động điện áp cực nhỏ (quy mô microvolt $\mu\text{V}$) sinh ra từ dòng điện ion chạy trong các tế bào nơ-ron của vỏ não.
- **Tính chất tín hiệu:** EEG là tín hiệu chuỗi thời gian đa kênh (multi-channel time-series), phi tuyến, không dừng (non-stationary), có tỉ lệ tín hiệu trên nhiễu (SNR) cực kỳ thấp do bị lẫn nhiễu từ cơ (EMG), nhiễu chớp mắt (EOG), nhiễu tim (ECG) và nhiễu tần số dòng điện (50/60 Hz).
- **Đặc trưng của cơn co giật (Seizure/Ictal):** Trong cơn co giật động kinh, các tế bào thần kinh hoạt động đồng bộ hóa bất thường với cường độ rất mạnh. Trên biểu đồ EEG, hiện tượng này biểu hiện dưới dạng các sóng nhọn (spikes), sóng nhọn - sóng chậm kết hợp (spike-and-wave discharges) có biên độ lớn và tần số cao xuất hiện liên tục và nhịp nhàng.

---

## 2. Các Phương pháp Phân tích Dữ liệu EEG Truyền thống

Để phân tích tín hiệu EEG, giới học thuật và y học thường chia làm 4 miền phân tích chính:

### A. Phân tích Miền Thời gian (Time-Domain Analysis)
Miền thời gian tập trung vào việc nghiên cứu hình thái sóng trực tiếp theo chuỗi thời gian:
1. **Các tham số thống kê:** Mean (Trung bình), Variance (Phương sai - đặc trưng cho công suất tín hiệu), Skewness (Độ bất đối xứng), và Kurtosis (Độ nhọn - rất nhạy cảm với các đỉnh nhọn đột biến của cơn co giật).
2. **Tham số Hjorth (Hjorth Parameters):** Đây là tập hợp 3 chỉ số gọn nhẹ thường được dùng trong các thuật toán nhúng và phần cứng biên:
   - **Activity (Hoạt tính):** Đo lường tổng công suất của tín hiệu (chính là phương sai $\sigma^2_x$).
   - **Mobility (Độ linh động):** Ước lượng tần số trung bình của tín hiệu:
     $$\text{Mobility} = \sqrt{\frac{\text{Var}(dx/dt)}{\text{Var}(x)}}$$
   - **Complexity (Độ phức tạp):** Đo lường mức độ lệch của tín hiệu so với dạng sóng sin chuẩn (số lượng đỉnh sóng phụ):
     $$\text{Complexity} = \frac{\text{Mobility}(dx/dt)}{\text{Mobility}(x)}$$

### B. Phân tích Miền Tần số (Frequency-Domain / Spectral Analysis)
Nhiều nghiên cứu lâm sàng chỉ ra rằng hoạt động não bộ phân chia rõ rệt theo các băng tần số chuẩn (Brainwave Bands):
- **Delta ($\delta$: 0.5 - 4 Hz):** Xuất hiện khi ngủ sâu.
- **Theta ($\theta$: 4 - 8 Hz):** Xuất hiện khi buồn ngủ, thư giãn hoặc căng thẳng não bộ.
- **Alpha ($\alpha$: 8 - 12 Hz):** Xuất hiện khi nghỉ ngơi tĩnh tâm, nhắm mắt.
- **Beta ($\beta$: 12 - 30 Hz):** Xuất hiện khi tư duy logic tích cực, tập trung cao độ.
- **Gamma ($\gamma$: > 30 Hz):** Xuất hiện khi xử lý nhận thức cao. **Trong cơn co giật động kinh, năng lượng tần số cao thuộc băng Gamma và Beta tăng vọt một cách bất thường.**

**Phương pháp thực hiện:** Sử dụng phép biến đổi Fourier nhanh (FFT) hoặc thuật toán Welch để ước lượng **Mật độ phổ công suất (Power Spectral Density - PSD)** nhằm tính toán tỷ lệ năng lượng của các băng tần trên.

### C. Phân tích Miền Thời gian - Tần số (Time-Frequency Analysis)
Vì EEG là tín hiệu không dừng (tần số thay đổi liên tục theo thời gian), việc chỉ dùng biến đổi Fourier sẽ làm mất thông tin thời điểm xảy ra sự kiện. Do đó ta sử dụng:
1. **Short-Time Fourier Transform (STFT):** Chia tín hiệu thành các cửa sổ nhỏ xếp chồng lên nhau rồi áp dụng FFT để tạo ra biểu đồ Spectrogram. Tuy nhiên, STFT bị giới hạn bởi định lý bất định Heisenberg (độ phân giải thời gian tốt thì độ phân giải tần số kém và ngược lại).
2. **Discrete Wavelet Transform (DWT - Biến đổi Wavelet rời rạc):** Decompose tín hiệu EEG qua các bộ lọc thông thấp (approximation coefficients) và thông cao (detail coefficients) ở nhiều mức phân giải (multi-resolution). DWT cực kỳ hiệu quả để phát hiện các gai động kinh ngắn hạn vì nó có độ phân giải thời gian rất tốt ở tần số cao.

### D. Phân tích Miền Không gian (Spatial Analysis)
Não bộ là một hệ thống phân tán đa vùng. Việc kết hợp thông tin không gian giữa các điện cực (channels) giúp định vị nguồn phát động kinh:
- **Correlation Matrix (Ma trận tương quan):** Tính hệ số tương quan Pearson chéo giữa 23 kênh để đo lường mức độ đồng bộ hóa không gian giữa các vùng não.
- **Common Spatial Patterns (CSP):** Lọc không gian để tối đa hóa phương sai của một lớp (seizure) đồng thời tối thiểu hóa phương sai của lớp khác (normal), rất phổ biến trong các ứng dụng BCI.

---

## 3. Cơ chế Trích xuất Đặc trưng Tự động của Mạng 1D-CNN

Mạng 1D-CNN là giải pháp đột phá thay thế cho việc thiết kế các đặc trưng thủ công (hand-crafted features) phức tạp phía trên:

1. **Layer Conv1D hoạt động như Bộ lọc số học (Filter Banks):**
   Mỗi bộ lọc (kernel) của lớp tích chập 1 chiều trượt dọc theo chiều thời gian của tín hiệu. Về mặt toán học, phép tích chập này tương đương với một **Bộ lọc thông dải (Bandpass Filter)**. Trong quá trình huấn luyện bằng lan truyền ngược (backpropagation), các trọng số của kernel sẽ tự động hội tụ để lọc lấy các dải tần số đặc trưng nhất của cơn co giật (ví dụ tự học cách lọc băng Gamma hoặc các xung nhọn nhọn nhọn đầu).
2. **BatchNorm & Activation (ReLU):**
   Giúp phi tuyến hóa tín hiệu và ổn định gradient, giúp mô hình nhận diện được các ngưỡng biên độ kích hoạt (threshold) của dòng điện não.
3. **Lớp Max Pooling (Giảm chiều thời gian):**
   Bằng cách lấy giá trị cực đại trong một cửa sổ thời gian, Max Pooling giúp trích xuất biên độ đỉnh (peak amplitude) của sóng - đặc trưng cực kỳ quan trọng y học để phát hiện các gai nhọn động kinh, đồng thời tăng tính bất biến dịch chuyển (translation invariance) giúp mô hình nhận diện cơn giật cho dù nó xảy ra ở đầu hay cuối cửa sổ 1 giây.
4. **Các lớp Fully Connected (Tích hợp thông tin không gian):**
   Sau khi trích xuất đặc trưng thời gian qua các lớp Conv, các lớp Linear cuối cùng sẽ hoạt động như bộ trộn không gian chéo kênh (cross-channel spatial fusion), phân tích mối liên hệ phi tuyến giữa 23 vị trí điện cực trên da đầu để đưa ra kết luận phân loại cuối cùng.

---

## 4. Các Phương pháp Tiền Xử lý và Huấn luyện Tối ưu cho CHB-MIT

Để mô hình đạt độ chính xác (Accuracy) cao nhất khi đưa lên chạy thử nghiệm thực tế trên server, chúng ta triển khai các chiến thuật tiền xử lý dữ liệu sau:

1. **Phân đoạn Cửa sổ trượt chồng chập (Sliding Window with Overlap):**
   Thay vì cắt không chồng chập, chúng ta sử dụng cơ chế cửa sổ trượt dài 1 hoặc 2 giây với độ chồng chập (overlap) 50% hoặc 75% cho các phân đoạn Seizure. Điều này giúp **tăng cường dữ liệu (data augmentation)** gấp 2 đến 4 lần cho lớp thiểu số (Seizure), giải quyết tình trạng thiếu hụt dữ liệu Ictal.
2. **Chuẩn hóa Kênh Độc lập (Channel-wise Standardization):**
   Tránh chuẩn hóa toàn bộ dữ liệu chung. Do vị trí đặt điện cực khác nhau (ví dụ điện cực gần mắt chịu nhiễu EOG rất lớn, điện cực vùng chẩm biên độ khác biệt), ta bắt buộc phải tính toán trung bình và độ lệch chuẩn của **từng kênh riêng biệt** trên tập huấn luyện để áp dụng chuẩn hóa.
3. **Lọc số dải thông (Bandpass Filtering - 0.5 to 50 Hz):**
   Trước khi đưa vào mạng CNN, ta có thể áp dụng bộ lọc số FIR thông dải 0.5 - 50 Hz để loại bỏ hoàn toàn nhiễu trôi nền (drift tần số <0.5Hz) và nhiễu dòng điện 60Hz. Điều này giúp mạng CNN chỉ cần tập trung học các đặc trưng sinh học hữu ích.
