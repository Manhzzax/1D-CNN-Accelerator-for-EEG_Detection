# Câu chuyện thực nghiệm CHB-MIT để trao đổi với giáo sư

## 1. Tóm tắt một phút

Đề tài là phát hiện cơn động kinh **liên tục** trên EEG CHB-MIT bằng một mạng
1D-CNN gọn để sau này triển khai trên KV260. Mô hình tham chiếu hiện nay được
gọi là **EpiSepNet-5K** (Epileptic Seizure depthwise-Separable Network, mạng
phát hiện động kinh dùng tích chập tách rời) với **5,013 tham số học được**.

Kết quả cửa sổ tốt nhất hiện tại là **90.07% validation window accuracy**
(độ chính xác phân loại từng cửa sổ trên tập validation), **90.76% seizure
sensitivity** (độ nhạy cửa sổ ictal), AUROC 96.58% và F1 90.14%. Kết quả này
thuộc thực nghiệm `run_21_raw_2s_temporal3`, dùng EEG thô 17 kênh, cửa sổ 2 s,
bước trượt 1 s và chuẩn hóa z-score theo kênh chỉ từ train.

Tuy nhiên, đây **chưa phải kết quả lâm sàng cuối cùng**. Khi chạy liên tục theo
thời gian thực và dùng policy (quy tắc biến các điểm số cửa sổ thành một báo
động), điểm tốt nhất trên validation đạt **23/29 cơn = 79.31% event
sensitivity** (độ nhạy theo cơn), **0.4671 FAR/h** (False Alarm Rate per hour,
số báo động giả mỗi giờ) và độ trễ trung vị 17 s. Mục tiêu nội bộ đang dùng là
event sensitivity >= 90%, FAR/h <= 0.5 và median delay <= 10 s. Vì thế ta có
một mô hình cửa sổ tốt và rất gọn, nhưng vẫn cần tối ưu chất lượng báo động
liên tục.

## 2. Quy trình dữ liệu đã được khóa

1. Tải và kiểm tra CHB-MIT: 686 EDF, checksum SHA-256 đúng, 198 cơn được
   khai báo/parse, không lỗi header và 17 kênh bipolar chuẩn có mặt ở 686/686
   bản ghi.
2. Chuẩn hóa đầu vào về 17 kênh bipolar chung, lấy mẫu 256 Hz.
3. Chia dữ liệu theo từng bệnh nhân và thời gian: train 399 bản ghi/107 cơn,
   validation 95/29, test 192/62. Đây là **within-case chronological split**
   (cùng bệnh nhân có thể xuất hiện ở train/validation/test, nhưng các đoạn ghi
   được sắp theo thời gian), không phải **patient-independent split** (chia
   bệnh nhân hoàn toàn độc lập).
4. Mọi tham số kiến trúc, threshold (ngưỡng xác suất) và policy đều phải chọn
   trên validation. Test chỉ nên dùng một lần sau khi chốt mô hình. Do test đã
   được xem ở các thực nghiệm thăm dò đầu, các số test lịch sử chỉ dùng để chẩn
   đoán, không dùng làm bằng chứng cuối cho bài báo.

## 3. Vì sao không chỉ dùng Accuracy

**Window accuracy** (tỷ lệ cửa sổ được phân loại đúng) trả lời câu hỏi: trong
một cửa sổ EEG đã cắt sẵn, mô hình có gán đúng nhãn không? Các bài báo CHB-MIT
thường báo chỉ số này, nhưng mỗi bài có độ dài cửa sổ, tỷ lệ lớp, số bệnh nhân
và cách chia khác nhau, nên không thể coi là một chuẩn lâm sàng duy nhất.

Ứng dụng detector cần đánh giá **continuous event-level evaluation** (đánh giá
liên tục theo từng cơn):

- **Event sensitivity** (độ nhạy theo cơn): một cơn được tính đúng nếu có ít
  nhất một báo động hợp lệ trong khoảng cơn.
- **FAR/h** (False Alarm Rate per hour, số cảnh báo sai/giờ): số báo động trong
  khoảng không có cơn chia cho tổng thời lượng interictal.
- **Detection delay** (độ trễ phát hiện): thời gian từ onset của cơn tới báo
  động đầu tiên.
- **Policy** (chính sách ra cảnh báo): quy tắc hậu xử lý điểm cửa sổ, ví dụ
  `10_of_20` nghĩa là cần tối thiểu 10 cửa sổ dương tính trong 20 cửa sổ gần
  nhất mới tạo một cảnh báo.
- **Threshold** (ngưỡng): xác suất ictal tối thiểu để một cửa sổ được tính là
  dương tính trước khi áp dụng policy.

Vì một tập test có rất nhiều cửa sổ không cơn, accuracy cao vẫn có thể đi kèm
với FAR/h cao hoặc bỏ sót cơn. Do đó hướng Q1 của đề tài là báo song song cả
window-level và event-level, ưu tiên event sensitivity/FAR/h/delay khi chọn
detector.

## 4. Câu chuyện 21 thực nghiệm

### Giai đoạn 1: xác lập baseline và phát hiện vấn đề báo động giả

**Run 01 - baseline 1D-CNN.** Mô hình đầu tiên cho event sensitivity test rất
cao, 60/62 = 96.77%, nhưng FAR/h lên tới 41.26. Bài học là classifier nhận ra
nhiều cửa sổ ictal, nhưng không đủ ổn định để chạy liên tục; không được dùng
accuracy cao để tuyên bố detector lâm sàng.

**Run 02 - hard negative 5:1.** Ta thử hard negative mining (khai thác các cửa
sổ interictal mà mô hình cũ dễ nhầm thành ictal) với tỷ lệ quá mạnh 5:1. Mô
hình bị **distribution shift** (phân bố train lệch quá xa dữ liệu vận hành),
test accuracy rơi còn 26.68%, AUROC 0.343. Kết luận: không thể thay phần lớn
normal bằng hard negative.

**Run 03 - mixed hard negative.** Ta giữ normal ban đầu và chỉ bổ sung hard
negative, tạo tập trộn. FAR/h giảm rất mạnh, nhưng event sensitivity cũng giảm
còn 36/62 trên test lịch sử. Đây là trade-off (đánh đổi): giảm báo động giả
quá mức có thể khiến detector bỏ sót cơn.

### Giai đoạn 2: thử xử lý theo thời gian và hard negative bền vững

**Run 04 - score TCN.** Ta ghép một causal TCN (Temporal Convolutional Network,
mạng tích chập theo thời gian nhân quả) phía sau chuỗi điểm số của Run 03. TCN
không nhìn vào tương lai nên phù hợp hướng online. Kết quả lịch sử không đủ tốt
để giữ: chạy trên điểm số đã mất thông tin EEG, event sensitivity 40/62 và
FAR/h 0.422. Ý tưởng TCN có cơ sở từ mạng nhân quả dùng cho chuỗi thời gian,
nhưng cách đặt sau score CNN không hiệu quả bằng cải thiện backbone EEG.

**Run 05 - persistent temporal hard negative.** Chỉ lấy negative tạo chuỗi
điểm cao liên tiếp và gán trọng số lấy mẫu cao hơn. Dữ liệu đủ điều kiện chỉ có
474 đoạn độc lập, nên coverage (độ phủ mẫu khó) thấp; kết quả lịch sử giảm còn
18/62 cơn và FAR/h 1.232. Kết luận: hard negative quá hiếm không được phép
quyết định phân bố train.

### Giai đoạn 3: chuyển từ CNN thường sang backbone nhỏ phù hợp phần cứng

**Run 06 - raw separable 1D-CNN.** Đây là bước chuyển kiến trúc chính. Thay
CNN thường bằng depthwise-separable convolution (tích chập theo từng kênh rồi
trộn kênh bằng pointwise convolution), lấy cảm hứng từ EEGNet. Cách này tách
temporal filtering (lọc mẫu dạng sóng theo thời gian) khỏi spatial mixing (trộn
thông tin giữa điện cực), giảm mạnh tham số và vẫn giữ EEG thô. Validation đạt
20/29 cơn, FAR/h 0.219; nhưng test thăm dò có nhiều cụm báo động ở chb04/chb05.

**Run 07 - per-record z-score.** Ta thử chuẩn hóa từng bản ghi riêng. Kết quả
xấu hơn vì mô hình không còn thấy được khác biệt biên độ có ý nghĩa giữa các
bản ghi, đồng thời chuẩn hóa theo cả bản ghi là khó bảo đảm nhân quả nếu triển
khai trực tuyến. Ta giữ **train-only channel-wise z-score** (trung bình và độ
lệch chuẩn mỗi kênh chỉ fit trên train) làm chuẩn.

**Run 08 - parallel multikernel CNN.** Ta thử nhiều kernel song song để bắt
nhịp EEG ngắn và dài. Mặc dù sensitivity rất cao, FAR/h cực lớn, vì các nhánh
đa tỉ lệ tăng kích hoạt false positive. Đây là lý do không chọn mô hình chỉ
theo sensitivity.

**Run 09.** Không có artifact thực nghiệm; đây chỉ là số ID dự phòng trong
script, không được trình bày là một run nghiên cứu.

### Giai đoạn 4: tối ưu có kiểm soát, không dùng test để chọn

**Run 10 - hyperparameter sweep.** Ta cố định raw separable backbone rồi thay
từng hyperparameter (siêu tham số) gồm learning rate, weight decay và
class-balanced sampling (lấy mẫu cân bằng lớp). Early stopping (dừng sớm khi
validation loss không còn cải thiện) dùng để tránh overfitting và lãng phí GPU.
Các ứng viên tốt đạt 21/29 cơn với FAR/h khoảng 0.40-0.46. Cần audit lại
selection trace (dấu vết chọn mô hình) trước khi viết paper, vì hai trial hòa
sensitivity nhưng tài liệu lịch sử ghi trial tham chiếu chưa phản ánh hoàn
toàn quy tắc tie-break (phá hòa) FAR thấp hơn.

**Run 11 - temporal policy sweep.** Không train lại model; chỉ quét threshold
và policy của Run 10 trên validation. Việc tách policy khỏi train cho phép biết
cải thiện do classifier hay do hậu xử lý.

**Run 12 - DWT + separable CNN.** DWT (Discrete Wavelet Transform, biến đổi
wavelet rời rạc) được thử vì nhiều bài CHB-MIT dùng phân giải thời gian-tần số.
Kết quả validation window accuracy 83.87%, event 14/29, delay 23 s: kém raw
EEG. Kết luận thực nghiệm: với backbone nhỏ hiện tại, DWT coefficient
concatenation (nối các hệ số wavelet) không mang lại lợi ích đủ để đổi lấy chi
phí tiền xử lý phần cứng.

**Run 13 - separable refine.** Bắt đầu từ Run 10, chỉ thay một yếu tố mỗi lần.
Tăng temporal filters/channel (số bộ lọc thời gian cho mỗi kênh) lên 3 là ứng
viên kiến trúc tốt nhất: 21/29 cơn, FAR/h 0.349, median delay 16 s. Đây là cơ
sở của EpiSepNet-5K.

**Run 14 - policy refinement.** Quét policy mịn cho Run 13, tìm `7_of_14`,
threshold 0.977: 21/29 cơn, FAR/h 0.449, delay 13 s. Không có huấn luyện ở run
này; đây chỉ là chọn quy tắc ra quyết định.

### Giai đoạn 5: kiểm tra lại giả thuyết mining và multi-scale

**Run 15 - policy-aligned hard negative.** Ta mining negative đúng kiểu false
alarm mà `7_of_14` tạo ra. Tuy nhiên chỉ có 113 candidate phù hợp; policy bị
quá bảo thủ, còn 19/29 cơn dù FAR/h xuống 0.106.

**Run 16 - balanced control.** Đây là control (thực nghiệm đối chứng) không
mining, cùng temporal-3 backbone. Nó đạt 19/29 nhưng FAR/h 0.633. So sánh
Run 15/16 xác nhận mining có giảm FAR, nhưng trả giá bằng sensitivity; không
phải hướng để đạt clinical gate.

**Run 17 - stricter mining.** Điều kiện mining nghiêm ngặt hơn cho 109 mẫu,
FAR/h 0.041 nhưng chỉ 15/29 cơn. Kết luận mạnh hơn: tối ưu mining không thể
thay cho nâng khả năng phân biệt tín hiệu.

**Run 18 - multiscale separable.** Đưa đa tỉ lệ trở lại nhưng theo separable
CNN gọn với kernel 15/63 mẫu. Kết quả tốt nhất chỉ 20/29 cơn, không vượt
temporal-3 về sensitivity, nên không chọn làm reference (mốc tham chiếu).

**Run 19 - multiscale policy sweep.** Chỉ quét policy cho Run 18; điểm tốt là
20/29, FAR/h 0.443, delay 14 s. Nó củng cố quyết định không thay backbone
temporal-3.

### Giai đoạn 6: sửa tính nhân quả và nâng chất lượng cửa sổ

**Run 20 - causal re-evaluation.** Ta sửa timestamp cảnh báo thành
`window_end_causal` (thời điểm kết thúc cửa sổ, tránh ngầm dùng tương lai). Với
Run 13, kết quả là 21/29, FAR/h 0.455, delay 14 s. Đây là mốc 1 s nhân quả hợp
lệ hơn các con số cũ.

**Run 21 - EpiSepNet-5K, cửa sổ 2 s.** Ta giữ separable temporal-3 và thử cửa
sổ 2 s, bước trượt 1 s. Kết quả validation window tốt nhất hiện có: accuracy
90.07%, sensitivity 90.76%, AUROC 96.58%, AP 96.98%, F1 90.14%. Với policy
`10_of_20` và threshold 0.975: 23/29 cơn, FAR/h 0.4671, median delay 17 s.
Đây là **reference cho chất lượng cửa sổ và footprint phần cứng**, nhưng chưa
qua clinical gate vì event sensitivity và delay.

## 5. Các kỹ thuật đã áp dụng và cơ sở

| Kỹ thuật | Mục đích trong dự án | Cơ sở nghiên cứu |
|---|---|---|
| Depthwise-separable 1D-CNN | Giảm tham số/MAC nhưng vẫn học temporal và spatial EEG | EEGNet: Lawhern et al., 2018 |
| DWT ablation | Kiểm tra đặc trưng thời gian-tần số phổ biến trên CHB-MIT | Kashefi Amiri et al., 2025; nhiều phương pháp DWT cổ điển |
| Causal temporal policy | Đổi chuỗi xác suất thành cảnh báo trực tuyến, không nhìn tương lai | Bản chất continuous detector của Shoeb and Guttag, 2010; TCN của Bai et al., 2018 |
| Hard-negative mining | Dạy mô hình phân biệt nhiễu/dạng normal dễ gây báo động | Thực hành metric learning/classification; được kiểm chứng bằng ablation Run 02, 03, 05, 15, 17 |
| Early stopping | Chọn checkpoint theo validation loss, giảm overfit và thời gian train | Prechelt, 1998 |
| Validation-only selection | Ngăn test leakage (rò rỉ thông tin test vào quyết định thiết kế) | Thực hành thực nghiệm tái lập được; cần thiết cho Q1 |

Nguồn chính: [EEGNet](https://doi.org/10.1088/1741-2552/aace8c),
[Shoeb and Guttag](https://physionet.org/files/chbmit/1.0.0/shoeb-icml-2010.pdf),
[TCN](https://arxiv.org/abs/1803.01271),
[Kashefi Amiri et al.](https://doi.org/10.1038/s41598-025-18479-9),
và [Prechelt](https://pubmed.ncbi.nlm.nih.gov/12662814/).

## 6. Cách trình bày với giáo sư và hướng tiếp theo

Tôi sẽ không trình bày 90.07% như kết quả lâm sàng. Cách nói chính xác là:

> Chúng em đã chọn được một backbone rất nhỏ, EpiSepNet-5K, đạt 90.07% độ
> chính xác cửa sổ trên validation, chỉ 5,013 tham số. Qua 21 thực nghiệm có
> đối chứng, chúng em đã loại các hướng DWT, hard-negative quá mạnh và
> multiscale vì chúng không tốt hơn khi xét báo động liên tục. Khoảng cách còn
> lại là nâng event sensitivity từ 79.31% lên ít nhất 90% trong khi giữ FAR/h
> dưới 0.5 và giảm độ trễ xuống dưới 10 s. Sau đó cần khóa protocol mới,
> patient-independent hoặc leave-one-patient-out, và chỉ đánh giá test cuối
> một lần để tạo bằng chứng đủ mạnh cho Q1.

Các bước nghiêm túc tiếp theo là:

1. Audit lại selection trace của Run 10 và đóng băng EpiSepNet-5K/Run 21 làm
   reference tái lập được.
2. Thiết kế một ablation nguyên nhân rõ ràng để cải thiện event sensitivity và
   delay, không tiếp tục điều chỉnh policy trên test.
3. Chạy protocol patient-held-out hoặc leave-one-patient-out (LOPO, lần lượt
   giữ một bệnh nhân ngoài train) để có kết quả tổng quát hóa đáng tin cậy.
4. Chỉ sau khi chốt detector mới chuyển pipeline lọc sang causal preprocessing
   (tiền xử lý nhân quả) và đo latency, LUT, DSP, BRAM, năng lượng trên KV260.

## 7. Định nghĩa nhanh các từ dễ nhầm

- **DẢ**: trong ngữ cảnh các trao đổi trước, nhiều khả năng là viết nhầm của
  **FAR** (False Alarm Rate, tần suất báo động giả), thường báo là FAR/h.
- **Ictal**: đoạn EEG đang xảy ra cơn động kinh.
- **Interictal**: đoạn EEG không có cơn.
- **Epoch/window**: đoạn EEG ngắn được cắt để mạng phân loại; không phải epoch
  huấn luyện.
- **Epoch training**: một lượt mô hình đi hết các batch của train set.
- **Backbone**: phần mạng chính biến tín hiệu đầu vào thành biểu diễn để phân
  loại.
- **AMP/FP16**: Automatic Mixed Precision (tính toán train hỗn hợp FP16/FP32
  trên GPU); không phải quantization (lượng tử hóa) mô hình đã triển khai.
- **INT16 quantization**: biểu diễn giá trị deploy bằng số nguyên có dấu 16-bit
  để giảm footprint; EpiSepNet-5K INT16 hiện có 99.9743% agreement (độ trùng
  dự đoán) với FP32 trên validation.
- **Ablation**: thực nghiệm thay hoặc bỏ đúng một thành phần để xác định đóng
  góp nhân quả của thành phần đó.
- **Clinical gate**: bộ tiêu chí sàng lọc nội bộ cho detector trước khi tuyên
  bố kết quả; không phải chuẩn pháp lý hay chuẩn chung của mọi journal.
