# CHB-MIT Paper Knowledge Base for EEG 1D-CNN Accelerator

Created: 2026-07-23

This document turns the user-provided CHB-MIT paper list into a working knowledge base for research, experiment planning, and architecture decisions in this project. It focuses on what should affect our data preprocessing, labeling, model design, evaluation protocol, and hardware-aware training path.

Important distinction: the listed works do not all solve the same task. Some are seizure detection papers, while others are seizure prediction papers. Their labels, windows, metrics, and conclusions cannot be compared directly without normalizing the protocol.

---

## 1. Project-Relevant CHB-MIT Standard

The current project is built around the CHB-MIT Scalp EEG Database from PhysioNet.

Key dataset facts that define our preprocessing:

- Data format: continuous scalp EEG files in EDF format, grouped by case/patient folders such as `chb01`, `chb02`, etc.
- Sampling rate: 256 Hz.
- Channel layout: most files contain 23 EEG signals, with some records having 24 or 26 signals; the recordings follow the International 10-20 electrode system.
- Annotation basis: each case has `chbXX-summary.txt`; seizure start and end times are given in seconds from the beginning of each EDF file. PhysioNet also provides `.seizure` annotation files for seizure records.
- Dataset scale: PhysioNet lists 664 EDF files and 129 files with seizures; the current PhysioNet page says the records include 198 seizures, while the older/original set had 182 annotated seizures.

Source:

- PhysioNet CHB-MIT v1.0.0: https://physionet.org/content/chbmit/1.0.0/
- Example summary file format: https://physionet.org/content/chbmit/1.0.0/chb16/chb16-summary.txt

Mapping to this repo:

- Current CHB-MIT preprocessing code: `AI_train_model/src/preprocess_chbmit.py`
- Current model input: `(channels=23, length=256)`, meaning 1 second of EEG at 256 Hz.
- Current task implemented in code: seizure detection, not seizure prediction.

---

## 2. Detection vs Prediction

This is the most important conceptual split.

| Task | Label rule | Typical positive class | Typical negative class | Current repo support |
|---|---|---|---|---|
| Seizure detection | Uses known seizure onset/end annotations | Ictal window, inside seizure | Interictal/non-seizure window | Supported now |
| Seizure prediction | Uses a preictal interval before seizure onset | Preictal window before seizure | Interictal window far from seizure | Not implemented yet |

For detection in this repo:

- A 1-second window entirely inside `[seizure_start, seizure_end]` gets label `1`.
- A 1-second window entirely outside all seizure intervals gets label `0`.
- Boundary windows are skipped by the current code because they are neither fully ictal nor fully normal.

For prediction, the repo would need a new labeling mode:

- Define a preictal length, such as 5, 10, 15, or 30 minutes before seizure onset.
- Define a seizure prediction horizon or exclusion gap before ictal onset, commonly several minutes.
- Label preictal windows as `1`.
- Label interictal windows sufficiently far from any seizure as `0`.
- Exclude ictal, postictal, and ambiguous near-seizure windows.

Do not train a "prediction" model using the current detection labels; that would answer a different clinical question.

---

## 3. Paper Matrix

### P1. CSP + CNN for Seizure Prediction

Paper:

- Zhang, Y.; Guo, Y.; Yang, P.; Chen, W.; Lo, B. "Epilepsy Seizure Prediction on EEG Using Common Spatial Pattern and Convolutional Neural Network." IEEE Journal of Biomedical and Health Informatics, 24(2), 465-474, 2020.
- DOI: `10.1109/JBHI.2019.2933046`
- White Rose record: https://eprints.whiterose.ac.uk/id/eprint/151225/
- Accepted-version PDF: https://eprints.whiterose.ac.uk/id/eprint/151225/15/JBHI__1_.pdf

Task:

- Seizure prediction: preictal vs interictal.

CHB-MIT use:

- Evaluated on 23 patients from the Boston Children's Hospital-MIT scalp EEG dataset.
- Uses leave-one-out cross-validation.

Method:

- Generates artificial preictal EEG signals by combining segmented preictal signals to reduce trial imbalance.
- Uses wavelet packet decomposition and common spatial pattern to extract time-domain and frequency-domain discriminative features.
- Uses a shallow CNN to classify preictal vs interictal state.

Reported result:

- Sensitivity: 92.2%.
- False prediction rate: 0.12/h.

What we should learn:

- CSP is useful when the model must explicitly exploit spatial covariance across channels.
- Prediction requires preictal labeling, not the current ictal/non-ictal labels.
- Leave-one-out or patient-aware validation matters for credibility.

Hardware relevance:

- CSP and wavelet packet decomposition add preprocessing cost.
- The shallow CNN part is hardware-friendly, but CSP covariance/eigendecomposition is less attractive for a simple FPGA/ASIC pipeline unless done offline or approximated.

---

### P2. Single-Channel Seizure Detection With Clinical Channel Selection

Paper:

- Chung, Y. G.; Cho, A.; Kim, H.; Kim, K. J. "Single-channel seizure detection with clinical confirmation of seizure locations using CHB-MIT dataset." Frontiers in Neurology, 15:1389731, 2024.
- DOI: `10.3389/fneur.2024.1389731`
- Article: https://www.frontiersin.org/journals/neurology/articles/10.3389/fneur.2024.1389731/full
- PDF: https://www.frontiersin.org/journals/neurology/articles/10.3389/fneur.2024.1389731/pdf

Task:

- Seizure detection: ictal vs interictal, with event-level seizure detection.

CHB-MIT use:

- Uses CHB-MIT scalp EEG.
- Builds patient-specific detectors.
- Uses 13 selected cases after neurologist review and channel-location suitability checks.

Method:

- Compares 18-channel, 4-channel, and single-channel seizure detectors.
- Single channel is selected clinically from wearable-friendly candidate channels.
- Neurologists re-annotate seizure onset/termination for clearer spatial seizure characteristics.
- Uses deep learning detectors; the paper discusses 2D CNN modules with different filter lengths.

Reported result:

- Multi- and single-channel detectors: sensitivity around 97.05-100%, false alarm rate 0.22-0.40/h, latency 2.1-3.4 s.
- Single-channel event-level result: sensitivity 99.62%, FAR 0.22/h, latency 3.3 s on average.

What we should learn:

- Channel reduction can work if channel selection is clinically/patient informed.
- Event-level metrics are more meaningful than only segment accuracy.
- Annotation quality can change false alarm and latency behavior.

Hardware relevance:

- Very relevant. Reducing from 23 channels to 18, 4, or 1 channel reduces multiply-accumulate count, RAM bandwidth, and input buffering.
- Good candidate for a later channel-pruning/channel-selection experiment.

---

### P3. Deep Hybrid CNN-GAN Model

Paper:

- Hasan, N.; Sharif, M. M.; Rahman, M.; Islam, M. S.; Khan, M. T. R.; Hoque, M. J.; Bahar, A. "Epileptic Seizure Prediction Using a Deep Hybrid CNN-GAN Model on EEG Data." 2024 IEEE 3rd International Conference on Robotics, Automation, Artificial-Intelligence and Internet-of-Things (RAAICON), pp. 270-275, 2024.
- DOI: `10.1109/RAAICON64172.2024.10928493`
- IEEE page: https://ieeexplore.ieee.org/document/10928493/
- ResearchGate page: https://www.researchgate.net/publication/390153672_Epileptic_Seizure_Prediction_Using_a_Deep_Hybrid_CNN-GAN_Model_on_EEG_Data

Task:

- Title says seizure prediction, but the abstract language mixes prediction and detection. Treat it as a GAN-augmented EEG classification/prediction paper until the full PDF protocol is reviewed.

CHB-MIT use:

- Reports results on BONN, CHB-MIT, and SWEC-ETHZ datasets.

Method:

- Preprocesses EEG to remove noise/artifacts.
- Uses data augmentation to address imbalance and overfitting.
- Uses a custom CNN for feature extraction.
- Uses a GAN-related component for synthetic data/classification.

Reported result:

- ResearchGate abstract reports 99.95% on BONN, 98.93% on CHB-MIT, and 98.80% on SWEC-ETHZ.

What we should learn:

- GAN-based synthetic EEG generation is one possible response to seizure-class scarcity.
- The method must be audited carefully because GAN augmentation can leak patterns or overstate performance if train/test separation is weak.

Hardware relevance:

- GAN is not appropriate for the deployment accelerator.
- Possible use: offline data augmentation during training only.
- The deployed model should remain a compact CNN or CNN-derived architecture.

Confidence note:

- Metadata and high-level abstract verified from IEEE/ResearchGate search results. Full methodology should be manually checked from the PDF before adopting reported details.

---

### P4. Semi-Supervised GAN for Seizure Prediction

Paper:

- Truong, N. D.; Kuhlmann, L.; Bonyadi, M. R.; Kavehei, O. "Semi-supervised Seizure Prediction with Generative Adversarial Networks." arXiv:1806.08235, 2018; also appeared in EMBC 2019.
- arXiv: https://arxiv.org/abs/1806.08235
- PDF: https://arxiv.org/pdf/1806.08235

Task:

- Seizure prediction.

CHB-MIT use:

- Evaluated on CHB-MIT scalp EEG and Freiburg Hospital intracranial EEG.

Method:

- Uses 28-second EEG windows.
- Applies short-time Fourier transform as preprocessing.
- Trains GAN discriminator without seizure-onset labels, then uses the discriminator as a feature extractor.
- Classifies extracted features with two fully connected layers, or another classifier.
- Motivation is semi-supervised learning from unlabeled EEG.

Reported result:

- AUC: 77.68% on CHB-MIT.
- AUC: 75.47% on Freiburg.

What we should learn:

- Semi-supervised pretraining may be useful when labeled seizure windows are scarce.
- STFT spectrogram representations are common for prediction.
- Reported AUC is modest, so this is more valuable as a concept than as the performance target.

Hardware relevance:

- GAN training is offline only.
- STFT frontend and FC classifier are possible, but current hardware direction is simpler with raw/DWT 1D-CNN.

---

### P5. DWT + 1D CNN-LSTM for Seizure Detection

Paper:

- Kashefi Amiri, H.; Zarei, M.; Daliri, M. R. "Epileptic seizure detection from electroencephalogram signals based on 1D CNN-LSTM deep learning model using discrete wavelet transform." Scientific Reports, 15:32820, 2025.
- DOI: `10.1038/s41598-025-18479-9`
- Article: https://www.nature.com/articles/s41598-025-18479-9
- PDF: https://www.nature.com/articles/s41598-025-18479-9.pdf

Task:

- Seizure detection on CHB-MIT.
- Also evaluates other datasets/tasks such as BONN and TUSZ.

CHB-MIT use:

- Binary seizure vs non-seizure classification.
- Uses 10-fold cross-validation and class-weighted loss to address imbalance.

Method:

- Applies DWT separately to each EEG channel.
- Uses Daubechies-3 as mother wavelet in the described setup.
- Concatenates DWT coefficients into 1D feature vectors while preserving channel-wise independence.
- Feeds DWT-derived vectors to a 1D CNN-LSTM model.
- Architecture summary: input layer, six convolutional layers, four pooling layers, one LSTM layer, one fully connected layer, softmax output.

Reported result on CHB-MIT:

- Accuracy: 96.94%.
- Kappa: 94.33%.
- GDR: 96.36%.
- Reported computational complexity for CHB-MIT: about `1.67e6`.

What we should learn:

- DWT is a strong candidate frontend for CHB-MIT because it improves separability on noisy real EEG.
- The ablation discussion says CHB-MIT performance drops sharply if DWT, CNN, or LSTM is removed, implying this dataset benefits from time-frequency and temporal modeling.

Hardware relevance:

- DWT can be implemented or approximated on hardware, but it increases preprocessing complexity.
- LSTM is less hardware-simple than Conv1D. For this project, test DWT + compact 1D-CNN before committing to LSTM.

---

### P6. Feature Fusion + CNN-Bi-LSTM for Seizure Detection

Paper:

- Cao, X.; Zheng, S.; Zhang, J.; Chen, W.; Du, G. "A hybrid CNN-Bi-LSTM model with feature fusion for accurate epilepsy seizure detection." BMC Medical Informatics and Decision Making, 25:6, 2025.
- DOI: `10.1186/s12911-024-02845-0`
- Article: https://link.springer.com/article/10.1186/s12911-024-02845-0
- PDF: https://link.springer.com/content/pdf/10.1186/s12911-024-02845-0.pdf

Task:

- Seizure detection.

CHB-MIT use:

- Main validation includes Bonn and New Delhi datasets.
- Additional validation on all 23 CHB-MIT cases.

Method:

- Applies DWT five-level decomposition.
- Extracts time-frequency and nonlinear features from decomposed sub-bands.
- Uses features including approximate entropy, fuzzy entropy, RMS, and Hurst exponent.
- Uses SVM-RFE for feature selection.
- Uses CNN-Bi-LSTM for classification.

Reported result on CHB-MIT:

- Accuracy: 98.43%.
- Sensitivity: 97.84%.
- Specificity: 99.21%.
- Precision: 99.14%.
- F1-score: 98.39%.

What we should learn:

- Feature fusion can improve performance on noisy clinical EEG.
- CHB16 is flagged as difficult due to class imbalance even after oversampling.
- The paper itself notes the CNN-Bi-LSTM model has high computational complexity and many parameters, which is important for our accelerator constraints.

Hardware relevance:

- Direct CNN-Bi-LSTM + handcrafted feature fusion is probably too complex for the first accelerator target.
- Useful for deciding which features matter if the compact 1D-CNN underperforms.

---

### P7. Epileptic Disorder Detection Using EEG Signals

Paper:

- Alharthi, M. K.; Moria, K. M.; Alghazzawi, D. M.; Tayeb, H. O. "Epileptic Disorder Detection of Seizures Using EEG Signals." Sensors, 22(17):6592, 2022.
- DOI: `10.3390/s22176592`
- MDPI article: https://www.mdpi.com/1424-8220/22/17/6592
- PubMed: https://pubmed.ncbi.nlm.nih.gov/36081048/

Task:

- Seizure detection.

CHB-MIT use:

- Integrates local EEG signals from King Abdulaziz University hospital with CHB-MIT using a compatibility framework.
- CHB-MIT is selected because it has similar scalp EEG recordings and annotations to the local KAU dataset.

Method:

- Proposes a compatibility framework for integrating local XLtek EEG data with CHB-MIT.
- Performs dominant channel selection.
- Tests integrated selective-channel datasets using a deep learning model composed of 1D-CNN, Bi-LSTM, and attention.

Reported result:

- Accuracy: up to 96.87%.
- Precision: 96.98%.
- Sensitivity: 96.85%.

What we should learn:

- Data compatibility is a first-class research issue when mixing hospital data and CHB-MIT.
- Dominant channel selection may reduce channel count without destroying detection performance.
- Attention can help but increases implementation complexity.

Hardware relevance:

- The 1D-CNN part matches our direction.
- Bi-LSTM and attention are heavier; treat them as research comparators, not first hardware baseline.

---

### P8. Hybrid DenseNet-LSTM for Seizure Prediction

Paper:

- Ryu, S.; Joe, I. "A Hybrid DenseNet-LSTM Model for Epileptic Seizure Prediction." Applied Sciences, 11(16):7661, 2021.
- DOI: `10.3390/app11167661`
- MDPI article: https://www.mdpi.com/2076-3417/11/16/7661
- Hanyang repository record: https://scholarworks.bwise.kr/hanyang/handle/2021.sw.hanyang/141393
- PDF: https://scholarworks.bwise.kr/hanyang/bitstream/2021.sw.hanyang/141393/1/applsci-11-07661-v2.pdf

Task:

- Seizure prediction.

CHB-MIT use:

- Uses CHB-MIT scalp EEG.
- Uses 18 common bipolar channels.
- Experiments with preictal lengths of 5, 10, and 15 minutes.

Method:

- Segments raw EEG by channel.
- Uses 10-second windows with 1-second overlap.
- Applies DWT using Daubechies-4 (`db4`) to convert EEG into time-frequency 2D images.
- Uses DWT level 7, with frequency coverage noted as 2-128 Hz.
- Excludes a 5-minute interval directly before ictal onset from the preictal class to leave reaction time before seizure.
- Uses DenseNet to extract image features and LSTM for temporal modeling.

Reported result:

- Best reported at 5-minute preictal length:
  - Accuracy: 93.28%.
  - Sensitivity: 92.92%.
  - Specificity: 93.65%.
  - False positive rate: 0.063/h.
  - F1-score: 0.923.

What we should learn:

- Prediction requires a separate data builder with preictal length and exclusion horizon.
- DWT + 2D image conversion is effective for prediction, but not automatically aligned with our current 1-second raw Conv1D design.

Hardware relevance:

- DenseNet-LSTM is heavier than our <100K-parameter target.
- Its label protocol is more valuable to us than its exact architecture.

---

### P9. Parallel Memristive CNNs for Seizure Detection and Prediction

Paper:

- Li, C.; Lammie, C.; Dong, X.; Amirsoleimani, A.; Azghadi, M. R.; Genov, R. "Seizure Detection and Prediction by Parallel Memristive Convolutional Neural Networks." IEEE Transactions on Biomedical Circuits and Systems, 2022.
- DOI: `10.1109/TBCAS.2022.3185584`
- arXiv: https://arxiv.org/abs/2206.09951
- PDF: https://arxiv.org/pdf/2206.09951

Task:

- Both seizure detection and seizure prediction.

CHB-MIT use:

- Evaluates seizure prediction on CHB-MIT.
- Also evaluates detection/prediction on University of Bonn and SWEC-ETHZ.

Method:

- Proposes a low-latency parallel CNN architecture.
- Reduces parameter count by 2-2,800x compared with prior CNN architectures.
- Maps CNN computations onto RRAM/memristive analog crossbar arrays.
- Parallelizes convolution kernels across separate analog crossbars.
- Studies non-ideal hardware effects.
- Uses quantization-aware training to mitigate low ADC/DAC resolution degradation.
- Proposes stuck-weight offsetting to recover degradation from stuck RON/ROFF memristor weights.

Reported result:

- 5-fold cross-validation accuracy:
  - 99.84% for seizure detection.
  - 99.01% and 97.54% for seizure prediction on CHB-MIT and SWEC-ETHZ respectively, based on the abstract wording.
- Hardware estimate for CNN component: about 2.791 W and 31.255 mm^2 in 22 nm FDSOI CMOS process.

What we should learn:

- This is the most directly aligned paper for the accelerator side.
- Parallel kernel execution, low parameter count, quantization-aware training, and robustness to non-ideal weights are directly relevant to FPGA/ASIC planning.

Hardware relevance:

- Very high. Use it as the main architecture/hardware reference, while adapting to our simpler Q16 fixed-point export path.

---

## 4. Cross-Paper Patterns

### 4.1 Labeling and validation matter more than headline accuracy

Reported accuracies are not directly comparable because:

- Detection and prediction have different labels.
- Some papers are patient-specific; others attempt patient-independent or all-subject validation.
- Some use segment-level accuracy; others report event-level sensitivity, false alarm rate, and latency.
- Some use re-annotation or clinical channel confirmation.
- Some use all CHB-MIT cases; others use selected subsets.

For this project, every experiment should log:

- Task: detection or prediction.
- Cases/patients used.
- Channels used.
- Window length and overlap.
- Labeling rule.
- Split strategy: random window split, patient-specific, leave-one-subject-out, leave-one-seizure-out, or k-fold.
- Metrics: segment accuracy, sensitivity, specificity, precision, F1, AUC if available, plus FAR/h and latency for continuous detection.

### 4.2 DWT appears repeatedly

DWT appears in the CSP+CNN, 1D CNN-LSTM, CNN-Bi-LSTM, and DenseNet-LSTM lines of work.

Research implication:

- DWT is likely worth testing as an optional preprocessing/frontend module.

Hardware implication:

- DWT costs additional logic and buffering.
- A pragmatic sequence is:
  1. Train raw-signal compact 1D-CNN.
  2. Add conventional bandpass filtering.
  3. Test DWT-derived channels/features with a compact 1D-CNN.
  4. Only consider LSTM/attention if accuracy or false alarms remain unacceptable.

### 4.3 Channel reduction is a strong accelerator lever

Single-channel and dominant-channel papers show that not all 23 channels may be necessary for a patient-specific detector.

Research implication:

- Add experiments with 23, 18, 4, and selected 1-channel inputs.

Hardware implication:

- Channel reduction directly lowers input memory, convolution MACs, and latency.

### 4.4 GANs are useful only as offline training tools

GAN papers focus on augmentation or semi-supervised feature learning.

Research implication:

- GANs may help class imbalance or unlabeled data usage.

Hardware implication:

- Do not deploy GANs on the accelerator.
- If used, keep them in the offline training pipeline and export only the compact classifier.

### 4.5 LSTM/Bi-LSTM/attention improve temporal modeling but hurt hardware simplicity

Several high-performing papers use LSTM, Bi-LSTM, or attention.

Research implication:

- These should be baselines/comparators if pure CNN underperforms.

Hardware implication:

- For a first FPGA/ASIC design, prefer Conv1D, pooling, ReLU, and FC layers.
- If temporal recurrence is needed, consider temporal convolution or dilated/depthwise Conv1D before LSTM.

---

## 5. Impact on Current Repository

The current repo pipeline is detection-oriented:

```text
EDF + summary annotations
  -> 1-second windows
  -> label 1 if fully ictal, label 0 if fully non-ictal
  -> balance classes
  -> train 23 x 256 compact 1D-CNN
  -> fold BatchNorm
  -> Q16 dynamic quantization
  -> export flat weights
```

Research gaps before claiming paper-level results:

1. The current preprocessing is restricted to the first 5 subject folders. Most papers use more CHB-MIT cases or explicitly state selected cases.
2. The current train/val/test split is a random window split after preprocessing. This can leak patient/file-specific patterns across train and test and may inflate accuracy.
3. The current code does not implement a seizure prediction label mode.
4. The current code does not implement event-level FAR/h and latency metrics.
5. The current code mentions bandpass filtering in docs, but the preprocessing code currently does not apply a bandpass filter.
6. The current model is compact and hardware-friendly, but does not yet include channel-reduction experiments, DWT frontend, QAT, or patient-aware validation.

---

## 6. Recommended Research Plan

### Phase A. Make the detection baseline scientifically defensible

- Use CHB-MIT EDF data, not the `Epileptic_Seizure_Recognition.csv`, unless building a separate CSV experiment.
- Process all intended cases, not only the first 5.
- Add config options for cases, channels, window size, and overlap.
- Add patient-aware split modes:
  - patient-specific train/test,
  - leave-one-subject-out,
  - leave-one-seizure-out,
  - random window split only as a smoke-test baseline.
- Add event-level metrics: sensitivity, FAR/h, latency.

### Phase B. Hardware-friendly architecture search

Start with models that can still be deployed simply:

1. Current raw 23-channel 1D-CNN baseline.
2. Raw 18-channel 1D-CNN using common CHB-MIT channels.
3. 4-channel and selected-channel 1D-CNN.
4. Depthwise-separable or grouped Conv1D.
5. DWT or bandpass frontend + compact 1D-CNN.
6. Quantization-aware training before Q16 export.

Avoid for first hardware version:

- GAN inference.
- Bi-LSTM/attention-heavy inference.
- Large DenseNet image pipeline.

### Phase C. Separate prediction branch

If the project expands from detection to prediction, add a separate data builder:

- `task: prediction`
- `preictal_minutes: [5, 10, 15]`
- `prediction_horizon_minutes: 5`
- `interictal_exclusion_minutes`: configurable
- Exclude ictal/postictal/boundary windows
- Report FAR/h and sensitivity under a continuous timeline protocol

---

## 7. Paper Priority for This Project

Highest relevance:

1. Li et al. 2022, parallel memristive CNNs: hardware and quantization direction.
2. Chung et al. 2024, single-channel detection: channel reduction and event-level evaluation.
3. Kashefi Amiri et al. 2025, DWT + 1D CNN-LSTM: DWT and compact-ish temporal frontend evidence.
4. Zhang et al. 2020, CSP + CNN: prediction protocol and spatial feature extraction.

Medium relevance:

5. Cao et al. 2025, feature fusion + CNN-Bi-LSTM: high-performing but heavy comparator.
6. Alharthi et al. 2022, compatibility framework: useful when mixing datasets or reducing channels.
7. Ryu and Joe 2021, DenseNet-LSTM: useful for prediction labeling and DWT image pipeline, less suitable for hardware.

Lower immediate relevance:

8. Truong et al. 2018/2019 semi-supervised GAN: useful for unlabeled-data thinking, modest reported AUC.
9. Hasan et al. 2024 CNN-GAN: potentially useful augmentation idea, but needs full PDF protocol audit.

---

## 8. Source Index

- CHB-MIT PhysioNet: https://physionet.org/content/chbmit/1.0.0/
- CHB-MIT example summary: https://physionet.org/content/chbmit/1.0.0/chb16/chb16-summary.txt
- Zhang et al. CSP+CNN White Rose: https://eprints.whiterose.ac.uk/id/eprint/151225/
- Zhang et al. CSP+CNN PDF: https://eprints.whiterose.ac.uk/id/eprint/151225/15/JBHI__1_.pdf
- Chung et al. Frontiers article: https://www.frontiersin.org/journals/neurology/articles/10.3389/fneur.2024.1389731/full
- Chung et al. PDF: https://www.frontiersin.org/journals/neurology/articles/10.3389/fneur.2024.1389731/pdf
- Hasan et al. IEEE page: https://ieeexplore.ieee.org/document/10928493/
- Hasan et al. ResearchGate page: https://www.researchgate.net/publication/390153672_Epileptic_Seizure_Prediction_Using_a_Deep_Hybrid_CNN-GAN_Model_on_EEG_Data
- Truong et al. arXiv: https://arxiv.org/abs/1806.08235
- Truong et al. PDF: https://arxiv.org/pdf/1806.08235
- Kashefi Amiri et al. Scientific Reports: https://www.nature.com/articles/s41598-025-18479-9
- Kashefi Amiri et al. PDF: https://www.nature.com/articles/s41598-025-18479-9.pdf
- Cao et al. BMC/Springer: https://link.springer.com/article/10.1186/s12911-024-02845-0
- Cao et al. PDF: https://link.springer.com/content/pdf/10.1186/s12911-024-02845-0.pdf
- Alharthi et al. MDPI: https://www.mdpi.com/1424-8220/22/17/6592
- Alharthi et al. PubMed: https://pubmed.ncbi.nlm.nih.gov/36081048/
- Ryu and Joe Hanyang record: https://scholarworks.bwise.kr/hanyang/handle/2021.sw.hanyang/141393
- Ryu and Joe PDF: https://scholarworks.bwise.kr/hanyang/bitstream/2021.sw.hanyang/141393/1/applsci-11-07661-v2.pdf
- Li et al. arXiv: https://arxiv.org/abs/2206.09951
- Li et al. PDF: https://arxiv.org/pdf/2206.09951
