# CHB-MIT / EEG Seizure Research PDF Corpus

Created: 2026-07-23
Last updated: 2026-07-23

This folder stores the local PDF corpus used for architecture research on the EEG 1D-CNN accelerator project.

Status: 22 PDF files have been downloaded and verified by checking the `%PDF` file header. Several useful papers remain blocked by IEEE, ResearchGate, MIT DSpace AWS WAF, MDPI Akamai, Royal Society/AHA Cloudflare, or SSRN connectivity.

---

## Downloaded And Verified PDFs

| ID | Local file | Paper / role | Research use |
|---|---|---|---|
| P01 | `01_zhang_2020_csp_cnn_seizure_prediction.pdf` | CSP + CNN seizure prediction | Prediction labeling, CSP, wavelet/spatial features |
| P02 | `02_chung_2024_single_channel_chbmit_detection.pdf` | Single-channel CHB-MIT seizure detection | Channel reduction, event-level metrics |
| P04 | `04_truong_2018_semisupervised_gan_prediction.pdf` | Semi-supervised GAN seizure prediction | Offline augmentation / unlabeled learning |
| P05 | `05_kashefi_2025_dwt_1dcnn_lstm_detection.pdf` | DWT + 1D CNN-LSTM seizure detection | DWT frontend, CNN-LSTM comparator |
| P06 | `06_cao_2025_feature_fusion_cnn_bilstm_detection.pdf` | Feature fusion + CNN-Bi-LSTM | Heavy high-accuracy comparator |
| P07 | `07_alharthi_2022_epileptic_disorder_detection.pdf` | EEG disorder detection / CHB-MIT integration | Dataset compatibility, channel selection |
| P08 | `08_ryu_2021_densenet_lstm_prediction.pdf` | DenseNet-LSTM seizure prediction | Preictal protocol, DWT image pipeline |
| P09 | `09_li_2022_parallel_memristive_cnn_detection_prediction.pdf` | Parallel memristive CNN | Main hardware/accelerator reference |
| P10 | `10_goldberger_2000_physionet_resource.pdf` | PhysioNet resource paper | Dataset infrastructure citation |
| P12 | `12_shoeb_guttag_2010_ml_seizure_detection_icml.pdf` | CHB-MIT seizure detection baseline | Core CHB-MIT detection citation |
| P13 | `13_ali_2024_chbmit_overlooked_perspectives.pdf` | CHB-MIT overlooked perspectives | Data leakage, imbalance, event detection |
| P14 | `14_zhang_2024_review_eeg_processing_deep_learning.pdf` | Review of EEG processing and DL | Survey/background for Q1 framing |
| P15 | `15_ozkurt_2021_chbmit_pediatric_survey.pdf` | CHB-MIT pediatric seizure detection survey | CHB-MIT-specific literature map |
| P16 | `16_lawhern_2018_eegnet_compact_cnn.pdf` | EEGNet compact CNN | Depthwise/separable EEG CNN design |
| P17 | `17_schirrmeister_2017_deep_cnn_eeg_decoding.pdf` | Deep CNNs for EEG decoding | Raw EEG CNN architecture rationale |
| P18 | `18_jacob_2018_integer_only_quantization.pdf` | Integer-only quantized inference | QAT / integer inference foundation |
| P19 | `19_ahlawat_2026_int8_quant_channel_pruning_snn.pdf` | INT8, channel pruning, SNN for CHB-MIT | Latest efficiency baseline; preprint |
| P21 | `21_eard_2024_earsd_lightweight_wearable_seizure_detection.pdf` | Ear-worn wearable seizure detection | Wearable constraints, practical monitoring |
| P25 | `25_alhammadi_2022_1dcnn_fpga_accelerator_hls.pdf` | 1D-CNN FPGA accelerator with HLS | Conv1D accelerator design reference |
| P26 | `26_biondi_2026_ai_wearable_seizure_detection_devices.pdf` | AI in wearable seizure detection devices | Clinical/wearable deployment context |
| P28 | `28_lee_2022_real_time_seizure_detection_eeg.pdf` | Real-time seizure detection using EEG | Continuous real-time evaluation framing |
| P30 | `30_frontiers_2026_channel_frequency_selection_chbmit.pdf` | Channel/frequency selection for CHB-MIT | Channel pruning and band selection |

---

## Missing Or Blocked PDFs

| ID | Paper | Source attempted | Status |
|---|---|---|---|
| P03 | Epileptic Seizure Prediction Using a Deep Hybrid CNN-GAN Model on EEG Data | IEEE direct PDF, IEEE stamp endpoint, ResearchGate | Blocked by IEEE URL rejection and ResearchGate `1020`/login protection |
| P11 | Shoeb 2009 PhD thesis: Application of Machine Learning to Epileptic Seizure Onset Detection and Treatment | MIT DSpace handle and bitstream | MIT DSpace AWS WAF/captcha returns HTML, not PDF |
| P20 | Reducing False Alarms in Wearable Seizure Detection With EEGformer | Polito repository | Cloudflare challenge |
| P22 | Tiny CNN for Seizure Prediction in Wearable Biomedical Devices | IEEE direct PDF | IEEE URL rejection |
| P23 | RISC-V CNN Coprocessor for Real-Time Epilepsy Detection in Wearable Application | IEEE direct PDF, ResearchGate public full-text link | IEEE rejection and ResearchGate `1020` |
| P24 | FPGA-Based Hardware Accelerator on Portable Equipment for EEG Signal Patterns Recognition | MDPI direct PDF | MDPI/Akamai access denied |
| P27 | Seizure detection using EEG on the CHB-MIT dataset via multi-domain feature engineering and classical ML | Springer PDF | Downloaded HTML challenge, removed from corpus |
| P29 | Epileptic seizure detection using FPGA-accelerated neural networks | SSRN PDF | Remote connection failed |

Manual placement names:

- `03_hasan_2024_hybrid_cnn_gan_prediction.pdf`
- `11_shoeb_2009_ml_seizure_onset_detection_thesis.pdf`
- `20_ingolfsson_2024_eegformer_false_alarms_mcu.pdf`
- `22_zhang_2022_tiny_cnn_seizure_prediction_wearable.pdf`
- `23_lee_2021_riscv_cnn_coprocessor_epilepsy_detection.pdf`
- `24_xie_2022_fpga_hardware_accelerator_eeg_patterns.pdf`
- `27_ghosh_2026_chbmit_patient_independent_no_leakage.pdf`
- `29_klos_2023_fpga_accelerated_neural_networks_seizure_detection.pdf`

---

## Source URLs For Downloaded PDFs

- P01: https://eprints.whiterose.ac.uk/id/eprint/151225/15/JBHI__1_.pdf
- P02: https://www.frontiersin.org/journals/neurology/articles/10.3389/fneur.2024.1389731/pdf
- P04: https://arxiv.org/pdf/1806.08235
- P05: https://europepmc.org/articles/PMC12464174?pdf=render
- P06: https://europepmc.org/articles/PMC11706039?pdf=render
- P07: https://europepmc.org/articles/PMC9459921?pdf=render
- P08: https://scholarworks.bwise.kr/hanyang/bitstream/2021.sw.hanyang/141393/1/applsci-11-07661-v2.pdf
- P09: https://arxiv.org/pdf/2206.09951
- P10: https://www.mcgill.ca/physiological-dynamics/sites/physiological-dynamics/files/componentsofanew_2000.pdf
- P12: https://physionet.org/files/chbmit/1.0.0/shoeb-icml-2010.pdf
- P13: https://europepmc.org/articles/PMC11286169?pdf=render
- P14: https://www.frontiersin.org/journals/neuroscience/articles/10.3389/fnins.2024.1468967/pdf
- P15: https://europepmc.org/articles/PMC8537151?pdf=render
- P16: https://arxiv.org/pdf/1611.08024
- P17: https://arxiv.org/pdf/1703.05051
- P18: https://openaccess.thecvf.com/content_cvpr_2018/papers/Jacob_Quantization_and_Training_CVPR_2018_paper.pdf
- P19: https://arxiv.org/pdf/2607.16296
- P21: https://arxiv.org/pdf/2401.05425
- P25: https://bura.brunel.ac.uk/bitstream/2438/25121/3/FullText.pdf
- P26: https://www.frontiersin.org/journals/neurology/articles/10.3389/fneur.2026.1756895/pdf
- P28: https://proceedings.mlr.press/v174/lee22a/lee22a.pdf
- P30: https://www.frontiersin.org/journals/neurology/articles/10.3389/fneur.2026.1831912/pdf

---

## How To Use This Corpus

Use this folder together with:

- `AI_train_model/docs/chbmit_paper_knowledge_base.md`
- `AI_train_model/docs/q1_research_readiness_knowledge.md`

For layer-by-layer model design, extract from each deployable paper:

- input shape and channel count;
- window length and overlap;
- preprocessing frontend: raw, bandpass, STFT, DWT, CSP, feature fusion;
- convolution layer count, channels, kernel sizes, pooling, normalization, activation;
- recurrent/attention blocks and whether they should be avoided for first hardware;
- split strategy and whether it prevents data leakage;
- segment-level and event-level metrics;
- quantization, pruning, sparsity, and hardware results.
