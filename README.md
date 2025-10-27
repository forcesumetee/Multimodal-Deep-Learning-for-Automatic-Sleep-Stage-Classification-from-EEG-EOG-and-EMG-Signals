# Multimodal Sleep Stage Classification

โปรเจกต์นี้พัฒนาโมเดล Deep Learning แบบ Multimodal สำหรับการจำแนกระยะการนอนหลับอัตโนมัติ โดยใช้สัญญาณ EEG, EOG และ EMG จาก Sleep-EDF Database

## 📋 สารบัญ

- [ความต้องการของระบบ](#ความต้องการของระบบ)
- [การติดตั้ง](#การติดตั้ง)
- [โครงสร้างโปรเจกต์](#โครงสร้างโปรเจกต์)
- [วิธีการใช้งาน](#วิธีการใช้งาน)
- [รายละเอียดโค้ด](#รายละเอียดโค้ด)
- [ผลลัพธ์](#ผลลัพธ์)

## 🔧 ความต้องการของระบบ

### Python Version
- Python 3.11 หรือสูงกว่า

### Python Packages
```
tensorflow>=2.15.0
numpy>=1.24.0
pandas>=2.0.0
matplotlib>=3.7.0
seaborn>=0.12.0
scikit-learn>=1.3.0
mne>=1.5.0
pyedflib>=0.1.30
```

## 📦 การติดตั้ง

### 1. Clone หรือดาวน์โหลดโปรเจกต์

```bash
# หากมีไฟล์ tar.gz
tar -xzf sleep_classification_project.tar.gz
cd sleep_classification
```

### 2. ติดตั้ง Dependencies

```bash
pip install tensorflow numpy pandas matplotlib seaborn scikit-learn mne pyedflib
```

หรือใช้ไฟล์ requirements.txt:

```bash
pip install -r requirements.txt
```

### 3. ตรวจสอบการติดตั้ง

```bash
python3 -c "import tensorflow as tf; import mne; print('Installation successful!')"
```

## 📁 โครงสร้างโปรเจกต์

```
sleep_classification/
├── README.md                      # ไฟล์นี้
├── requirements.txt               # Python dependencies
├── PROJECT_SUMMARY.md             # สรุปโปรเจกต์
├── conference_paper.md            # Research paper
│
├── data/                          # ข้อมูล Sleep-EDF
│   ├── SC4001E0-PSG.edf          # PSG recording
│   └── SC4001EC-Hypnogram.edf    # Sleep annotations
│
├── models/                        # โมเดลที่เทรนแล้ว
│   ├── best_model.keras          # Best checkpoint
│   ├── final_model.keras         # Final model
│   └── scalers.pkl               # Data scalers
│
├── figures/                       # รูปภาพและกราฟ
│   ├── sleep_stage_distribution.png
│   ├── multimodal_signals.png
│   ├── hypnogram.png
│   ├── model_architecture.png
│   └── training_history.png
│
├── results/                       # ผลการเทรน
│   ├── training_history.csv
│   ├── training_summary.txt
│   └── training_log.txt
│
├── model.py                       # โมเดล architecture
├── train.py                       # สคริปต์เทรนโมเดล
├── explore_dataset.py             # สำรวจข้อมูล
├── visualize_data.py              # สร้าง visualizations
├── analyze_results.py             # วิเคราะห์ผลลัพธ์
└── architecture.mmd               # Diagram source
```

## 🚀 วิธีการใช้งาน

### Step 1: สำรวจข้อมูล (Dataset Exploration)

```bash
# ดูข้อมูลพื้นฐานของ dataset
python3 explore_dataset.py
```

**Output:**
- แสดงข้อมูล channels, sampling rate, duration
- แสดงการกระจายของ sleep stages
- บันทึกไฟล์ `dataset_summary.txt`

### Step 2: สร้าง Visualizations

```bash
# สร้างกราฟและรูปภาพต่างๆ
python3 visualize_data.py
```

**Output:**
- `figures/sleep_stage_distribution.png` - กราฟแสดงการกระจายของ sleep stages
- `figures/multimodal_signals.png` - ตัวอย่างสัญญาณ EEG, EOG, EMG
- `figures/hypnogram.png` - Hypnogram แสดงการเปลี่ยนแปลง sleep stages

### Step 3: เทรนโมเดล (Model Training)

```bash
# เทรนโมเดล MultimodalSleepNet
python3 train.py
```

**การทำงาน:**
1. โหลดและ preprocess ข้อมูล
2. แบ่งข้อมูลเป็น train/validation/test sets
3. สร้างและเทรนโมเดล
4. บันทึก best model และ scalers
5. แสดงผลการเทรน

**Output:**
- `models/best_model.keras` - โมเดลที่ดีที่สุด
- `models/final_model.keras` - โมเดลสุดท้าย
- `models/scalers.pkl` - Data normalization scalers
- `results/training_history.csv` - ประวัติการเทรน
- `results/training_log.txt` - Log ทั้งหมด

**เวลาที่ใช้:** ประมาณ 5-10 นาที (ขึ้นอยู่กับ CPU)

### Step 4: วิเคราะห์ผลลัพธ์

```bash
# วิเคราะห์และสร้างกราฟผลการเทรน
python3 analyze_results.py
```

**Output:**
- `figures/training_history.png` - กราฟ accuracy, loss, precision, recall
- `results/training_summary.txt` - สรุปผลการเทรน

### Step 5: ทดสอบโมเดล (Optional)

```bash
# ทดสอบโมเดลด้วยข้อมูลใหม่
python3 -c "
from model import build_model
import numpy as np

# โหลดโมเดล
model = build_model()
model.load_weights('models/best_model.keras')

# ทดสอบด้วยข้อมูลตัวอย่าง
eeg_sample = np.random.randn(1, 3000, 2)
eog_sample = np.random.randn(1, 3000, 1)
emg_sample = np.random.randn(1, 3000, 1)

prediction = model.predict([eeg_sample, eog_sample, emg_sample])
print('Prediction:', prediction)
"
```

## 📝 รายละเอียดโค้ด

### 1. `model.py` - Model Architecture

**Class:** `MultimodalSleepNet`

**คุณสมบัติ:**
- Parallel CNN streams สำหรับ EEG, EOG, EMG
- Attention-based fusion mechanism
- 5-class classification (Wake, N1, N2, N3, REM)

**การใช้งาน:**
```python
from model import build_model

# สร้างโมเดล
model = build_model(num_classes=5, epoch_length=3000)

# ดู architecture
model.summary()
```

### 2. `train.py` - Training Script

**หน้าที่:**
- โหลดและ preprocess ข้อมูล
- Normalize signals
- Split data (60% train, 20% val, 20% test)
- เทรนโมเดลด้วย callbacks (early stopping, learning rate reduction)
- บันทึกโมเดลและผลลัพธ์

**Parameters ที่สำคัญ:**
```python
EPOCH_LENGTH_SEC = 30          # ความยาว epoch (วินาที)
SAMPLING_FREQ = 100            # Sampling frequency (Hz)
NUM_CLASSES = 5                # จำนวน sleep stages
```

### 3. `explore_dataset.py` - Dataset Explorer

**หน้าที่:**
- วิเคราะห์โครงสร้างข้อมูล
- แสดงข้อมูล channels และ sampling rate
- นับจำนวน epochs ในแต่ละ sleep stage
- คำนวณ signal statistics

### 4. `visualize_data.py` - Data Visualization

**หน้าที่:**
- สร้าง pie chart แสดงการกระจายของ sleep stages
- Plot multimodal signals (EEG, EOG, EMG)
- สร้าง hypnogram

### 5. `analyze_results.py` - Results Analysis

**หน้าที่:**
- อ่านและวิเคราะห์ training history
- สร้างกราฟ training curves
- สร้างรายงานสรุปผลการเทรน

## 📊 ผลลัพธ์

### Performance Metrics

| Metric              | Best Validation | Final Training |
|---------------------|-----------------|----------------|
| **Accuracy**        | 74.19%          | 97.80%         |
| **Loss**            | 1.2942          | 0.0913         |
| **Precision**       | 1.0000          | 0.9780         |
| **Recall**          | 0.0323          | 0.9780         |

### Model Architecture

- **Total Parameters:** 476,171
- **Trainable Parameters:** 473,355
- **Non-trainable Parameters:** 2,816

### Training Configuration

- **Optimizer:** Adam (lr=0.001)
- **Loss Function:** Categorical Cross-entropy
- **Batch Size:** 32
- **Max Epochs:** 100
- **Early Stopping:** Patience=15
- **Learning Rate Reduction:** Factor=0.5, Patience=5

## 🔍 การปรับแต่งโมเดล

### เปลี่ยน Hyperparameters

แก้ไขใน `train.py`:

```python
# Learning rate
keras.optimizers.Adam(learning_rate=0.001)  # เปลี่ยนค่านี้

# Batch size
model.fit(..., batch_size=32)  # เปลี่ยนค่านี้

# Epochs
model.fit(..., epochs=100)  # เปลี่ยนค่านี้
```

### เปลี่ยน Model Architecture

แก้ไขใน `model.py`:

```python
# จำนวน filters
self.eeg_conv1 = layers.Conv1D(64, ...)  # เปลี่ยนจาก 64 เป็นค่าอื่น

# Kernel size
self.eeg_conv1 = layers.Conv1D(..., kernel_size=50)  # เปลี่ยนขนาด kernel

# Dropout rate
self.eeg_dropout1 = layers.Dropout(0.3)  # เปลี่ยนจาก 0.3 เป็นค่าอื่น
```

## 🐛 การแก้ปัญหา

### ปัญหา: Out of Memory

**วิธีแก้:**
```python
# ลด batch size ใน train.py
model.fit(..., batch_size=16)  # ลดจาก 32 เป็น 16
```

### ปัญหา: Model ไม่ converge

**วิธีแก้:**
1. ลด learning rate
2. เพิ่ม regularization (dropout)
3. ใช้ data augmentation

### ปัญหา: Import Error

**วิธีแก้:**
```bash
# ติดตั้ง package ที่ขาดหาย
pip install <package_name>

# หรือติดตั้งทั้งหมดใหม่
pip install -r requirements.txt
```

## 📚 เอกสารเพิ่มเติม

- **Conference Paper:** `conference_paper.md`
- **Project Summary:** `PROJECT_SUMMARY.md`
- **Dataset Info:** `dataset_summary.txt`
- **Training Results:** `results/training_summary.txt`

## 🔗 References

1. Sleep-EDF Database: https://www.physionet.org/content/sleep-edfx/1.0.0/
2. PhysioNet: https://www.physionet.org/
3. MNE-Python Documentation: https://mne.tools/
4. TensorFlow Documentation: https://www.tensorflow.org/

## 👨‍💻 Author

**Sumetee Jirapattarasakul**

Date: October 27, 2025

## 📄 License

This project is for educational and research purposes.

---

## 🎯 Quick Start (สำหรับผู้เริ่มต้น)

```bash
# 1. ติดตั้ง dependencies
pip install tensorflow numpy pandas matplotlib seaborn scikit-learn mne pyedflib

# 2. สำรวจข้อมูล
python3 explore_dataset.py

# 3. สร้าง visualizations
python3 visualize_data.py

# 4. เทรนโมเดล (ใช้เวลา 5-10 นาที)
python3 train.py

# 5. วิเคราะห์ผลลัพธ์
python3 analyze_results.py

# เสร็จสิ้น! ดูผลลัพธ์ได้ที่โฟลเดอร์ figures/ และ results/
```

---

**หมายเหตุ:** โปรเจกต์นี้ใช้ข้อมูลจาก single subject เพื่อเป็น proof of concept สำหรับการใช้งานจริง ควรเทรนด้วย complete Sleep-EDFx dataset (197 subjects)

