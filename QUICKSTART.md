# 🚀 Quick Start Guide

คู่มือเริ่มต้นใช้งานอย่างรวดเร็วสำหรับโปรเจกต์ Multimodal Sleep Stage Classification

## ⚡ เริ่มต้นภายใน 5 นาที

### 1️⃣ ติดตั้ง Dependencies

```bash
pip install -r requirements.txt
```

หรือติดตั้งแบบ manual:

```bash
pip install tensorflow numpy pandas matplotlib seaborn scikit-learn mne pyedflib
```

### 2️⃣ รันโค้ดทั้งหมดตามลำดับ

```bash
# Step 1: สำรวจข้อมูล
python3 explore_dataset.py

# Step 2: สร้างกราฟ
python3 visualize_data.py

# Step 3: เทรนโมเดล (ใช้เวลา 5-10 นาที)
python3 train.py

# Step 4: วิเคราะห์ผลลัพธ์
python3 analyze_results.py
```

### 3️⃣ ดูผลลัพธ์

```bash
# ดูรูปภาพ
ls figures/

# ดูผลการเทรน
cat results/training_summary.txt

# ดู paper
cat conference_paper.md
```

## 📂 ไฟล์ที่สำคัญ

| ไฟล์ | คำอธิบาย |
|------|----------|
| `model.py` | โมเดล MultimodalSleepNet |
| `train.py` | สคริปต์เทรนโมเดล |
| `explore_dataset.py` | สำรวจข้อมูล |
| `visualize_data.py` | สร้างกราฟ |
| `analyze_results.py` | วิเคราะห์ผลลัพธ์ |
| `conference_paper.md` | Research paper |
| `README.md` | คู่มือฉบับเต็ม |

## 🎯 ผลลัพธ์ที่คาดหวัง

หลังจากรันโค้ดทั้งหมด คุณจะได้:

### ✅ Figures (รูปภาพ)
- `sleep_stage_distribution.png` - การกระจายของ sleep stages
- `multimodal_signals.png` - ตัวอย่างสัญญาณ
- `hypnogram.png` - Hypnogram
- `model_architecture.png` - สถาปัตยกรรมโมเดล
- `training_history.png` - กราฟการเทรน

### ✅ Models (โมเดลที่เทรนแล้ว)
- `best_model.keras` - โมเดลที่ดีที่สุด
- `final_model.keras` - โมเดลสุดท้าย
- `scalers.pkl` - Data scalers

### ✅ Results (ผลลัพธ์)
- `training_history.csv` - ประวัติการเทรน
- `training_summary.txt` - สรุปผลการเทรน
- `training_log.txt` - Log ทั้งหมด

## 🔧 การแก้ปัญหาเบื้องต้น

### ❌ ImportError: No module named 'xxx'

```bash
pip install xxx
```

### ❌ FileNotFoundError: data/xxx.edf

ดาวน์โหลดข้อมูลจาก:
```bash
cd data/
wget https://physionet.org/files/sleep-edfx/1.0.0/sleep-cassette/SC4001E0-PSG.edf
wget https://physionet.org/files/sleep-edfx/1.0.0/sleep-cassette/SC4001EC-Hypnogram.edf
```

### ❌ Out of Memory

ลด batch size ใน `train.py`:
```python
model.fit(..., batch_size=16)  # ลดจาก 32 เป็น 16
```

## 📊 Performance Metrics

| Metric | Value |
|--------|-------|
| Best Validation Accuracy | **74.19%** |
| Training Accuracy | 97.80% |
| Total Parameters | 476,171 |
| Training Time | ~5-10 minutes |

## 🎓 เรียนรู้เพิ่มเติม

- อ่าน `README.md` สำหรับคู่มือฉบับเต็ม
- อ่าน `conference_paper.md` สำหรับรายละเอียดทางวิชาการ
- อ่าน `PROJECT_SUMMARY.md` สำหรับภาพรวมโปรเจกต์

## 💡 Tips

1. **ดูโค้ดทีละไฟล์** - เริ่มจาก `model.py` เพื่อเข้าใจ architecture
2. **ทดลองปรับ parameters** - ลองเปลี่ยน learning rate, batch size
3. **ใช้ข้อมูลเพิ่ม** - ดาวน์โหลด Sleep-EDF ทั้งหมด (197 subjects)

## 📞 ต้องการความช่วยเหลือ?

อ่านคู่มือฉบับเต็มที่ `README.md` หรือดูตัวอย่างการใช้งานในแต่ละไฟล์ Python

---

**Happy Coding! 🎉**

