# 😴 Sleep Detection from Accelerometer Data

A Machine Learning project that classifies **sleep vs. wake states** from wrist accelerometer signals using Python and Scikit-learn.

---

## 📌 Project Overview

Sleep tracking using wearable devices generates raw accelerometer data. This project builds an ML pipeline to automatically detect whether a person is asleep or awake based on wrist movement signals collected over multiple days.

---

## 🎯 Objectives

- Preprocess raw multi-day accelerometer signals
- Engineer meaningful features from raw signal data (anglez, enmo)
- Train and compare multiple ML classifiers
- Evaluate model performance using standard metrics

---

## 🛠️ Tech Stack

| Tool | Purpose |
|------|---------|
| Python | Core programming language |
| Pandas | Data loading and preprocessing |
| NumPy | Numerical operations |
| Matplotlib & Seaborn | Data visualization |
| Scikit-learn | ML model training and evaluation |

---

## 📊 Models Used

| Model | Accuracy |
|-------|----------|
| Random Forest | **82%** ✅ Best |
| K-Nearest Neighbors | 78% |
| Naive Bayes | 74% |

---

## 🔄 ML Pipeline

```
Raw Accelerometer Data
        ↓
Data Preprocessing & Cleaning
        ↓
Exploratory Data Analysis (EDA)
        ↓
Feature Engineering (anglez, enmo)
        ↓
Model Training & Comparison
        ↓
Evaluation (Precision, Recall, F1, Confusion Matrix)
```

---

## 📁 Project Structure

```
sleep-detection/
│
├── sleep_detection.py       # Main ML pipeline script
├── README.md                # Project documentation
└── requirements.txt         # Dependencies
```

---

## ▶️ How to Run

```bash
# 1. Clone the repository
git clone https://github.com/RAHULCHARAN31/sleep-detection.git
cd sleep-detection

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the project
python sleep_detection.py
```

---

## 📈 Results

- Best Model: **Random Forest** with **82% accuracy**
- Key features: `anglez` (angle of wrist), `enmo` (Euclidean Norm Minus One)
- Circadian rhythm patterns clearly visible in EDA visualizations

---

## 👤 Author

**Rahul Charan Erigirala**
- LinkedIn: [linkedin.com/in/rahulcharan46](https://linkedin.com/in/rahulcharan46)
- GitHub: [github.com/RAHULCHARAN31](https://github.com/RAHULCHARAN31)
- Email: rahulcharan6667@gmail.com
