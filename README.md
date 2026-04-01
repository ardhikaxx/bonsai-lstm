# 🌿 Bonsai LSTM System

Sistem cerdas berbasis Long Short-Term Memory (LSTM) untuk monitoring dan prediksi kebutuhan penyiraman tanaman bonsai.

## 📋 Deskripsi

Sistem ini menggunakan model LSTM untuk memprediksi kebutuhan penyiraman bonsai berdasarkan data sensor:
- **Suhu** (temperature_c)
- **Kelembapan udara** (humidity_air_pct)
- **Kelembapan tanah** (soil_moisture_pct)

## 🗂️ Struktur Proyek

```
bonsai-lstm/
├── .venv/                    # Virtual environment
├── data/
│   └── raw/
│       └── dataset_bonsai.csv
├── artifacts/                 # File antara
│   ├── data_train.npz
│   ├── data_test.npz
│   ├── scaler_bonsai.pkl
│   ├── label_info.json
│   ├── model_bonsai_lstm.keras
│   ├── training_history.csv
│   ├── predictions.csv
│   └── y_test_labels.csv
├── output/                   # Hasil visualisasi & laporan
├── notebooks/
│   ├── 01_preprocessing.ipynb
│   ├── 02_training.ipynb
│   ├── 03_testing.ipynb
│   └── 04_evaluasi.ipynb
├── requirements.txt
└── README.md
```

## 🚀 Panduan Setup

### 1. Buat Virtual Environment
```bash
# Windows (PowerShell)
python -m venv .venv
.venv\Scripts\Activate.ps1

# Linux/Mac
python -m venv .venv
source .venv/bin/activate
```

### 2. Install Dependensi
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 3. Registrasi Kernel Jupyter
```bash
python -m ipykernel install --user \
    --name=bonsai-lstm \
    --display-name "Python (bonsai-lstm)"
```

### 4. Jalankan Notebook (Urutan Wajib)

1. **`01_preprocessing.ipynb`** - Preprocessing data, normalisasi, labeling
2. **`02_training.ipynb`** - Training model LSTM
3. **`03_testing.ipynb`** - Prediksi pada data testing
4. **`04_evaluasi.ipynb`** - Evaluasi & visualisasi hasil

> ⚠️ **PENTING**: Jalankan notebook secara berurutan. Setiap notebook membutuhkan artefak dari notebook sebelumnya.

## 📊 Target Performa

| Metrik | Target |
|--------|--------|
| Accuracy | ≥ 85% |
| F1-Score | ≥ 80% |
| AUC-ROC | ≥ 85% |
| RMSE | ≤ 5% |
| MAE | ≤ 3% |

## 🔧 Output

- `output/01_confusion_matrix.png`
- `output/02_roc_curve.png`
- `output/03_training_history.png`
- `output/04_prediction_vs_actual.png`
- `output/05_residual_plot.png`
- `output/06_classification_report.csv`
- `output/07_metrics_summary.csv`

## 📝 Lisensi

Proyek ini dibuat untuk implementasi metode LSTM pada sistem monitoring tanaman bonsai.