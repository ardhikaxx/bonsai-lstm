# 🌿 RULE-BONSAI: Sistem Cerdas LSTM untuk Monitoring & Prediksi Bonsai

> Dokumen ini mendefinisikan seluruh aturan pengembangan sistem prediksi berbasis
> Long Short-Term Memory (LSTM) untuk tanaman bonsai. Setiap tahap pipeline
> **wajib** ditulis dalam file Jupyter Notebook (`.ipynb`) yang terpisah dan
> dijalankan di dalam Virtual Environment (venv) yang terdedikasi.

---

## 📋 Daftar Isi

1. [Gambaran Umum Sistem](#1-gambaran-umum-sistem)
2. [Struktur Proyek](#2-struktur-proyek)
3. [Rule Setup Virtual Environment (VENV)](#3-rule-setup-virtual-environment-venv)
4. [Rule Kernel Jupyter & Registrasi VENV](#4-rule-kernel-jupyter--registrasi-venv)
5. [Rule Umum Notebook (.ipynb)](#5-rule-umum-notebook-ipynb)
6. [Notebook 01 — Preprocessing](#6-notebook-01--preprocessing)
7. [Notebook 02 — Training](#7-notebook-02--training)
8. [Notebook 03 — Testing](#8-notebook-03--testing)
9. [Notebook 04 — Evaluasi](#9-notebook-04--evaluasi)
10. [Rule Artefak & File Antara](#10-rule-artefak--file-antara)
11. [Rule Logika Pengendalian Atap Otomatis](#11-rule-logika-pengendalian-atap-otomatis)
12. [Rule Reproduksibilitas & Versi](#12-rule-reproduksibilitas--versi)
13. [Referensi Metrik & Formula](#13-referensi-metrik--formula)

---

## 1. Gambaran Umum Sistem

### Tujuan
Membangun sistem cerdas yang mampu memprediksi kebutuhan penyiraman tanaman
bonsai secara otomatis menggunakan model LSTM yang dilatih dari data historis
tiga sensor: **suhu**, **kelembapan udara**, dan **kelembapan tanah**.

### Alur Pipeline

```
dataset_bonsai_lstm.csv
        │
        ▼
┌─────────────────────┐
│  01_preprocessing   │  ← Cleaning, Normalisasi, Sekuens, Split, Labeling
│       .ipynb        │
└──────────┬──────────┘
           │  artifacts: data_train.npz · data_test.npz
           │             scaler_bonsai.pkl · label_info.json
           ▼
┌─────────────────────┐
│   02_training       │  ← Bangun arsitektur LSTM, latih, simpan model
│       .ipynb        │
└──────────┬──────────┘
           │  artifacts: model_bonsai_lstm.keras · training_history.csv
           ▼
┌─────────────────────┐
│   03_testing        │  ← Muat model, jalankan prediksi pada data test
│       .ipynb        │
└──────────┬──────────┘
           │  artifacts: predictions.csv · y_test_labels.csv
           ▼
┌─────────────────────┐
│   04_evaluasi       │  ← Hitung semua metrik, buat seluruh visualisasi
│       .ipynb        │
└─────────────────────┘
           │  output/: confusion_matrix.png · roc_curve.png
           │           training_history.png · prediction_vs_actual.png
           │           residual_plot.png · classification_report.csv
           │           metrics_summary.csv
```

### Struktur Data Sensor

| Kolom | Tipe | Satuan | Deskripsi |
|---|---|---|---|
| `timestamp` | datetime | — | Waktu pengukuran (interval 30 menit) |
| `temperature_c` | float64 | °C | Suhu udara sekitar bonsai |
| `humidity_air_pct` | float64 | % | Kelembapan udara relatif |
| `soil_moisture_pct` | float64 | % | Kelembapan tanah pot bonsai |

**Karakteristik Dataset:**
- Rentang waktu : 2025-01-01 s.d. 2025-03-31
- Total sampel  : 4.320 baris (48 titik/hari)
- Frekuensi     : setiap 30 menit

---

## 2. Struktur Proyek

Seluruh file proyek **wajib** mengikuti hierarki direktori berikut.
Tidak boleh ada file notebook di luar folder `notebooks/`.

```
bonsai-lstm/                          ← root proyek
│
├── .venv/                            ← Virtual environment
│
├── data/
│   └── raw/
│       └── dataset_bonsai.csv   ← Data mentah asli (READ-ONLY)
│
├── artifacts/                        ← File antara antar notebook
│   ├── data_train.npz
│   ├── data_test.npz
│   ├── scaler_bonsai.pkl
│   ├── label_info.json
│   ├── model_bonsai_lstm.keras
│   └── training_history.csv
│
├── output/                           ← Seluruh hasil visualisasi & laporan
│   ├── 01_confusion_matrix.png
│   ├── 02_roc_curve.png
│   ├── 03_training_history.png
│   ├── 04_prediction_vs_actual.png
│   ├── 05_residual_plot.png
│   ├── 06_classification_report.csv
│   └── 07_metrics_summary.csv
│
├── notebooks/
│   ├── 01_preprocessing.ipynb
│   ├── 02_training.ipynb
│   ├── 03_testing.ipynb
│   └── 04_evaluasi.ipynb
│
├── requirements.txt                  ← Daftar dependensi dengan versi terpaku
└── README.md                         ← Panduan singkat setup & urutan eksekusi
```

### RULE-STRUCT-01: Immutabilitas Data Mentah
```
- Folder data/raw/ bersifat READ-ONLY.
- Tidak ada notebook yang boleh menulis, memodifikasi, atau menghapus
  file di dalam data/raw/.
- Semua operasi membaca dataset menggunakan path relatif:
    DATA_PATH = "../data/raw/dataset_bonsai_lstm.csv"
```

### RULE-STRUCT-02: Path Relatif
```
- Semua path di dalam notebook WAJIB menggunakan path relatif terhadap
  root proyek (bonsai-lstm/).
- Dilarang menggunakan path absolut seperti C:/Users/... atau /home/user/...
```

### RULE-STRUCT-03: Pemisahan Artefak & Output
```
- artifacts/ → file biner/data yang dibutuhkan notebook berikutnya (NPZ, PKL, KERAS)
- output/    → file hasil akhir untuk manusia (PNG, CSV laporan)
- Kedua folder dibuat otomatis oleh notebook jika belum ada:
    os.makedirs("../artifacts", exist_ok=True)
    os.makedirs("../output", exist_ok=True)
```

---

## 3. Rule Setup Virtual Environment (VENV)

### RULE-VENV-01: Pembuatan Lingkungan Virtual

Virtual environment **wajib** dibuat satu kali di root proyek sebelum apapun
dijalankan. Python versi minimum yang diizinkan adalah **3.9**.

```bash
# 1. Masuk ke root proyek
cd bonsai-lstm/

# 2. Buat venv dengan nama .venv
python -m venv .venv

# 3. Aktivasi (pilih sesuai OS)

#    Windows — Command Prompt
.venv\Scripts\activate.bat

#    Windows — PowerShell
.venv\Scripts\Activate.ps1

#    macOS / Linux
source .venv/bin/activate

# 4. Verifikasi — prompt harus menampilkan (.venv)
python --version
```

### RULE-VENV-02: File requirements.txt

File `requirements.txt` **wajib** ada di root proyek dengan versi yang
terpaku (pinned) untuk menjamin reproduksibilitas antar mesin.

```
# requirements.txt — Sistem LSTM Bonsai
# Python >= 3.9

tensorflow==2.15.0
scikit-learn==1.4.0
pandas==2.2.0
numpy==1.26.3
matplotlib==3.8.2
seaborn==0.13.2
joblib==1.3.2
ipykernel==6.29.0
jupyter==1.0.0
notebook==7.0.7
ipywidgets==8.1.1
```

### RULE-VENV-03: Instalasi Dependensi

```bash
# Pastikan venv sudah aktif (ada tanda (.venv) di prompt)
pip install --upgrade pip
pip install -r requirements.txt

# Verifikasi instalasi
pip list
```

### RULE-VENV-04: File .gitignore

Tambahkan baris berikut ke `.gitignore` di root proyek:

```gitignore
# Virtual environment — JANGAN di-commit
.venv/
venv/
ENV/

# Artefak model & data olahan
artifacts/

# Cache Python & Jupyter
__pycache__/
*.pyc
*.pyo
.ipynb_checkpoints/

# Output visualisasi (opsional)
output/*.png
```

---

## 4. Rule Kernel Jupyter & Registrasi VENV

### RULE-KERNEL-01: Registrasi Kernel ke Jupyter

Setelah venv aktif dan dependensi terinstall, daftarkan venv sebagai
kernel Jupyter agar seluruh notebook menggunakan lingkungan yang sama.

```bash
# Pastikan venv aktif
python -m ipykernel install --user \
    --name=bonsai-lstm \
    --display-name "Python (bonsai-lstm)"
```

### RULE-KERNEL-02: Pemilihan Kernel di Notebook

```
Setiap kali membuka notebook di Jupyter:
  Kernel → Change Kernel → Python (bonsai-lstm)

Di VS Code:
  Klik nama kernel di kanan atas → Pilih "Python (bonsai-lstm)"
```

### RULE-KERNEL-03: Cell Verifikasi Lingkungan (Wajib)

**Cell pertama** di setiap notebook harus berisi blok verifikasi ini:

```python
# ── VERIFIKASI LINGKUNGAN ──────────────────────────────────────────────
import sys, os

assert ".venv" in sys.executable or "venv" in sys.executable, (
    "⛔ Kernel bukan dari .venv!\n"
    "Ganti kernel ke: Python (bonsai-lstm)\n"
    f"Kernel saat ini: {sys.executable}"
)
print("✅ Kernel  :", sys.executable)
print("✅ Python  :", sys.version)
# ──────────────────────────────────────────────────────────────────────
```

---

## 5. Rule Umum Notebook (.ipynb)

### RULE-NB-01: Struktur Wajib Setiap Notebook

Setiap file `.ipynb` harus memiliki sel dalam urutan berikut:

```
[Cell 1]  — Markdown : Header tabel (Judul, Tujuan, Input, Output, Urutan)
[Cell 2]  — Code     : Verifikasi lingkungan (RULE-KERNEL-03)
[Cell 3]  — Code     : Import library, konstanta global, seed (RULE-REPRO-01)
[Cell 4+] — Code     : Logika utama sesuai tujuan notebook
[Cell -2] — Code     : Simpan artefak / output
[Cell -1] — Markdown : Ringkasan hasil & instruksi langkah selanjutnya
```

### RULE-NB-02: Header Markdown Wajib (Cell 1)

```markdown
# 🌿 [Nama Tahap] — Sistem LSTM Bonsai

| Item       | Detail                                      |
|------------|---------------------------------------------|
| **File**   | `notebooks/0X_nama.ipynb`                   |
| **Tujuan** | [deskripsi singkat tujuan notebook ini]     |
| **Input**  | [file/data yang dibutuhkan sebagai input]   |
| **Output** | [file/artefak yang dihasilkan]              |
| **Urutan** | Jalankan setelah: `0(X-1)_sebelumnya.ipynb` |
```

### RULE-NB-03: Eksekusi Top-to-Bottom
```
- Seluruh sel wajib dapat dijalankan dari atas ke bawah tanpa error
  menggunakan: Kernel → Restart & Run All
- Dilarang mengandalkan state variabel dari sesi kernel sebelumnya.
- Setelah menulis/mengubah notebook, wajib lakukan Restart & Run All
  sebagai verifikasi akhir sebelum menyimpan file.
```

### RULE-NB-04: Seed Reproduksibilitas (Cell 3)
```
Wajib ada di Cell 3 di SETIAP notebook:
```

```python
import random, os
import numpy as np
import tensorflow as tf

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)
os.environ["PYTHONHASHSEED"] = str(SEED)
print(f"[SEED] Global random seed = {SEED}")
```

### RULE-NB-05: Penamaan File Notebook
```
Format wajib : XX_nama_singkat.ipynb
  XX         → nomor urut dua digit (01, 02, 03, 04)
  nama       → huruf kecil, kata dipisahkan underscore

Benar  : 01_preprocessing.ipynb / 02_training.ipynb
Salah  : Preprocessing.ipynb / notebook1.ipynb / pre processing.ipynb
```

### RULE-NB-06: Validasi Artefak di Awal Notebook (Cell 3 atau 4)

Setiap notebook kecuali `01_preprocessing.ipynb` **wajib** memvalidasi
keberadaan artefak yang dibutuhkan sebelum menjalankan logika utama:

```python
REQUIRED_ARTIFACTS = [
    "../artifacts/data_train.npz",
    # (sesuaikan daftar ini per notebook)
]
missing = [f for f in REQUIRED_ARTIFACTS if not os.path.exists(f)]
assert not missing, (
    f"⛔ Artefak tidak ditemukan: {missing}\n"
    "Jalankan notebook sebelumnya terlebih dahulu."
)
print("✅ Semua artefak yang dibutuhkan tersedia.")
```

---

## 6. Notebook 01 — Preprocessing

**File:** `notebooks/01_preprocessing.ipynb`

**Tujuan:** Membersihkan data mentah sensor, melakukan normalisasi
Min-Max, membentuk sekuens waktu, membuat label klasifikasi penyiraman,
membagi data training/testing, lalu menyimpan semua artefak.

**Input:**  `data/raw/dataset_bonsai_lstm.csv`

**Output:**
- `artifacts/data_train.npz` — X_train, y_train_cls, y_train_reg
- `artifacts/data_test.npz`  — X_test, y_test_cls, y_test_reg
- `artifacts/scaler_bonsai.pkl` — MinMaxScaler yang sudah di-fit
- `artifacts/label_info.json`   — distribusi label, threshold, metadata

---

### RULE-PRE-01: Konstanta Global

```python
DATA_PATH      = "../data/raw/dataset_bonsai_lstm.csv"
ARTIFACTS_DIR  = "../artifacts"
TRAIN_RATIO    = 0.80          # 80% training, 20% testing
LOOK_BACK      = 24            # 24 langkah × 30 menit = 12 jam historis
SOIL_THRESHOLD = 60.0          # % batas keputusan penyiraman
FEATURES       = ["temperature_c", "humidity_air_pct", "soil_moisture_pct"]

# Batas valid per sensor
BOUNDS = {
    "temperature_c"     : (10.0, 45.0),
    "humidity_air_pct"  : (0.0,  100.0),
    "soil_moisture_pct" : (0.0,  100.0),
}
STAGNANT_WINDOW = 12  # rolling std = 0 selama 12 titik → anomali sensor
```

---

### RULE-PRE-02: Pembersihan Data (Data Cleaning)

#### RULE-CLEAN-01 — Parsing & Pengurutan Timestamp
```python
df["timestamp"] = pd.to_datetime(df["timestamp"])
df = df.sort_values("timestamp").reset_index(drop=True)
print(f"[CLEAN-01] Timestamp diparse. Rentang: {df['timestamp'].iloc[0]} → {df['timestamp'].iloc[-1]}")
```

#### RULE-CLEAN-02 — Hapus Nilai Kosong (Missing Values)
```python
n_null = df.isnull().sum().sum()
df = df.dropna().reset_index(drop=True)
print(f"[CLEAN-02] Nilai null dihapus: {n_null} | Sisa baris: {len(df)}")
```

#### RULE-CLEAN-03 — Hapus Outlier Nilai Ekstrem
```
Aturan batas valid setiap kolom sensor:
  temperature_c      : 10°C  ≤ nilai ≤ 45°C
  humidity_air_pct   : 0%    < nilai ≤ 100%
  soil_moisture_pct  : 0%    < nilai ≤ 100%

Tindakan: baris yang melanggar batas ini DIHAPUS dari dataset.
```
```python
mask_valid = pd.Series([True] * len(df))
for col, (lo, hi) in BOUNDS.items():
    invalid = ~df[col].between(lo, hi, inclusive="both")
    print(f"[CLEAN-03] {col}: {invalid.sum()} outlier ditemukan")
    mask_valid &= ~invalid
df = df[mask_valid].reset_index(drop=True)
print(f"[CLEAN-03] Sisa setelah hapus outlier: {len(df)}")
```

#### RULE-CLEAN-04 — Hapus Segmen Stagnan (Flat Signal)
```
Definisi: rolling std = 0 selama STAGNANT_WINDOW langkah berturut-turut
Makna   : sensor tidak berubah ≥ 6 jam → indikasi kerusakan hardware
Tindakan: seluruh titik dalam jendela stagnan DIHAPUS
```
```python
stagnant_mask = pd.Series([False] * len(df))
for col in FEATURES:
    rolling_std = df[col].rolling(window=STAGNANT_WINDOW).std()
    stagnant_mask |= (rolling_std == 0)
n_stagnant = stagnant_mask.sum()
df = df[~stagnant_mask].reset_index(drop=True)
print(f"[CLEAN-04] Titik stagnan dihapus: {n_stagnant} | Sisa: {len(df)}")
```

#### RULE-CLEAN-05 — Hapus Duplikat Timestamp
```python
n_dup = df.duplicated(subset="timestamp").sum()
df = df.drop_duplicates(subset="timestamp", keep="first").reset_index(drop=True)
print(f"[CLEAN-05] Duplikat timestamp dihapus: {n_dup} | Sisa: {len(df)}")
```

---

### RULE-PRE-03: Split Data Training / Testing

```
WAJIB dilakukan SEBELUM normalisasi untuk mencegah data leakage.
WAJIB menggunakan split kronologis — DILARANG random split.
  Training : 80% pertama secara kronologis
  Testing  : 20% terakhir secara kronologis
```

```python
split_idx = int(len(df) * TRAIN_RATIO)
df_train  = df.iloc[:split_idx].reset_index(drop=True)
df_test   = df.iloc[split_idx:].reset_index(drop=True)

print(f"[SPLIT] Training : {len(df_train)} baris "
      f"({df_train['timestamp'].iloc[0]} → {df_train['timestamp'].iloc[-1]})")
print(f"[SPLIT] Testing  : {len(df_test)} baris "
      f"({df_test['timestamp'].iloc[0]} → {df_test['timestamp'].iloc[-1]})")
```

---

### RULE-PRE-04: Normalisasi Min-Max Scaling

$$X_{norm} = \frac{X - X_{min}}{X_{max} - X_{min}}$$

```
Aturan kritis:
  - Scaler di-fit HANYA pada df_train (bukan seluruh df).
  - df_test dinormalisasi menggunakan parameter dari training (transform saja).
  - Scaler disimpan ke artifacts/scaler_bonsai.pkl untuk dipakai ulang.
  - Rentang output normalisasi: [0, 1]
```

```python
from sklearn.preprocessing import MinMaxScaler
import joblib, os

os.makedirs(ARTIFACTS_DIR, exist_ok=True)

scaler       = MinMaxScaler(feature_range=(0, 1))
train_scaled = scaler.fit_transform(df_train[FEATURES])   # fit + transform
test_scaled  = scaler.transform(df_test[FEATURES])        # transform saja

joblib.dump(scaler, f"{ARTIFACTS_DIR}/scaler_bonsai.pkl")
print(f"[NORM] Scaler disimpan → artifacts/scaler_bonsai.pkl")
print(f"[NORM] Min per fitur   : {dict(zip(FEATURES, scaler.data_min_))}")
print(f"[NORM] Max per fitur   : {dict(zip(FEATURES, scaler.data_max_))}")
```

---

### RULE-PRE-05: Pembentukan Sekuens Waktu

```
Setiap sampel input  : LOOK_BACK langkah × 3 fitur → shape (24, 3)
Setiap target output : nilai soil_moisture_pct (kolom indeks 2) pada t+1
Fungsi create_sequences HARUS identik di semua notebook yang memakainya.
```

```python
def create_sequences(data: np.ndarray, look_back: int):
    """
    Membentuk pasangan input-output untuk LSTM.
    X[i] = data[i : i+look_back, :]     shape → (look_back, n_features)
    y[i] = data[i+look_back, 2]         indeks 2 = soil_moisture_pct
    """
    X, y = [], []
    for i in range(look_back, len(data)):
        X.append(data[i - look_back : i, :])
        y.append(data[i, 2])
    return np.array(X), np.array(y)

X_train, y_train_reg = create_sequences(train_scaled, LOOK_BACK)
X_test,  y_test_reg  = create_sequences(test_scaled,  LOOK_BACK)

print(f"[SEQ] X_train shape : {X_train.shape}  → (sampel, look_back, fitur)")
print(f"[SEQ] X_test  shape : {X_test.shape}")
```

---

### RULE-PRE-06: Pelabelan Klasifikasi Penyiraman

```
Inverse-transform y (skala normal) → skala asli (%) sebelum membuat label.

Label biner berdasarkan threshold SOIL_THRESHOLD = 60%:
  1 → BUTUH PENYIRAMAN   (soil_moisture_pct < 60%)
  0 → TIDAK BUTUH SIRAM  (soil_moisture_pct ≥ 60%)

Tingkat keparahan (untuk referensi & dokumentasi):
  KRITIS  : soil < 40%         → siram segera
  RENDAH  : 40% ≤ soil < 60%  → perlu disiram
  OPTIMAL : 60% ≤ soil < 85%  → kondisi baik
  TINGGI  : soil ≥ 85%        → terlalu lembab, tunda penyiraman
```

```python
def inverse_soil(y_norm: np.ndarray, scaler, n_features: int) -> np.ndarray:
    """Kembalikan nilai normalized soil ke skala asli (%)."""
    dummy = np.zeros((len(y_norm), n_features))
    dummy[:, 2] = y_norm                          # indeks 2 = soil_moisture_pct
    return scaler.inverse_transform(dummy)[:, 2]

def create_labels(soil_values: np.ndarray, threshold: float) -> np.ndarray:
    """Label 1 = butuh siram, 0 = tidak butuh siram."""
    return (soil_values < threshold).astype(int)

y_train_orig = inverse_soil(y_train_reg, scaler, len(FEATURES))
y_test_orig  = inverse_soil(y_test_reg,  scaler, len(FEATURES))
y_train_cls  = create_labels(y_train_orig, SOIL_THRESHOLD)
y_test_cls   = create_labels(y_test_orig,  SOIL_THRESHOLD)

print(f"[LABEL] TRAIN → Tidak Siram (0): {(y_train_cls==0).sum()} | Siram (1): {(y_train_cls==1).sum()}")
print(f"[LABEL] TEST  → Tidak Siram (0): {(y_test_cls==0).sum()}  | Siram (1): {(y_test_cls==1).sum()}")

# Imbalance check
rasio = max((y_train_cls==0).sum(), (y_train_cls==1).sum()) / \
        max(min((y_train_cls==0).sum(), (y_train_cls==1).sum()), 1)
print(f"[LABEL] Rasio kelas (mayor:minor) = {rasio:.2f}x {'⚠️ Imbalanced' if rasio > 4 else '✅ Balanced'}")
```

---

### RULE-PRE-07: Simpan Semua Artefak Preprocessing

```python
import json

# Array numpy
np.savez_compressed(
    f"{ARTIFACTS_DIR}/data_train.npz",
    X_train=X_train, y_train_cls=y_train_cls, y_train_reg=y_train_orig
)
np.savez_compressed(
    f"{ARTIFACTS_DIR}/data_test.npz",
    X_test=X_test, y_test_cls=y_test_cls, y_test_reg=y_test_orig
)

# Metadata label
label_info = {
    "soil_threshold" : SOIL_THRESHOLD,
    "look_back"      : LOOK_BACK,
    "train_ratio"    : TRAIN_RATIO,
    "features"       : FEATURES,
    "n_features"     : len(FEATURES),
    "train_label_0"  : int((y_train_cls == 0).sum()),
    "train_label_1"  : int((y_train_cls == 1).sum()),
    "test_label_0"   : int((y_test_cls  == 0).sum()),
    "test_label_1"   : int((y_test_cls  == 1).sum()),
}
with open(f"{ARTIFACTS_DIR}/label_info.json", "w") as f:
    json.dump(label_info, f, indent=2)

print("[SAVE] ✅ artifacts/data_train.npz")
print("[SAVE] ✅ artifacts/data_test.npz")
print("[SAVE] ✅ artifacts/scaler_bonsai.pkl")
print("[SAVE] ✅ artifacts/label_info.json")
```

---

## 7. Notebook 02 — Training

**File:** `notebooks/02_training.ipynb`

**Tujuan:** Membangun arsitektur model LSTM, melatih menggunakan data
training yang sudah diproses, menyimpan model terbaik beserta histori
pelatihan sebagai artefak.

**Input:**
- `artifacts/data_train.npz`
- `artifacts/label_info.json`

**Output:**
- `artifacts/model_bonsai_lstm.keras`
- `artifacts/training_history.csv`

---

### RULE-TRAIN-01: Konstanta Training

```python
ARTIFACTS_DIR  = "../artifacts"
MODEL_PATH     = f"{ARTIFACTS_DIR}/model_bonsai_lstm.keras"
HISTORY_PATH   = f"{ARTIFACTS_DIR}/training_history.csv"

EPOCHS         = 50
BATCH_SIZE     = 32
LEARNING_RATE  = 0.001
VAL_SPLIT      = 0.10      # 10% dari training untuk validasi internal
PATIENCE       = 10        # EarlyStopping: hentikan jika tidak membaik
```

---

### RULE-TRAIN-02: Muat Artefak dari Preprocessing

```python
import numpy as np, json

train_data  = np.load(f"{ARTIFACTS_DIR}/data_train.npz")
X_train     = train_data["X_train"]
y_train     = train_data["y_train_cls"]
y_train_reg = train_data["y_train_reg"]

with open(f"{ARTIFACTS_DIR}/label_info.json") as f:
    label_info = json.load(f)

LOOK_BACK  = label_info["look_back"]
N_FEATURES = label_info["n_features"]

print(f"[LOAD] X_train : {X_train.shape}")
print(f"[LOAD] y_train : {y_train.shape}  (0={( y_train==0).sum()}, 1={(y_train==1).sum()})")
```

---

### RULE-TRAIN-03: Penanganan Class Imbalance

```
- Gunakan class_weight='balanced' → dihitung otomatis oleh sklearn.
- Diterapkan sebagai argumen class_weight pada model.fit().
- JANGAN menghitung class weight secara manual.
```

```python
from sklearn.utils.class_weight import compute_class_weight

classes  = np.unique(y_train)
cw_arr   = compute_class_weight("balanced", classes=classes, y=y_train)
cw_dict  = dict(zip(classes.tolist(), cw_arr.tolist()))

print(f"[CW] Class weights yang diterapkan: {cw_dict}")
```

---

### RULE-TRAIN-04: Arsitektur Model LSTM

```
Layer 1 : LSTM(64,  return_sequences=True)  → tangkap pola jangka panjang
Layer 2 : Dropout(0.2)                      → regularisasi
Layer 3 : LSTM(32,  return_sequences=False) → ekstrak representasi akhir
Layer 4 : Dropout(0.2)                      → regularisasi
Layer 5 : Dense(16, activation='relu')      → fully connected intermediate
Layer 6 : Dense(1,  activation='sigmoid')   → output biner [0, 1]

Loss     : binary_crossentropy
Optimizer: Adam(lr=0.001)
Metrics  : [accuracy, AUC]
```

```python
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

def build_model(look_back: int, n_features: int) -> tf.keras.Model:
    model = Sequential([
        LSTM(64, return_sequences=True,  input_shape=(look_back, n_features)),
        Dropout(0.2),
        LSTM(32, return_sequences=False),
        Dropout(0.2),
        Dense(16, activation="relu"),
        Dense(1,  activation="sigmoid"),
    ], name="bonsai_lstm")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss="binary_crossentropy",
        metrics=["accuracy", tf.keras.metrics.AUC(name="auc")],
    )
    return model

model = build_model(LOOK_BACK, N_FEATURES)
model.summary()
```

---

### RULE-TRAIN-05: Callbacks Wajib

```python
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, CSVLogger

callbacks = [
    # Hentikan jika val_loss tidak membaik selama PATIENCE epoch
    EarlyStopping(
        monitor="val_loss",
        patience=PATIENCE,
        restore_best_weights=True,
        verbose=1
    ),
    # Simpan hanya bobot terbaik
    ModelCheckpoint(
        filepath=MODEL_PATH,
        monitor="val_loss",
        save_best_only=True,
        verbose=1
    ),
    # Catat histori training ke CSV secara otomatis
    CSVLogger(HISTORY_PATH, separator=",", append=False),
]
```

---

### RULE-TRAIN-06: Eksekusi Pelatihan

```python
history = model.fit(
    X_train, y_train,
    epochs           = EPOCHS,
    batch_size       = BATCH_SIZE,
    validation_split = VAL_SPLIT,
    callbacks        = callbacks,
    class_weight     = cw_dict,
    verbose          = 1,
)
print(f"\n[TRAIN] ✅ Model disimpan → {MODEL_PATH}")
print(f"[TRAIN] ✅ Histori disimpan → {HISTORY_PATH}")
```

---

### RULE-TRAIN-07: Tampilkan Ringkasan Training

```python
import pandas as pd
hist_df    = pd.read_csv(HISTORY_PATH)
best_epoch = hist_df["val_loss"].idxmin() + 1

print(f"\n[SUMMARY] Epoch terbaik    : {best_epoch} / {len(hist_df)}")
print(f"[SUMMARY] Val Loss terbaik : {hist_df['val_loss'].min():.4f}")
print(f"[SUMMARY] Val Accuracy     : {hist_df['val_accuracy'].iloc[best_epoch-1]:.4f}")
print(f"[SUMMARY] Val AUC          : {hist_df['val_auc'].iloc[best_epoch-1]:.4f}")
```

---

## 8. Notebook 03 — Testing

**File:** `notebooks/03_testing.ipynb`

**Tujuan:** Memuat model yang sudah dilatih, menjalankan prediksi pada
data testing, menyimpan hasil prediksi (probabilitas, kelas, nilai
estimasi kelembaban) untuk digunakan di notebook evaluasi.

**Input:**
- `artifacts/model_bonsai_lstm.keras`
- `artifacts/data_test.npz`
- `artifacts/scaler_bonsai.pkl`
- `artifacts/label_info.json`

**Output:**
- `artifacts/predictions.csv`    — prob, kelas prediksi, estimasi soil %
- `artifacts/y_test_labels.csv`  — label aktual & nilai aktual soil %

---

### RULE-TEST-01: Konstanta Testing

```python
ARTIFACTS_DIR = "../artifacts"
SOIL_MIN      = 0.0
SOIL_MAX      = 100.0
```

---

### RULE-TEST-02: Muat Semua Artefak

```python
import numpy as np, pandas as pd, json, joblib, os
import tensorflow as tf

# Validasi keberadaan artefak (RULE-NB-06)
REQUIRED = [
    f"{ARTIFACTS_DIR}/model_bonsai_lstm.keras",
    f"{ARTIFACTS_DIR}/data_test.npz",
    f"{ARTIFACTS_DIR}/scaler_bonsai.pkl",
    f"{ARTIFACTS_DIR}/label_info.json",
]
missing = [f for f in REQUIRED if not os.path.exists(f)]
assert not missing, f"⛔ Artefak tidak ada: {missing}"
print("✅ Semua artefak tersedia.")

# Muat data & model
test_data  = np.load(f"{ARTIFACTS_DIR}/data_test.npz")
X_test     = test_data["X_test"]
y_test_cls = test_data["y_test_cls"]
y_test_reg = test_data["y_test_reg"]

model  = tf.keras.models.load_model(f"{ARTIFACTS_DIR}/model_bonsai_lstm.keras")
scaler = joblib.load(f"{ARTIFACTS_DIR}/scaler_bonsai.pkl")

with open(f"{ARTIFACTS_DIR}/label_info.json") as f:
    label_info = json.load(f)

print(f"[LOAD] X_test shape : {X_test.shape}")
print(f"[LOAD] Model        : {model.name}")
```

---

### RULE-TEST-03: Menjalankan Prediksi

```python
# Probabilitas output sigmoid ∈ [0, 1]
y_pred_prob  = model.predict(X_test, verbose=0).flatten()

# Label biner dengan threshold 0.5
y_pred_class = (y_pred_prob >= 0.5).astype(int)

# Estimasi nilai kelembaban tanah dari probabilitas (konversi linier)
y_pred_soil  = y_pred_prob * (SOIL_MAX - SOIL_MIN) + SOIL_MIN

print(f"[PREDICT] Total prediksi       : {len(y_pred_prob)}")
print(f"[PREDICT] Prediksi Siram (1)   : {y_pred_class.sum()}")
print(f"[PREDICT] Prediksi Tidak (0)   : {(y_pred_class==0).sum()}")
print(f"[PREDICT] Probabilitas rata2   : {y_pred_prob.mean():.4f}")
```

---

### RULE-TEST-04: Simpan Hasil Prediksi

```python
pred_df = pd.DataFrame({
    "y_pred_prob"    : y_pred_prob,
    "y_pred_class"   : y_pred_class,
    "y_pred_soil_pct": y_pred_soil,
})
pred_df.to_csv(f"{ARTIFACTS_DIR}/predictions.csv", index=False)

actual_df = pd.DataFrame({
    "y_true_class"   : y_test_cls,
    "y_true_soil_pct": y_test_reg,
})
actual_df.to_csv(f"{ARTIFACTS_DIR}/y_test_labels.csv", index=False)

print("[SAVE] ✅ artifacts/predictions.csv")
print("[SAVE] ✅ artifacts/y_test_labels.csv")
```

---

### RULE-TEST-05: Sanity Check Hasil Prediksi

```python
check_df = pd.concat([actual_df, pred_df], axis=1)
print("\n[SANITY CHECK] 10 sampel pertama:")
print(check_df.head(10).to_string(index=False))

correct = (check_df["y_true_class"] == check_df["y_pred_class"]).sum()
total   = len(check_df)
print(f"\n[SANITY] Prediksi benar: {correct}/{total} = {correct/total*100:.2f}%")
```

---

## 9. Notebook 04 — Evaluasi

**File:** `notebooks/04_evaluasi.ipynb`

**Tujuan:** Menghitung seluruh metrik evaluasi (klasifikasi & regresi)
dan menghasilkan semua visualisasi output yang membuktikan performa sistem.

**Input:**
- `artifacts/predictions.csv`
- `artifacts/y_test_labels.csv`
- `artifacts/training_history.csv`

**Output (semua ke `output/`):**
- `01_confusion_matrix.png`
- `02_roc_curve.png`
- `03_training_history.png`
- `04_prediction_vs_actual.png`
- `05_residual_plot.png`
- `06_classification_report.csv`
- `07_metrics_summary.csv`

---

### RULE-EVAL-01: Konstanta Evaluasi

```python
ARTIFACTS_DIR = "../artifacts"
OUTPUT_DIR    = "../output"
DPI           = 150
N_SHOW        = 200    # jumlah sampel ditampilkan di plot prediksi

# Batas performa minimum yang harus dipenuhi
TARGET = {
    "accuracy" : 0.85,
    "f1"       : 0.80,
    "auc_roc"  : 0.85,
    "rmse_pct" : 5.0,
    "mae_pct"  : 3.0,
}
```

---

### RULE-EVAL-02: Muat Data Evaluasi

```python
import numpy as np, pandas as pd, os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix, classification_report,
    roc_curve, auc, accuracy_score,
    mean_squared_error, mean_absolute_error, f1_score
)

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Validasi artefak
REQUIRED = [
    f"{ARTIFACTS_DIR}/predictions.csv",
    f"{ARTIFACTS_DIR}/y_test_labels.csv",
    f"{ARTIFACTS_DIR}/training_history.csv",
]
missing = [f for f in REQUIRED if not os.path.exists(f)]
assert not missing, f"⛔ Artefak tidak ada: {missing}"

pred_df   = pd.read_csv(f"{ARTIFACTS_DIR}/predictions.csv")
actual_df = pd.read_csv(f"{ARTIFACTS_DIR}/y_test_labels.csv")
hist_df   = pd.read_csv(f"{ARTIFACTS_DIR}/training_history.csv")

y_true       = actual_df["y_true_class"].values
y_true_soil  = actual_df["y_true_soil_pct"].values
y_pred_prob  = pred_df["y_pred_prob"].values
y_pred_class = pred_df["y_pred_class"].values
y_pred_soil  = pred_df["y_pred_soil_pct"].values
print("✅ Data evaluasi dimuat.")
```

---

### RULE-EVAL-03: Hitung Metrik Regresi

**RMSE:**
$$RMSE = \sqrt{\frac{1}{n}\sum_{i=1}^{n}(y_i - \hat{y}_i)^2}$$

**MAE:**
$$MAE = \frac{1}{n}\sum_{i=1}^{n}|y_i - \hat{y}_i|$$

```python
rmse = np.sqrt(mean_squared_error(y_true_soil, y_pred_soil))
mae  = mean_absolute_error(y_true_soil, y_pred_soil)

print(f"[RMSE] {rmse:.4f}% | Target ≤ {TARGET['rmse_pct']}% → {'✅ PASS' if rmse<=TARGET['rmse_pct'] else '❌ FAIL'}")
print(f"[MAE ] {mae:.4f}%  | Target ≤ {TARGET['mae_pct']}%  → {'✅ PASS' if mae<=TARGET['mae_pct']  else '❌ FAIL'}")
```

---

### RULE-EVAL-04: Hitung Metrik Klasifikasi

```python
acc     = accuracy_score(y_true, y_pred_class)
f1      = f1_score(y_true, y_pred_class, average="weighted")
fpr, tpr, _ = roc_curve(y_true, y_pred_prob)
roc_auc = auc(fpr, tpr)

print(f"[ACC    ] {acc:.4f} | Target ≥ {TARGET['accuracy']}  → {'✅ PASS' if acc>=TARGET['accuracy'] else '❌ FAIL'}")
print(f"[F1     ] {f1:.4f} | Target ≥ {TARGET['f1']}       → {'✅ PASS' if f1>=TARGET['f1'] else '❌ FAIL'}")
print(f"[AUC-ROC] {roc_auc:.4f} | Target ≥ {TARGET['auc_roc']} → {'✅ PASS' if roc_auc>=TARGET['auc_roc'] else '❌ FAIL'}")
```

---

### RULE-EVAL-05: Confusion Matrix

```
Output : output/01_confusion_matrix.png
Format : Heatmap 2×2 dengan anotasi TP/TN/FP/FN + ringkasan metrik
```

```python
def plot_confusion_matrix(y_true, y_pred, path, dpi):
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    prec = tp/(tp+fp) if (tp+fp) else 0
    rec  = tp/(tp+fn) if (tp+fn) else 0
    f1_  = 2*prec*rec/(prec+rec) if (prec+rec) else 0

    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Tidak Siram","Siram"],
                yticklabels=["Tidak Siram","Siram"],
                linewidths=1, linecolor="gray", ax=ax, annot_kws={"size":14})
    ax.set_xlabel("Prediksi", fontsize=12)
    ax.set_ylabel("Aktual",   fontsize=12)
    ax.set_title("Confusion Matrix — LSTM Bonsai", fontsize=14, fontweight="bold")
    info = (f"TP={tp}  FP={fp}  FN={fn}  TN={tn}\n"
            f"Precision={prec:.4f}  Recall={rec:.4f}  F1-Score={f1_:.4f}")
    ax.text(0.5, -0.18, info, transform=ax.transAxes, ha="center", fontsize=10,
            bbox=dict(boxstyle="round,pad=0.4", facecolor="lightyellow", edgecolor="gray"))
    plt.tight_layout()
    plt.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close()
    print(f"[VIZ] ✅ {path}")

plot_confusion_matrix(y_true, y_pred_class, f"{OUTPUT_DIR}/01_confusion_matrix.png", DPI)
```

---

### RULE-EVAL-06: ROC Curve

```
Output : output/02_roc_curve.png
Format : Kurva AUC-ROC dengan area shading & nilai AUC di legend
```

```python
def plot_roc(fpr, tpr, roc_auc, path, dpi):
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(fpr, tpr, color="steelblue", lw=2, label=f"ROC (AUC = {roc_auc:.4f})")
    ax.plot([0,1],[0,1], color="gray", linestyle="--", lw=1.5, label="Random Classifier")
    ax.fill_between(fpr, tpr, alpha=0.15, color="steelblue")
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate",  fontsize=12)
    ax.set_title("ROC Curve — LSTM Bonsai", fontsize=14, fontweight="bold")
    ax.legend(loc="lower right", fontsize=11); ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close()
    print(f"[VIZ] ✅ {path}")

plot_roc(fpr, tpr, roc_auc, f"{OUTPUT_DIR}/02_roc_curve.png", DPI)
```

---

### RULE-EVAL-07: Training History

```
Output : output/03_training_history.png
Format : 2 subplot — Loss per epoch (kiri), Accuracy per epoch (kanan)
```

```python
def plot_history(hist_df, path, dpi):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("Training History — LSTM Bonsai", fontsize=14, fontweight="bold")

    axes[0].plot(hist_df["loss"],         label="Train Loss",    color="tomato")
    axes[0].plot(hist_df["val_loss"],     label="Val Loss",      color="steelblue", ls="--")
    axes[0].set_title("Loss per Epoch");  axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Loss")
    axes[0].legend(); axes[0].grid(alpha=0.3)

    axes[1].plot(hist_df["accuracy"],     label="Train Accuracy",color="seagreen")
    axes[1].plot(hist_df["val_accuracy"], label="Val Accuracy",  color="orange",    ls="--")
    axes[1].set_title("Accuracy per Epoch"); axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Accuracy")
    axes[1].legend(); axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close()
    print(f"[VIZ] ✅ {path}")

plot_history(hist_df, f"{OUTPUT_DIR}/03_training_history.png", DPI)
```

---

### RULE-EVAL-08: Prediksi vs Aktual & Residual Plot

```
Output : output/04_prediction_vs_actual.png
         output/05_residual_plot.png
Format : Line chart (prediksi vs aktual) + histogram distribusi error
```

```python
def plot_pred_vs_actual(y_act, y_pred, n, path, dpi):
    fig, ax = plt.subplots(figsize=(13, 5))
    ax.plot(y_act[:n],  label="Aktual",   color="steelblue", lw=1.5)
    ax.plot(y_pred[:n], label="Prediksi", color="tomato",    lw=1.5, ls="--")
    ax.axhline(60, color="green", ls=":", lw=1.5, label="Threshold 60%")
    ax.set_xlabel("Timestep"); ax.set_ylabel("Kelembaban Tanah (%)")
    ax.set_title(f"Prediksi vs Aktual ({n} sampel) — LSTM Bonsai", fontsize=14, fontweight="bold")
    ax.legend(); ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close()
    print(f"[VIZ] ✅ {path}")

def plot_residual(y_act, y_pred, path, dpi):
    residuals = y_act - y_pred
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("Residual Plot — Error Prediksi LSTM Bonsai", fontsize=14, fontweight="bold")

    axes[0].plot(residuals, color="purple", lw=0.8, alpha=0.7)
    axes[0].axhline(0, color="black", ls="--", lw=1)
    axes[0].fill_between(range(len(residuals)), residuals, alpha=0.15, color="purple")
    axes[0].set_xlabel("Timestep"); axes[0].set_ylabel("Aktual - Prediksi")
    axes[0].set_title("Residual per Timestep"); axes[0].grid(alpha=0.3)

    axes[1].hist(residuals, bins=40, color="purple", alpha=0.7, edgecolor="white")
    axes[1].axvline(0, color="black", ls="--", lw=1.5)
    axes[1].set_xlabel("Error"); axes[1].set_ylabel("Frekuensi")
    axes[1].set_title("Distribusi Error (Histogram)"); axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close()
    print(f"[VIZ] ✅ {path}")

plot_pred_vs_actual(y_true_soil, y_pred_soil, N_SHOW, f"{OUTPUT_DIR}/04_prediction_vs_actual.png", DPI)
plot_residual(y_true_soil, y_pred_soil, f"{OUTPUT_DIR}/05_residual_plot.png", DPI)
```

---

### RULE-EVAL-09: Simpan Classification Report & Metrics Summary

```python
# Classification Report → CSV
report = classification_report(
    y_true, y_pred_class,
    target_names=["Tidak Siram (0)", "Siram (1)"],
    output_dict=True, digits=4
)
pd.DataFrame(report).T.to_csv(f"{OUTPUT_DIR}/06_classification_report.csv")
print(classification_report(y_true, y_pred_class,
      target_names=["Tidak Siram (0)", "Siram (1)"], digits=4))
print(f"[SAVE] ✅ output/06_classification_report.csv")

# Metrics Summary → CSV
summary_df = pd.DataFrame({
    "Metric" : ["Accuracy", "F1-Score (weighted)", "AUC-ROC", "RMSE (%)", "MAE (%)"],
    "Value"  : [round(acc,4), round(f1,4), round(roc_auc,4), round(rmse,4), round(mae,4)],
    "Target" : ["≥ 0.85", "≥ 0.80", "≥ 0.85", "≤ 5.0", "≤ 3.0"],
    "Status" : [
        "PASS" if acc     >= TARGET["accuracy"]  else "FAIL",
        "PASS" if f1      >= TARGET["f1"]         else "FAIL",
        "PASS" if roc_auc >= TARGET["auc_roc"]    else "FAIL",
        "PASS" if rmse    <= TARGET["rmse_pct"]   else "FAIL",
        "PASS" if mae     <= TARGET["mae_pct"]    else "FAIL",
    ]
})
summary_df.to_csv(f"{OUTPUT_DIR}/07_metrics_summary.csv", index=False)
print(summary_df.to_string(index=False))
print(f"[SAVE] ✅ output/07_metrics_summary.csv")
```

---

## 10. Rule Artefak & File Antara

### RULE-ART-01: Peta Ketergantungan Antar Notebook

```
Notebook          Menghasilkan                   Dibutuhkan oleh
──────────────────────────────────────────────────────────────────────
01_preprocessing  data_train.npz              →  02_training
                  data_test.npz               →  03_testing, 04_evaluasi
                  scaler_bonsai.pkl           →  03_testing
                  label_info.json             →  02_training, 03_testing

02_training       model_bonsai_lstm.keras     →  03_testing
                  training_history.csv        →  04_evaluasi

03_testing        predictions.csv             →  04_evaluasi
                  y_test_labels.csv           →  04_evaluasi
```

### RULE-ART-02: Urutan Eksekusi — WAJIB BERURUTAN

```
01_preprocessing.ipynb  →  selesai & artefak tersimpan
       ↓
02_training.ipynb       →  selesai & model tersimpan
       ↓
03_testing.ipynb        →  selesai & prediksi tersimpan
       ↓
04_evaluasi.ipynb       →  output akhir dihasilkan

DILARANG menjalankan notebook secara paralel atau melompat urutan.
```

### RULE-ART-03: Format File Artefak

| File | Format | Keterangan |
|---|---|---|
| `data_train.npz` | NumPy compressed | X_train, y_train_cls, y_train_reg |
| `data_test.npz`  | NumPy compressed | X_test, y_test_cls, y_test_reg |
| `scaler_bonsai.pkl` | Joblib pickle | MinMaxScaler object |
| `label_info.json`   | JSON | Metadata & threshold |
| `model_bonsai_lstm.keras` | Keras SavedModel | Bobot & arsitektur |
| `training_history.csv`    | CSV | Loss, acc, AUC per epoch |
| `predictions.csv`         | CSV | Probabilitas & kelas prediksi |
| `y_test_labels.csv`       | CSV | Label & nilai aktual |

---

## 11. Rule Logika Pengendalian Atap Otomatis

> Pengendalian atap **tidak menggunakan model ML**, melainkan rule-based
> threshold. Implementasikan sebagai fungsi di `01_preprocessing.ipynb`
> dan demonstrasikan hasilnya di `04_evaluasi.ipynb`.

### RULE-ROOF-01: Fungsi Kontrol Atap

```python
def roof_control(temp_c: float, humidity_air: float, soil_moisture: float) -> str:
    """
    Menentukan status atap berdasarkan nilai sensor saat ini.

    TUTUP atap jika salah satu kondisi terpenuhi:
      - humidity_air   > 85%   → potensi hujan / kelembapan ekstrem
      - soil_moisture  > 85%   → tanah sudah sangat lembab
      - temp_c         < 15°C  → suhu terlalu rendah (risiko frost)

    Sensor error (nilai di luar batas valid):
      → default TUTUP (keamanan lebih utama dari optimasi)

    Returns: "TUTUP" | "BUKA"
    """
    # Validasi sensor — jika error, default TUTUP
    sensor_ok = (
        10 <= temp_c <= 45 and
        0  <  humidity_air  <= 100 and
        0  <  soil_moisture <= 100
    )
    if not sensor_ok:
        return "TUTUP"   # safety default

    if humidity_air > 85 or soil_moisture > 85 or temp_c < 15:
        return "TUTUP"
    return "BUKA"
```

### RULE-ROOF-02: Prioritas Keamanan (Safety First)
```
Keamanan SELALU lebih utama dari optimasi.
Jika sensor error → status TUTUP tanpa exception, tanpa delay.
```

---

## 12. Rule Reproduksibilitas & Versi

### RULE-REPRO-01: Seed Global (Wajib di Cell 3 Setiap Notebook)

```python
import random, os
import numpy as np
import tensorflow as tf

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)
os.environ["PYTHONHASHSEED"] = str(SEED)
print(f"[SEED] Global seed = {SEED}")
```

### RULE-REPRO-02: Verifikasi Versi Library

```python
import tensorflow as tf, sklearn, pandas as pd
import numpy as np, matplotlib, seaborn

libs = {
    "TensorFlow"  : tf.__version__,
    "Scikit-learn": sklearn.__version__,
    "Pandas"      : pd.__version__,
    "NumPy"       : np.__version__,
    "Matplotlib"  : matplotlib.__version__,
    "Seaborn"     : seaborn.__version__,
}
for lib, ver in libs.items():
    print(f"  {lib:<14}: {ver}")
```

### RULE-REPRO-03: Versi yang Didukung

| Library | Versi Terpaku | Python minimum |
|---|---|---|
| TensorFlow | 2.15.0 | 3.9 |
| Scikit-learn | 1.4.0 | |
| Pandas | 2.2.0 | |
| NumPy | 1.26.3 | |
| Matplotlib | 3.8.2 | |
| Seaborn | 0.13.2 | |
| Joblib | 1.3.2 | |

---

## 13. Referensi Metrik & Formula

| Metrik | Formula | Target Minimum |
|---|---|---|
| Accuracy | $(TP+TN)/(TP+TN+FP+FN)$ | ≥ 85% |
| Precision | $TP/(TP+FP)$ | ≥ 0.80 |
| Recall | $TP/(TP+FN)$ | ≥ 0.80 |
| F1-Score | $2 \times (P \times R)/(P+R)$ | ≥ 0.80 |
| AUC-ROC | Area Under ROC Curve | ≥ 0.85 |
| RMSE | $\sqrt{\frac{1}{n}\sum(y_i-\hat{y}_i)^2}$ | ≤ 5% |
| MAE | $\frac{1}{n}\sum\|y_i-\hat{y}_i\|$ | ≤ 3% |

**Interpretasi Confusion Matrix:**

| | Prediksi: Tidak Siram | Prediksi: Siram |
|---|---|---|
| **Aktual: Tidak Siram** | TN ✅ Benar tidak siram | FP ⚠️ Salah siram |
| **Aktual: Siram** | FN 🚨 **Kritis** — gagal deteksi | TP ✅ Benar siram |

> **FN (False Negative) adalah kesalahan paling kritis** dalam sistem ini
> karena berarti model gagal mendeteksi bahwa tanaman membutuhkan penyiraman,
> yang berpotensi menyebabkan tanaman bonsai kekeringan dan mati.

---

*Dokumen ini bersifat normatif. Setiap aturan yang tercantum wajib diikuti
kecuali secara eksplisit ditandai "(opsional)". Dibuat untuk proyek:
Implementasi Metode LSTM pada Sistem Monitoring Tanaman Bonsai.*
