import pandas as pd
import mlflow
import dagshub
import os
import sys
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# --- 1. KONFIGURASI ENV VAR (SOLUSI SSLEOFError) ---
# Taruh ini PALING ATAS, sebelum mlflow memulai aktivitas apapun.
# Ini memaksa MLflow untuk mencoba ulang lebih sering dan menunggu lebih lama.
os.environ["MLFLOW_HTTP_REQUEST_MAX_RETRIES"] = "5"  # Coba ulang hingga 5 kali
os.environ["MLFLOW_HTTP_REQUEST_TIMEOUT"] = "120"     # Tunggu respon hingga 120 detik
os.environ["MLFLOW_GCS_DEFAULT_TIMEOUT"] = "120"     # Timeout untuk operasi artifact

REPO_OWNER = "Nessa-ayu-safitri"
REPO_NAME = "Eksperimen_SML_Nessa-ayu-safitri"

# --- 2. INIT DAGSHUB ---
try:
    dagshub.init(repo_owner=REPO_OWNER, repo_name=REPO_NAME, mlflow=True)
    print("DagsHub Initialized successfully.")
except Exception as e:
    print(f"Peringatan inisialisasi DagsHub: {e}")

mlflow.set_experiment("Walmart_Sales_CI_Production")

# --- 3. LOAD DATA ---
filename = 'peprocessing/Walmart_Sales_preprocessing.csv'
if os.path.exists(filename):
    data_path = filename
elif os.path.exists(f"../{filename}"):
    data_path = f"../{filename}"
elif os.path.exists(f"Workflow-CI/{filename}"): # Tambahan cek path
    data_path = f"Workflow-CI/{filename}"
else:
    # Fallback terakhir: Coba cari di folder preprocessing root
    data_path = "preprocessing/Walmart_Sales_preprocessing.csv"
    if not os.path.exists(data_path):
         print(f"Error: {filename} tidak ditemukan dimanapun.")
         sys.exit(1)

print(f"Memuat data dari: {data_path}")
df = pd.read_csv(data_path)

# --- 4. SPLIT DATA ---
X = df.drop('Weekly_Sales', axis=1)
y = df['Weekly_Sales']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- 5. TRAINING & LOGGING ---
print("Memulai Training...")

# Kode solusi Anda: Pastikan tidak ada inisialisasi berat di dalam blok 'with'
# Biarkan blok 'with' hanya fokus pada logging.

with mlflow.start_run(run_name="Docker_CI_Build") as run:
    
    # Train Model
    params = {"n_estimators": 100, "max_depth": 10, "random_state": 42}
    model = RandomForestRegressor(**params)
    model.fit(X_train, y_train)

    # Evaluasi
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f"Hasil Evaluasi -> MSE: {mse:.2f}, R2: {r2:.4f}")

    # Log Metrics & Params
    mlflow.log_params(params)
    mlflow.log_metric("mse", mse)
    mlflow.log_metric("r2_score", r2)

    # Log Model (Bagian yang sering error)
    print("Mencoba upload model ke DagsHub...")
    mlflow.sklearn.log_model(model, "model")
    print("Model berhasil diupload (Check artifact tab if warnings appeared).")
    
    # Simpan Run ID
    with open("run_id.txt", "w") as f:
        f.write(run.info.run_id)

print(f"Proses Selesai. Run ID: {run.info.run_id}")
