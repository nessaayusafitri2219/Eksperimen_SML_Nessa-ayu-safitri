import pandas as pd
import mlflow
import dagshub
import os
import shutil # Tambahan untuk menghapus folder lama
import sys
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# --- KONFIGURASI ---
REPO_OWNER = "nessaayusafitri2219"
REPO_NAME = "Eksperimen_SML_Nessa-ayu-safitri"

# Tetap init DagsHub hanya untuk logging Metrics (Angka), bukan Model
try:
    dagshub.init(repo_owner=REPO_OWNER, repo_name=REPO_NAME, mlflow=True)
except Exception as e:
    print(f"Warning DagsHub: {e}")

mlflow.set_experiment("Walmart_Sales_CI_Production")

# --- LOAD DATA ---
filename = 'preprocessing/Walmart_Sales_preprocessing.csv'
# Logika pencarian file (sama seperti sebelumnya)
if os.path.exists(filename): data_path = filename
elif os.path.exists(f"../{filename}"): data_path = f"../{filename}"
elif os.path.exists(f"Workflow-CI/{filename}"): data_path = f"Workflow-CI/{filename}"
else: data_path = "preprocessing/Walmart_Sales_preprocessing.csv"

print(f"Loading data: {data_path}")
df = pd.read_csv(data_path)

X = df.drop('Weekly_Sales', axis=1)
y = df['Weekly_Sales']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- TRAINING ---
print("Memulai Training...")

with mlflow.start_run(run_name="Docker_CI_Build_Local") as run:
    params = {"n_estimators": 100, "max_depth": 10, "random_state": 42}
    model = RandomForestRegressor(**params)
    model.fit(X_train, y_train)

    # Evaluasi
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f"MSE: {mse:.2f}, R2: {r2:.4f}")

    # 1. Log Metrics ke DagsHub (Kecil, jadi aman)
    mlflow.log_params(params)
    mlflow.log_metric("mse", mse)
    mlflow.log_metric("r2_score", r2)

    # 2. SIMPAN MODEL SECARA LOKAL (Solusi Masalah Network)
    # Kita simpan di folder bernama 'model_output' di dalam runner
    local_model_path = "model_output"
    
    # Hapus folder jika sudah ada (supaya bersih)
    if os.path.exists(local_model_path):
        shutil.rmtree(local_model_path)
        
    print(f"Menyimpan model ke folder lokal: {local_model_path}...")
    mlflow.sklearn.save_model(model, local_model_path)
    print("Model berhasil disimpan secara lokal!")

    # Tidak perlu simpan run_id.txt lagi karena kita pakai path lokal

    # --- 6. SIMPAN RUN ID (WAJIB ADA) ---
    print(f"Menyimpan Run ID: {run.info.run_id}")
    with open("run_id.txt", "w") as f:
        f.write(run.info.run_id)
