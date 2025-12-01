import pandas as pd
import mlflow
import dagshub
import os
import sys
import time
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# --- 1. SETTING ANTI-GAGAL KONEKSI (PENTING) ---
# Taruh ini di paling atas!
os.environ["MLFLOW_HTTP_REQUEST_MAX_RETRIES"] = "10"  # Coba ulang sampai 10x
os.environ["MLFLOW_HTTP_REQUEST_BACKOFF_FACTOR"] = "2" # Tunggu makin lama tiap gagal
os.environ["MLFLOW_HTTP_REQUEST_TIMEOUT"] = "300"     # Tunggu respon 5 menit
os.environ["MLFLOW_GCS_DEFAULT_TIMEOUT"] = "300"

# --- 2. KONFIGURASI ---
REPO_OWNER = "nessaayusafitri2219"
REPO_NAME = "Eksperimen_SML_Nessa-ayu-safitri"

try:
    dagshub.init(repo_owner=REPO_OWNER, repo_name=REPO_NAME, mlflow=True)
except Exception as e:
    print(f"Warning Init DagsHub: {e}")

mlflow.set_experiment("Walmart_Sales_CI_Production")

# --- 3. LOAD DATA ---
filename = 'preprocessing/Walmart_Sales_preprocessing.csv'
if os.path.exists(filename): data_path = filename
elif os.path.exists(f"../{filename}"): data_path = f"../{filename}"
elif os.path.exists(f"Workflow-CI/{filename}"): data_path = f"Workflow-CI/{filename}"
else: data_path = "preprocessing/Walmart_Sales_preprocessing.csv"

print(f"Loading data: {data_path}")
df = pd.read_csv(data_path)

X = df.drop('Weekly_Sales', axis=1)
y = df['Weekly_Sales']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- 4. TRAINING ---
print("Memulai Training...")

with mlflow.start_run(run_name="Docker_CI_Build") as run:
    params = {"n_estimators": 100, "max_depth": 10, "random_state": 42}
    model = RandomForestRegressor(**params)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f"MSE: {mse:.2f}, R2: {r2:.4f}")

    mlflow.log_params(params)
    mlflow.log_metric("mse", mse)
    mlflow.log_metric("r2_score", r2)

    # --- 5. LOG MODEL (UPLOAD) DENGAN RETRY MANUAL ---
    print("Sedang mengupload model ke DagsHub (Mohon tunggu)...")
    
    # Kita coba paksa upload
    try:
        mlflow.sklearn.log_model(model, "model")
        print("✅ Model berhasil diupload!")
    except Exception as e:
        print(f"❌ Gagal upload pertama: {e}")
        print("Mencoba lagi dalam 10 detik...")
        time.sleep(10)
        mlflow.sklearn.log_model(model, "model") # Coba kedua kali
        print("✅ Model berhasil diupload pada percobaan kedua!")

    # --- 6. SIMPAN RUN ID (WAJIB ADA) ---
    print(f"Menyimpan Run ID: {run.info.run_id}")
    with open("run_id.txt", "w") as f:
        f.write(run.info.run_id)
