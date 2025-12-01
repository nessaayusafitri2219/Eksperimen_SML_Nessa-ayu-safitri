import pandas as pd
import mlflow
import dagshub
import os
import sys
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# --- 1. KONFIGURASI ---
REPO_OWNER = "nessaayusafitri2219"
REPO_NAME = "Eksperimen_SML_Nessa-ayu-safitri"

# Inisialisasi DagsHub
# Di GitHub Actions, otentikasi akan ditangani oleh Environment Variables (MLFLOW_TRACKING_USERNAME & PASSWORD)
try:
    dagshub.init(repo_owner=REPO_OWNER, repo_name=REPO_NAME, mlflow=True)
except Exception as e:
    print(f"Peringatan inisialisasi DagsHub: {e}")

mlflow.set_experiment("Walmart_Sales_CI_Production")

# --- 2. LOAD DATA ---
# Script akan mencari file csv di direktori yang sama atau satu level di atasnya
filename = 'Walmart_Sales_preprocessing.csv'
if os.path.exists(filename):
    data_path = filename
elif os.path.exists(f"../{filename}"):
    data_path = f"../{filename}"
else:
    print(f"Error: {filename} tidak ditemukan.")
    sys.exit(1)

print(f"Memuat data dari: {data_path}")
df = pd.read_csv(data_path)

# --- 3. SPLIT DATA ---
X = df.drop('Weekly_Sales', axis=1)
y = df['Weekly_Sales']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- 4. TRAINING & LOGGING (SIMPLIFIED) ---
print("Memulai Training...")

with mlflow.start_run(run_name="Docker_CI_Build") as run:
    
    # Parameter Model (Gunakan parameter terbaik hasil eksperimen sebelumnya)
    params = {
        "n_estimators": 100,
        "max_depth": 10,
        "random_state": 42
    }
    
    # Log Parameter
    mlflow.log_params(params)
    mlflow.log_param("model_type", "RandomForestRegressor")

    # Train Model
    model = RandomForestRegressor(**params)
    model.fit(X_train, y_train)

    # Evaluasi
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print(f"Hasil Evaluasi -> MSE: {mse:.2f}, R2: {r2:.4f}")

    # Log Metrik
    mlflow.log_metric("mse", mse)
    mlflow.log_metric("r2_score", r2)

    # Log Model (PENTING: Ini yang akan dibuild menjadi Docker Image)
    mlflow.sklearn.log_model(model, "model")
    
    # Simpan Run ID ke file text agar bisa dibaca oleh GitHub Actions (opsional tapi berguna)
    with open("run_id.txt", "w") as f:
        f.write(run.info.run_id)

print(f"Model berhasil dilatih dan disimpan di MLflow. Run ID: {run.info.run_id}")
