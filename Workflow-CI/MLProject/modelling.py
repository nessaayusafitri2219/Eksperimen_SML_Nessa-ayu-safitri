import pandas as pd
import mlflow
import dagshub
import os
import shutil
import sys
import matplotlib.pyplot as plt # Wajib untuk plot
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# --- KONFIGURASI ---
REPO_OWNER = "nessaayusafitri2219"
REPO_NAME = "Eksperimen_SML_Nessa-ayu-safitri"

# Init DagsHub
try:
    dagshub.init(repo_owner=REPO_OWNER, repo_name=REPO_NAME, mlflow=True)
except Exception as e:
    print(f"Warning DagsHub: {e}")

mlflow.set_experiment("Walmart_Sales_CI_Production")

# --- LOAD DATA ---
filename = 'Walmart_Sales_preprocessing.csv'
# Logika pencarian file
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

    # 1. Log Metrics & Params ke DagsHub
    mlflow.log_params(params)
    mlflow.log_metric("mse", mse)
    mlflow.log_metric("r2_score", r2)

    # --- 2. LOG ARTEFAK VISUALISASI KE DAGSHUB (Minimal 2) ---
    print("Mengupload artefak visualisasi ke DagsHub...")
    
    # Artefak A: Actual vs Predicted Plot
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_pred, alpha=0.3)
    plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', lw=2)
    plt.xlabel('Actual')
    plt.ylabel('Predicted')
    plt.title('Actual vs Predicted Sales')
    plt.savefig("actual_vs_predicted.png")
    plt.close()
    mlflow.log_artifact("actual_vs_predicted.png") # Upload ke DagsHub

    # Artefak B: Feature Importance Plot
    plt.figure(figsize=(10, 6))
    feat_importances = pd.Series(model.feature_importances_, index=X.columns)
    feat_importances.nlargest(10).plot(kind='barh')
    plt.title('Feature Importance')
    plt.savefig("feature_importance.png")
    plt.close()
    mlflow.log_artifact("feature_importance.png") # Upload ke DagsHub
    
    print("Artefak visualisasi berhasil diupload!")

    # --- 3. SIMPAN MODEL SECARA LOKAL (Untuk Docker Build) ---
    local_model_path = "model_output"
    if os.path.exists(local_model_path):
        shutil.rmtree(local_model_path)
        
    print(f"Menyimpan model ke folder lokal: {local_model_path}...")
    mlflow.sklearn.save_model(model, local_model_path)

    # --- 4. SIMPAN RUN ID ---
    print(f"Menyimpan Run ID: {run.info.run_id}")
    with open("run_id.txt", "w") as f:
        f.write(run.info.run_id)

    # Bersihkan file gambar sementara
    if os.path.exists("actual_vs_predicted.png"): os.remove("actual_vs_predicted.png")
    if os.path.exists("feature_importance.png"): os.remove("feature_importance.png")