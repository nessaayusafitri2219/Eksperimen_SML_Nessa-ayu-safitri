import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import mlflow
import dagshub
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# --- KONFIGURASI DAGSHUB ---
REPO_OWNER = "nessaayusafitri2219"
REPO_NAME = "Eksperimen_SML_Nessa-ayu-safitri"
dagshub.init(repo_owner=REPO_OWNER, repo_name=REPO_NAME, mlflow=True)
mlflow.set_experiment("Walmart_Sales_Experiment")

# Load Data (Pastikan path benar)
# Jika dijalankan lokal, sesuaikan path ini
data_path = 'preprocessing/Walmart_Sales_preprocessing.csv' 
if not os.path.exists(data_path):
    # Fallback jika dijalankan langsung di dalam folder Membangun_model
    data_path = '../preprocessing/Walmart_Sales_preprocessing.csv'

df = pd.read_csv(data_path)

# Split Data
X = df.drop('Weekly_Sales', axis=1)
y = df['Weekly_Sales']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- MANUAL LOGGING ---
with mlflow.start_run(run_name="RandomForest_Manual_Log"):
    
    # 1. Log Parameter Manual
    n_estimators = 100
    max_depth = 10
    mlflow.log_param("n_estimators", n_estimators)
    mlflow.log_param("max_depth", max_depth)
    mlflow.log_param("model_type", "RandomForestRegressor")

    # Train Model
    model = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
    model.fit(X_train, y_train)
    
    # Predict & Evaluate
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    # 2. Log Metrik Manual
    print(f"MSE: {mse}")
    mlflow.log_metric("mse", mse)
    mlflow.log_metric("r2_score", r2)

    # 3. Log Artefak Tambahan (Minimal 2)
    
    # Artefak A: Scatter Plot (Actual vs Predicted)
    plt.figure(figsize=(8, 6))
    plt.scatter(y_test, y_pred, alpha=0.3)
    plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', lw=2)
    plt.xlabel('Actual')
    plt.ylabel('Predicted')
    plt.title('Actual vs Predicted Sales')
    plt.savefig("pred_vs_actual.png")
    mlflow.log_artifact("pred_vs_actual.png")
    plt.close()

    # Artefak B: Feature Importance
    plt.figure(figsize=(10, 6))
    feat_importances = pd.Series(model.feature_importances_, index=X.columns)
    feat_importances.nlargest(10).plot(kind='barh')
    plt.title('Top 10 Feature Importances')
    plt.savefig("feature_importance.png")
    mlflow.log_artifact("feature_importance.png")
    plt.close()

    # Log Model Sklearn
    mlflow.sklearn.log_model(model, "model")
    
    # Bersihkan file lokal
    if os.path.exists("pred_vs_actual.png"): os.remove("pred_vs_actual.png")
    if os.path.exists("feature_importance.png"): os.remove("feature_importance.png")

print("Training selesai. Cek DagsHub UI.")
