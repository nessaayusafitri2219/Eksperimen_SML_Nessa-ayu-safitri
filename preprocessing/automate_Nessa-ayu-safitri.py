import pandas as pd
import os
import sys

# Fungsi Load Data
def load_data(path):
    if not os.path.exists(path):
        print(f"Error: File {path} tidak ditemukan.")
        sys.exit(1)
    print(f"Loading data from {path}...")
    return pd.read_csv(path)

# Fungsi Preprocessing (Sesuaikan dengan EDA Anda sebelumnya)
def preprocess_data(df):
    print("Memulai preprocessing...")
    # 1. Hapus Duplikat
    df = df.drop_duplicates()
    
    # 2. Parsing Tanggal (Feature Engineering)
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'], dayfirst=True)
        df['Year'] = df['Date'].dt.year
        df['Month'] = df['Date'].dt.month
        df['Is_Holiday'] = df['Holiday_Flag'].apply(lambda x: 1 if x == 1 else 0)
        df = df.drop(columns=['Date']) # Hapus kolom asli agar siap dilatih
    
    # 3. Handling Missing Values
    df = df.dropna()
    
    return df

# Fungsi Simpan Data
def save_data(df, output_path):
    output_dir = os.path.dirname(output_path)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    df.to_csv(output_path, index=False)
    print(f"Data preprocessing tersimpan di: {output_path}")

if __name__ == "__main__":
    # Path file (Relative terhadap root repository)
    input_file = 'Walmart_Sales_raw.csv'
    output_file = 'preprocessing/Walmart_Sales_preprocessing.csv'
    
    df = load_data(input_file)
    df_clean = preprocess_data(df)
    save_data(df_clean, output_file)
