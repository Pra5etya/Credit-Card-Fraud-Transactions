import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import boxcox, yeojohnson, zscore
from sklearn.preprocessing import PowerTransformer
from statsmodels.stats.stattools import medcouple
from scipy.stats.mstats import winsorize

# Fungsi untuk menampilkan histogram dengan KDE
def plot_histogram(df, title="Feature Distribution"):
    fig, axes = plt.subplots(nrows=1, ncols=len(df.columns), figsize=(20, 5))
    
    for i, col in enumerate(df.columns):
        ax = axes[i]
        sns.histplot(df[col], kde=True, ax=ax, bins=30)
        ax.axvline(df[col].mean(), color='r', linestyle='dashed', linewidth=1)
        ax.set_title(f'{col}\nMean: {df[col].mean():.2f}')
    
    plt.suptitle(title)
    plt.tight_layout()
    plt.show()

# Fungsi untuk menerapkan transformasi data
def transform_features(df):
    df["log_distance_km"] = np.log1p(df["distance_km"].clip(lower=1e-5))
    df["log_speed_km/h"] = np.log1p(df["speed_km/h"].clip(lower=1e-5))
    df["sqrt_time_diff_hour"] = np.sqrt(df["time_diff_hour"].clip(lower=0))
    df["bc_transaction_dollar"], _ = boxcox(df["transaction_dollar_amount"].clip(lower=1e-5))
    df["yj_distance_km"], _ = yeojohnson(df["log_distance_km"])  # Yeo-Johnson untuk log
    df["yj_speed_km/h"], _ = yeojohnson(df["log_speed_km/h"])  # Yeo-Johnson untuk log
    return df

# Fungsi untuk menghapus outlier berdasarkan Z-score dan IQR
def remove_outliers(df, features, method='zscore', threshold=3):
    if method == 'zscore':
        df = df[(np.abs(zscore(df[features])) < threshold).all(axis=1)]
    elif method == 'iqr':
        Q1 = df[features].quantile(0.25)
        Q3 = df[features].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        df = df[~((df[features] < lower_bound) | (df[features] > upper_bound)).any(axis=1)]
    return df

# Fungsi untuk melakukan Winsorizing

def apply_winsorizing(df, feature, limits=(0.05, 0.05)):
    df[feature] = winsorize(df[feature], limits=limits)
    return df

# Contoh Penggunaan
if __name__ == "__main__":
    # Membaca dataset (Contoh dummy)
    df = pd.read_csv("data.csv")
    
    # Menampilkan histogram sebelum transformasi
    plot_histogram(df[["distance_km", "speed_km/h", "time_diff_hour", "transaction_dollar_amount"]], 
                   title="Before Transformation")
    
    # Transformasi fitur
    df = transform_features(df)
    
    # Menampilkan histogram setelah transformasi
    plot_histogram(df[["log_distance_km", "log_speed_km/h", "sqrt_time_diff_hour", "bc_transaction_dollar"]], 
                   title="After Transformation")
    
    # Menghapus outlier dengan metode Z-score
    df = remove_outliers(df, ["log_distance_km", "log_speed_km/h", "sqrt_time_diff_hour", "bc_transaction_dollar"], method='zscore')
    
    # Menstabilkan distribusi dengan Winsorizing
    df = apply_winsorizing(df, "yj_distance_km", limits=(0.05, 0.05))
    
    # Menampilkan informasi dataset setelah preprocessing
    print(df.info())
