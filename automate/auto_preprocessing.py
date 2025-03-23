# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns
# from scipy.stats import boxcox, yeojohnson, zscore
# from sklearn.preprocessing import PowerTransformer
# from statsmodels.stats.stattools import medcouple
# from scipy.stats.mstats import winsorize

# # Fungsi untuk menampilkan histogram dengan KDE
# def plot_histogram(df, title="Feature Distribution"):
#     fig, axes = plt.subplots(nrows=1, ncols=len(df.columns), figsize=(20, 5))
    
#     for i, col in enumerate(df.columns):
#         ax = axes[i]
#         sns.histplot(df[col], kde=True, ax=ax, bins=30)
#         ax.axvline(df[col].mean(), color='r', linestyle='dashed', linewidth=1)
#         ax.set_title(f'{col}\nMean: {df[col].mean():.2f}')
    
#     plt.suptitle(title)
#     plt.tight_layout()
#     plt.show()

# # Fungsi untuk menerapkan transformasi data
# def transform_features(df):
#     df["log_distance_km"] = np.log1p(df["distance_km"].clip(lower=1e-5))
#     df["log_speed_km/h"] = np.log1p(df["speed_km/h"].clip(lower=1e-5))
#     df["sqrt_time_diff_hour"] = np.sqrt(df["time_diff_hour"].clip(lower=0))
#     df["bc_transaction_dollar"], _ = boxcox(df["transaction_dollar_amount"].clip(lower=1e-5))
#     df["yj_distance_km"], _ = yeojohnson(df["log_distance_km"])  # Yeo-Johnson untuk log
#     df["yj_speed_km/h"], _ = yeojohnson(df["log_speed_km/h"])  # Yeo-Johnson untuk log
#     return df

# # Fungsi untuk menghapus outlier berdasarkan Z-score dan IQR
# def remove_outliers(df, features, method='zscore', threshold=3):
#     if method == 'zscore':
#         df = df[(np.abs(zscore(df[features])) < threshold).all(axis=1)]
#     elif method == 'iqr':
#         Q1 = df[features].quantile(0.25)
#         Q3 = df[features].quantile(0.75)
#         IQR = Q3 - Q1
#         lower_bound = Q1 - 1.5 * IQR
#         upper_bound = Q3 + 1.5 * IQR
#         df = df[~((df[features] < lower_bound) | (df[features] > upper_bound)).any(axis=1)]
#     return df

# # Fungsi untuk melakukan Winsorizing

# def apply_winsorizing(df, feature, limits=(0.05, 0.05)):
#     df[feature] = winsorize(df[feature], limits=limits)
#     return df

# # Contoh Penggunaan
# if __name__ == "__main__":
#     # Membaca dataset (Contoh dummy)
#     df = pd.read_csv("data.csv")
    
#     # Menampilkan histogram sebelum transformasi
#     plot_histogram(df[["distance_km", "speed_km/h", "time_diff_hour", "transaction_dollar_amount"]], 
#                    title="Before Transformation")
    
#     # Transformasi fitur
#     df = transform_features(df)
    
#     # Menampilkan histogram setelah transformasi
#     plot_histogram(df[["log_distance_km", "log_speed_km/h", "sqrt_time_diff_hour", "bc_transaction_dollar"]], 
#                    title="After Transformation")
    
#     # Menghapus outlier dengan metode Z-score
#     df = remove_outliers(df, ["log_distance_km", "log_speed_km/h", "sqrt_time_diff_hour", "bc_transaction_dollar"], method='zscore')
    
#     # Menstabilkan distribusi dengan Winsorizing
#     df = apply_winsorizing(df, "yj_distance_km", limits=(0.05, 0.05))
    
#     # Menampilkan informasi dataset setelah preprocessing
#     print(df.info())



import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import boxcox, zscore
from sklearn.preprocessing import power_transform
from scipy.stats.mstats import winsorize

# Function to preprocess data
def preprocess_data(df):
    # Exclude non-numeric columns
    exc_data = ['credit_card', 'long', 'lat', 'zipcode', 'year', 'prev_long', 'prev_lat', 'transaction_count']
    
    hist_data = df.select_dtypes(include=['number']).drop(columns=exc_data, errors='ignore')
    print(hist_data.columns)
    
    # Filter out columns with very low variance
    valid_cols = hist_data.var()[hist_data.var() > 1e-12].index
    hist_data = hist_data[valid_cols]
    
    print("Valid columns after variance filtering:", list(valid_cols))  # Debugging step
    
    # Add small noise to avoid precision loss
    hist_data += np.random.normal(0, 1e-3, hist_data.shape)  # Adjusted noise level
    
    # Compute skewness and kurtosis only on valid columns
    skewness_values = hist_data.skew()
    kurtosis_values = hist_data.kurtosis()
    
    print("Skewness before transformation:\n", skewness_values, '\n')
    print("Kurtosis before transformation:\n", kurtosis_values, '\n')
    
    # Determine columns that need transformation
    columns_to_transform = skewness_values[(abs(skewness_values) > 0.5) | (kurtosis_values > 3)].index.tolist()
    
    if columns_to_transform:
        print("Columns to transform:", columns_to_transform)
        # Apply transformations only on selected columns
        for col in columns_to_transform:
            if col in df.columns:
                df[f'log_{col}'] = np.log1p(df[col] + abs(df[col].min()) + 1)  # Adjust for negative values
                df[f'yj_{col}'] = power_transform(df[[col]], method='yeo-johnson')
                
        # Winsorization for normalization
        for col in columns_to_transform:
            if f'yj_{col}' in df.columns:
                df[f'wins_yj_{col}'] = np.clip(df[f'yj_{col}'], 
                                               np.percentile(df[f'yj_{col}'], 2), 
                                               np.percentile(df[f'yj_{col}'], 98))  # Avoiding winsorize issues
    else:
        print("No columns need transformation.")
    
    # Outlier detection using Z-score
    columns_to_check = hist_data.columns
    df[[f'zscore_{col}' for col in columns_to_check]] = df[columns_to_check].apply(zscore)
    
    # Filter outliers
    outlier_condition = (df[[f'zscore_{col}' for col in columns_to_check]].abs() > 3).any(axis=1)
    df_filtered = df[~outlier_condition]
    
    # IQR method to remove outliers
    def filter_outliers(df, columns):
        while True:
            outlier_indices = set()
            for col in columns:
                if df[col].isna().all():  # Skip columns that are entirely NaN
                    continue
                Q1 = np.percentile(df[col].dropna(), 25, method='midpoint')
                Q3 = np.percentile(df[col].dropna(), 75, method='midpoint')
                IQR = Q3 - Q1
                upper_bound = Q3 + 1.5 * IQR
                lower_bound = Q1 - 1.5 * IQR
                col_outliers = df[(df[col] >= upper_bound) | (df[col] <= lower_bound)].index
                outlier_indices.update(col_outliers)
            if not outlier_indices:
                break
            df.drop(index=outlier_indices, inplace=True)
        return df
    
    df_cleaned = filter_outliers(df_filtered.copy(), columns_to_check)
    
    # Drop unnecessary columns
    exc_col = ['prev_long', 'prev_lat'] + [f'log_{col}' for col in columns_to_transform] + \
              [f'yj_{col}' for col in columns_to_transform] + [f'wins_yj_{col}' for col in columns_to_transform]
    raw_df = df_cleaned.drop(columns=exc_col, errors='ignore')
    
    # Visualize histogram of cleaned data
    plt.figure(figsize=(15, 10))
    raw_df.hist(bins=30, figsize=(15, 10), layout=(len(raw_df.columns) // 4 + 1, 4))
    plt.tight_layout()
    plt.show()
    
    return raw_df