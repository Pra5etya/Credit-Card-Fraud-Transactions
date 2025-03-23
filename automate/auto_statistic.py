# from scipy.stats import anderson, normaltest, levene, mannwhitneyu, permutation_test
# import numpy as np
# import pandas as pd

# def safe_anderson(data):
#     """Handle cases where Anderson-Darling test might fail due to low variance."""
#     if np.std(data) == 0:
#         return np.nan, np.nan
    
#     result = anderson(data, dist='norm')
#     return result.statistic, result.critical_values[2]

# def safe_normaltest(data):
#     """Handle normality test when data has near-zero variance."""
#     if np.std(data) == 0:
#         return np.nan, 1.0
    
#     return normaltest(data)

# def safe_levene(group1, group2):
#     """Handle variance homogeneity test when one or both groups have no variance."""
#     if np.std(group1) == 0 or np.std(group2) == 0:
#         return np.nan, 1.0
    
#     return levene(group1, group2)

# def analyze_numeric_features(cc_df):
#     # Menghapus kolom yang tidak relevan
#     exc_data = ['credit_card', 'long', 'lat', 'zipcode', 'year']
#     num_data = cc_df.select_dtypes(include = ['number']).drop(columns = exc_data, errors = 'ignore')
    
#     print("===== Uji Normalitas Data =====")
#     for col in num_data:
#         data = cc_df[col].dropna().values  # Hindari NaN

#         # Skip jika datanya terlalu kecil
#         if len(data) < 8:
#             print(f"Column {col}: Data terlalu sedikit untuk uji normalitas.\n")
#             continue

#         stat, crit_5_percent = safe_anderson(data)
#         if np.isnan(stat):
#             print(f"Column {col}: Variansi nol, tidak dapat diuji.\n")
#         else:
#             print(f"Column: {col}\nStatistic: {stat:.4f}, Critical Value (5%): {crit_5_percent:.4f}")
#             print(f"Column {col}: {'Data berdistribusi normal' if stat < crit_5_percent else 'Data tidak berdistribusi normal'}\n")
        
#         stat, p = safe_normaltest(data)
#         print(f"Column: {col}, Statistics = {stat:.4f}, p = {p:.4f}")
#         print(f"Column {col}: {'Data berdistribusi normal' if p > 0.05 else 'Data tidak berdistribusi normal'}\n")

#         print('-' * 50, '\n')

#     print('=' * 50, '\n')
#     print("===== Uji Homogenitas Varians =====")
#     for col in num_data:
#         group1 = cc_df[cc_df['season'] == 'fall'][col].dropna().values
#         group2 = cc_df[cc_df['season'] == 'summer'][col].dropna().values
        
#         size1, size2 = len(group1), len(group2)
#         print(f"Column: {col}, Ukuran group1 (fall): {size1}, Ukuran group2 (summer): {size2}")
        
#         if size1 > 10 and size2 > 10:
#             stat, p = safe_levene(group1, group2)
#             if np.isnan(stat):
#                 print(f"Column {col}: Variansi nol, tidak dapat diuji.\n")
#             else:
#                 print(f"Statistics = {stat:.2f}, p = {p:.4f}")
#                 print(f"Column {col}: {'Varians antar kelompok homogen' if p > 0.05 else 'Varians antar kelompok tidak homogen'}\n")

#         elif size1 > 5 and size2 > 5:
#             result = permutation_test(
#                 (group1, group2), 
#                 statistic=lambda x, y: abs(x.var() - y.var()),
#                 permutation_type='independent'
#             )
#             print(f"Permutation Test P-Value: {result.pvalue:.4f}")
#             print(f"Column {col}: {'Varians antar kelompok homogen (Permutation Test)' if result.pvalue > 0.05 else 'Varians antar kelompok tidak homogen (Permutation Test)'}\n")
        
#         else:
#             print(f"Column {col}: Ukuran sampel terlalu kecil untuk uji statistik.\n")

#         print('-' * 50, '\n')
    
#     print('=' * 50, '\n')
#     print("===== Uji Perbandingan Rata-rata =====")
#     for col in num_data:
#         group1 = cc_df[cc_df['season'] == 'fall'][col].dropna().values
#         group2 = cc_df[cc_df['season'] == 'summer'][col].dropna().values
        
#         size1, size2 = len(group1), len(group2)
#         print(f"Column: {col}, Ukuran group1 (fall): {size1}, Ukuran group2 (summer): {size2}")
        
#         if size1 > 10 and size2 > 10:
#             stat, p = mannwhitneyu(group1, group2, alternative='two-sided')
#             print(f"Statistics = {stat:.2f}, p = {p:.4f}")
#             print(f"Column {col}: {'Tidak ada perbedaan signifikan' if p > 0.05 else 'Ada perbedaan signifikan'}\n")
        
#         elif size1 > 5 and size2 > 5:
#             result = permutation_test(
#                 (group1, group2), 
#                 statistic=lambda x, y: abs(x.mean() - y.mean()),
#                 permutation_type='independent'
#             )
#             print(f"Permutation Test P-Value: {result.pvalue:.4f}")
#             print(f"Column {col}: {'Tidak ada perbedaan signifikan (Permutation Test)' if result.pvalue > 0.05 else 'Ada perbedaan signifikan (Permutation Test)'}\n")
        
#         else:
#             print(f"Column {col}: Ukuran sampel terlalu kecil untuk uji statistik.\n")

#         print('-' * 50, '\n')



from scipy.stats import anderson, normaltest, levene, mannwhitneyu, permutation_test
import numpy as np
import pandas as pd

def safe_anderson(data):
    if np.std(data) == 0:
        return np.nan, np.nan
    
    result = anderson(data, dist='norm')
    return result.statistic, result.critical_values[2]

def safe_normaltest(data):
    if np.std(data) == 0:
        return np.nan, 1.0
    
    return normaltest(data)

def safe_levene(group1, group2):
    if np.std(group1) == 0 or np.std(group2) == 0:
        return np.nan, 1.0
        
    return levene(group1, group2)

def analyze_numeric_features(cc_df):
    exc_data = ['credit_card', 'long', 'lat', 'zipcode', 'year']
    num_data = cc_df.select_dtypes(include=['number']).drop(columns=exc_data, errors='ignore')
    
    normal_columns = []
    non_normal_columns = []
    
    print("===== Uji Normalitas Data =====")
    for col in num_data:
        data = cc_df[col].dropna().values
        if len(data) < 8:
            continue

        stat, crit_5_percent = safe_anderson(data)
        stat_normtest, p_normtest = safe_normaltest(data)

        if (stat < crit_5_percent) and (p_normtest > 0.05):
            normal_columns.append(col)
        else:
            non_normal_columns.append(col)

    print("\nKolom dengan distribusi normal:", normal_columns)
    print("Kolom dengan distribusi tidak normal:", non_normal_columns)
    print('-' * 50, '\n')
    
    print("===== Uji Homogenitas Varians =====")
    for col in num_data:
        group1 = cc_df[cc_df['season'] == 'fall'][col].dropna().values
        group2 = cc_df[cc_df['season'] == 'summer'][col].dropna().values

        if len(group1) > 10 and len(group2) > 10:
            stat, p = safe_levene(group1, group2)
            print(f"Column {col}: \t{'Varians antar kelompok homogen' if p > 0.05 else 'Varians antar kelompok tidak homogen'}")
    print('-' * 50, '\n')

    print("===== Uji Perbandingan Rata-rata =====")
    for col in num_data:
        group1 = cc_df[cc_df['season'] == 'fall'][col].dropna().values
        group2 = cc_df[cc_df['season'] == 'summer'][col].dropna().values

        if len(group1) > 10 and len(group2) > 10:
            stat, p = mannwhitneyu(group1, group2, alternative='two-sided')
            print(f"Column {col}: \t{'Tidak ada perbedaan signifikan' if p > 0.05 else 'Ada perbedaan signifikan'}")
    print('-' * 50, '\n')
