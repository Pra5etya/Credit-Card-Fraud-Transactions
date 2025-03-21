import pandas as pd
import numpy as np
from geopy.distance import geodesic

def preprocess_credit_card_data(df):
    # Cek nilai null sebelum proses
    print("Missing values before processing:")
    print(df.isnull().sum())
    
    # Cek duplikasi sebelum proses
    print('=' * 50, '\n')
    print(f"Total duplicate data before processing: {df.duplicated().sum()}")
    
    # Timestamp processing
    df['datetime'] = pd.to_datetime(df['datetime'], format='mixed', dayfirst=True)
    df['date'] = df['datetime'].dt.date
    df['year'] = df['datetime'].dt.year
    df['month'] = df['datetime'].dt.month_name().str.lower()
    df['quarter'] = df['datetime'].dt.to_period('Q').astype(str)
    df['day'] = df['datetime'].dt.day_name().str.lower()
    df['time'] = df['datetime'].dt.time
    df['week_cat'] = np.where(df['day'].isin(['monday', 'tuesday', 'wednesday', 'thursday', 'friday']), 'weekday', 'weekend')
    
    # Mapping for Seasons
    season_mapping = {
        'spring': [('03-01', '05-31')],
        'summer': [('06-01', '08-31')],
        'fall': [('09-01', '11-30')],
        'winter': [('12-01', '12-31'), ('01-01', '02-29')]
    }
    
    def get_season(date):
        month_day = date.strftime('%m-%d')
        for season, ranges in season_mapping.items():
            for start, end in ranges:
                if (start <= month_day <= end) or (start > end and (month_day >= start or month_day <= end)):
                    return season
    
    df['season'] = df['datetime'].apply(get_season)
    
    # Categorization based on credit limit
    bins = [10000, 20000, 30000, 40000]
    categories = ["very_low", "low", "medium", "high", "very_high"]
    credit_categories = np.digitize(df['credit_card_limit'], bins)
    df['limit_cat'] = np.array(categories)[credit_categories]
    
    # Process string columns
    string_columns = df.select_dtypes(include=['object']).columns.tolist()
    excluded_columns = ['quarter', 'week_cat', 'season', 'limit_cat']
    string_columns = [col for col in string_columns if col not in excluded_columns]
    
    for column in string_columns:
        if column in df.columns:
            df[column] = df[column].astype(str).str.lower().str.replace(r'\s{2,}', ' ', regex=True)
    
    # Fraud Detection - Checking Duplicates
    duplicates_subset = df[df.duplicated(subset=['datetime', 'credit_card'], keep=False)]
    duplicates_subset = duplicates_subset.sort_values(by=['datetime', 'credit_card'])
    
    # 
    print(f'Total duplicate transactions: {len(duplicates_subset)}')
    
    df['transaction_count'] = duplicates_subset.groupby(['credit_card', 'datetime'])['datetime'].transform('count')
    df['transaction_count'] = df['transaction_count'].fillna(1)
    
    # Time-based fraud detection
    df['prev_time'] = df.groupby('credit_card')['datetime'].shift(1).fillna(df['datetime'])
    df['time_diff_hour'] = abs(((df['datetime'] - df['prev_time']).dt.total_seconds()) / 3600).fillna(0)
    
    # Geolocation-based fraud detection
    df['prev_long'] = df.groupby('credit_card')['long'].shift(1).fillna(0)
    df['prev_lat'] = df.groupby('credit_card')['lat'].shift(1).fillna(0)
    df['distance_km'] = df.apply(lambda row: 0 if pd.isnull(row['prev_long']) or pd.isnull(row['prev_lat']) 
                                    else geodesic((row['lat'], row['long']), (row['prev_lat'], row['prev_long'])).kilometers, axis=1).fillna(0)
    
    df['geo_cat'] = np.where((df['distance_km'] > 500) | (df['time_diff_hour'] < 1), 'anomaly', 'normal')
    df['speed_km/h'] = (df['distance_km'] / df['time_diff_hour']).fillna(0).round(2)
    
    # Fraud classification
    cc_limit_sus = df['transaction_dollar_amount'] > df['credit_card_limit'] * 0.8
    cc_trx_sus = df['transaction_dollar_amount'] > df['transaction_dollar_amount'].mean() * 5
    df['fraud'] = np.where((df['geo_cat'] == 'anomaly') | (df['time_diff_hour'] < 1) |
                          (df['speed_km/h'] > 50) | (cc_limit_sus) | (cc_trx_sus), 'fraud', 'not_fraud')
    
    # Tetapkan hanya sebagian kecil data sebagai fraud (misalnya 5%)
    fraud_mask = df['fraud'] == 'fraud'
    fraud_indices = df[fraud_mask].sample(frac=0.05, random_state=42).index
    df['fraud'] = 'not_fraud'
    df.loc[fraud_indices, 'fraud'] = 'fraud'

    # Handling missing values and infinite values
    df.replace([np.inf, -np.inf], 0, inplace=True)
    
    # Fraud summary
    print('=' * 50, '\n')
    print("Fraud cases percentage:")
    print(df['fraud'].value_counts(normalize=True) * 100)
    
    print('=' * 50, '\n')
    print("Fraud cases count:")
    print(df['fraud'].value_counts())
    
    return df
