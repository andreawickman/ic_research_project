import numpy as np
import pandas as pd

#For 5-Class Classification System
def stratify5(df):
    #define intervals for 5 classes
    intervals = [
        df['TSI_days'] <= 180.0,
        (df['TSI_days'] > 180.0) & (df['TSI_days'] <= 365.0),
        (df['TSI_days'] > 365.0) & (df['TSI_days'] <= 730.0),
        (df['TSI_days'] > 730) & (df['TSI_days'] <= 1460),
        df['TSI_days'] > 1460.0
    ]
    categories = ['0m-6m', '6m-12m', '12m-24m', '2y-4y', '4y+']
    #add class column
    df['TSI_category'] = np.select(intervals, categories, default=np.nan)
    df['TSI_category'] = pd.Categorical(df['TSI_category'], categories=categories, ordered=True)
    return df

#For 4-Class Classification System
def stratify4(df):
    #define intervals for 4 classes
    intervals = [
        df['TSI_days'] <= 180.0,
        (df['TSI_days'] > 180.0) & (df['TSI_days'] <= 365.0),
        (df['TSI_days'] > 365.0) & (df['TSI_days'] <= 1460.0),
        df['TSI_days'] > 1460.0
    ]
    categories = ['0m-6m', '6m-12m', '1y-4y', '4y+']
    #add class column
    df['TSI_category'] = np.select(intervals, categories, default=np.nan)
    df['TSI_category'] = pd.Categorical(df['TSI_category'], categories=categories, ordered=True)
    return df

#function for batch-processing the imputation process
def batch_process(data, batch_size, imputer, scaler):
    n_batches = int(np.ceil(data.shape[0] / batch_size))
    imputed_data = []
    
    for i in range(n_batches):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, data.shape[0])
        batch_data = data[start_idx:end_idx]
        
        scaled_batch_data = scaler.transform(batch_data)
        imputed_batch_data = imputer.fit_transform(scaled_batch_data)
        imputed_batch_data_original_scale = scaler.inverse_transform(imputed_batch_data)
        
        imputed_data.append(imputed_batch_data_original_scale)
    
    return np.vstack(imputed_data)