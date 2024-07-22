import numpy as np
import pandas as pd

def stratify5(df):
    intervals = [
        df['TSI_days'] <= 180.0,
        (df['TSI_days'] > 180.0) & (df['TSI_days'] <= 365.0),
        (df['TSI_days'] > 365.0) & (df['TSI_days'] <= 730.0),
        (df['TSI_days'] > 730) & (df['TSI_days'] <= 1460),
        df['TSI_days'] > 1460.0
    ]
    categories = ['0m-6m', '6m-12m', '12m-24m', '2y-4y', '4y+']
    df['TSI_category'] = np.select(intervals, categories, default=np.nan)
    df['TSI_category'] = pd.Categorical(df['TSI_category'], categories=categories, ordered=True)
    return df

def stratify4(df):
    intervals = [
        df['TSI_days'] <= 180.0,
        (df['TSI_days'] > 180.0) & (df['TSI_days'] <= 365.0),
        (df['TSI_days'] > 365.0) & (df['TSI_days'] <= 1460.0),
        df['TSI_days'] > 1460.0
    ]
    categories = ['0m-6m', '6m-12m', '1y-4y', '4y+']
    df['TSI_category'] = np.select(intervals, categories, default=np.nan)
    df['TSI_category'] = pd.Categorical(df['TSI_category'], categories=categories, ordered=True)
    return df