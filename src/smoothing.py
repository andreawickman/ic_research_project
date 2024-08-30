from scipy.interpolate import splrep, BSpline, make_lsq_spline
from scipy.optimize import minimize
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
import logging
logging.basicConfig(level=logging.INFO)

#Function to Apply B-spline Smoothing
def bspline_smoothing_automatic_knots(df, x, y, num_knots, smoothing_factor=1):
    try:
        #ensure data is sorted by coordinate of genome
        df = df.sort_values(by=x)

        # Check for NaN values
        if df[x].isna().any() or df[y].isna().any():
            logging.warning(f"NaN values found in group {df['RENAME_ID'].iloc[0]}")
            return np.full(df.shape[0], np.nan), np.full(num_knots + 3, np.nan)
        if df[x].nunique() == 1 or df[y].nunique() == 1:
            logging.warning(f"Constant values found in group {df['RENAME_ID'].iloc[0]}")
            return np.full(df.shape[0], np.nan), np.full(num_knots + 3, np.nan)

        #get min and max values of cooridnates
        min_x, max_x = df[x].min(), df[x].max()

        # Check for low variation in x values
        if max_x - min_x < 1e-5:
            logging.warning(f"Low variation in x values for group {df['RENAME_ID'].iloc[0]}")
            return np.full(df.shape[0], np.nan), np.full(num_knots + 3, np.nan)

        # Generate knots
        knots = np.linspace(df[x].min(), df[x].max(), num_knots)[1:-1]
        # Fit B-spline
        t, xi, k = splrep(df[x], df[y], s=smoothing_factor, k=3, t=knots)
        bspline = BSpline(t, xi, k, extrapolate=False)
        smoothed_values = bspline(df[x])
        #logging.info(f"Coefficients for group {df['RENAME_ID'].iloc[0]}: {xi}")
        return smoothed_values, xi
    
    except Exception as e:
        print(f"Error in B-spline smoothing for {x}, {y} with {num_knots} knots: {e}")
        return np.full(df.shape[0], np.nan), np.full(num_knots + 3, np.nan) # Return NaN if smoothing failed

#Function to apply smoothing for each sample/participant individually
def apply_smoothing(df, x, y, num_knots):
    smoothed_values = df.groupby('RENAME_ID', group_keys=False).apply(
        lambda group: pd.Series(bspline_smoothing_automatic_knots(group, x, y, num_knots)[0], index=group.index)
    )
    xi = df.groupby('RENAME_ID').apply(
        lambda group: pd.Series(bspline_smoothing_automatic_knots(group, x, y, num_knots)[1])
    )
    return smoothed_values, xi

def evaluate_spline(df, x, y, num_knots, k=3):
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    mse_list = []
    aic_list = []
    bic_list = []
    
    for train_index, test_index in kf.split(df):
        train, test = df.iloc[train_index], df.iloc[test_index]
        y_train_smoothed, knots = bspline_smoothing_automatic_knots(train, x, y, num_knots=num_knots)        
        test_x = test[x].values
        test_y = test[y].values

        mask = np.isfinite(test_x) & np.isfinite(test_y)
        test_x = test_x[mask]
        test_y = test_y[mask]
        
        bspline_test = BSpline(knots, y_train_smoothed, k, extrapolate=False)
        y_test_pred = bspline_test(test_x)
        
        mse = mean_squared_error(test_y, y_test_pred)
        mse_list.append(mse)
        
        n = len(test_y)
        p = len(knots)
        
        residuals = test_y - y_test_pred
        sse = np.sum(residuals**2)
        
        aic = n * np.log(sse/n) + 2 * p
        bic = n * np.log(sse/n) + np.log(n) * p
        
        aic_list.append(aic)
        bic_list.append(bic)
    
    return np.mean(mse_list), np.mean(aic_list), np.mean(bic_list)

def check_knots(knots, data_points):
    return [knot for knot in knots if knot in data_points]

#Function to Apply P-spline smoothing
def pspline_penalized_smoothing(df, x, y, num_knots, lambda_penalty):
    #sort by genome coordinates 
    df = df.sort_values(by=x)
    
    # Check for NaN values and constant values
    if df[x].isna().any() or df[y].isna().any():
        print(f"NaN values found in group {df['RENAME_ID'].iloc[0]}")
        return np.full(df.shape[0], np.nan), np.full(num_knots + 3, np.nan)
    if df[x].nunique() == 1 or df[y].nunique() == 1:
        print(f"Constant values found in group {df['RENAME_ID'].iloc[0]}")
        return np.full(df.shape[0], np.nan), np.full(num_knots + 3, np.nan)
    
    # Extract the values of the predictor and response variables
    x_vals = df[x].values
    y_vals = df[y].values
    
    # Define the knots
    knots = np.linspace(x_vals.min(), x_vals.max(), num_knots)[1:-1]
    
    # Initial spline fitting - obtain knots and coefficients
    t, c, k = splrep(x_vals, y_vals, t=knots, k=3)
    
    # define bjective function for penalization
    def objective(c):
        spline = BSpline(t, c, k, extrapolate=False)
        residuals = y_vals - spline(x_vals)
        penalty = lambda_penalty * np.sum(np.diff(c, 2)**2)
        return np.sum(residuals**2) + penalty
    
    # Optimize the spline coefficients with penalty term
    result = minimize(objective, c)
    c_opt = result.x
    
    # Generate the smoothed values using the optimized coefficients.
    bspline = BSpline(t, c_opt, k, extrapolate=False)
    smoothed_values = bspline(x_vals)
    
    return smoothed_values, c_opt

#Function to apply P-spline smoothing for each sample individually 
def apply_pspline_smoothing(df, x, y, num_knots, lambda_penalty):
    smoothed_values = df.groupby('RENAME_ID', group_keys=False).apply(
        lambda group: pd.Series(pspline_penalized_smoothing(group, x, y, num_knots, lambda_penalty)[0], index=group.index)
    )
    coefficients = df.groupby('RENAME_ID').apply(
        lambda group: pd.Series(pspline_penalized_smoothing(group, x, y, num_knots, lambda_penalty)[1])
    )
    return smoothed_values, coefficients

#expand coefficients to get Spline Coefficients 
def expand_coefficients(df, coeffs, prefix):
    coeffs_df = coeffs.reset_index()
    coeffs_df.columns = [f'{prefix}_coeff_{col}' if col != 'RENAME_ID' else 'RENAME_ID' for col in coeffs_df.columns]
    coeffs_df = coeffs_df.iloc[:, :-4]  # Drop the last four columns
    return pd.merge(df, coeffs_df, on='RENAME_ID', how='left')

