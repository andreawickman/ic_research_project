from scipy.interpolate import splrep, BSpline, make_lsq_spline
from scipy.optimize import minimize
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
import logging

logging.basicConfig(level=logging.INFO)


def bspline_smoothing(df, x, y, smoothing_factor=1, k = 3, t = None):
    '''
    Input: Noisy dataset 
    Returns: B-Spline Smoothed response variable y
    '''

    t, c, k = splrep(df[x], df[y], s=smoothing_factor, t = t)
    bspline =  BSpline(t, c, k, extrapolate=False)

    return bspline(df[x])

def bspline_smoothing_MAF(df, x, y, smoothing_factor=1, k = 3, t = None, optimize_knots=False, num_knots=10):
    '''
    Input: Noisy dataset 
    Returns: B-Spline Smoothed response variable y
    '''
    
    df = df.dropna(subset=[x, y])
    sorted_df = df.sort_values(by=x).drop_duplicates(subset=[x])
    x_vals = sorted_df[x].values
    y_vals = sorted_df[y].values

    mask = np.isfinite(x_vals) & np.isfinite(y_vals)
    x_vals = x_vals[mask]
    y_vals = y_vals[mask]

    if optimize_knots:
        t = np.linspace(x_vals.min(), x_vals.max(), num_knots - 2 * k)
        t = np.concatenate(([x_vals.min()] * k, t, [x_vals.max()] * k))
        tck = make_lsq_spline(x_vals, y_vals, t, k)
        bspline = BSpline(tck.t, tck.c, tck.k, extrapolate=False)
        knots = tck.t
    else:
        t, c, k = splrep(x_vals, y_vals, s=smoothing_factor, t=t)
        bspline = BSpline(t, c, k, extrapolate=False)
        knots = t
    
    return bspline(x_vals), knots


def bspline_smoothing_automatic_knots(df, x, y, num_knots, smoothing_factor=1):
    try:
        # Ensure the data is sorted
        df = df.sort_values(by=x)
        # Check for NaN values and constant values
        if df[x].isna().any() or df[y].isna().any():
            logging.warning(f"NaN values found in group {df['RENAME_ID'].iloc[0]}")
            return np.full(df.shape[0], np.nan), np.full(num_knots + 3, np.nan)
        if df[x].nunique() == 1 or df[y].nunique() == 1:
            logging.warning(f"Constant values found in group {df['RENAME_ID'].iloc[0]}")
            return np.full(df.shape[0], np.nan), np.full(num_knots + 3, np.nan)

        # Log min and max values of xcoord for each group
        min_x, max_x = df[x].min(), df[x].max()
        #logging.info(f"Group {df['RENAME_ID'].iloc[0]}: min_x = {min_x}, max_x = {max_x}")

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
        
        y_train_smoothed, knots = bspline_smoothing_MAF(train, x, y, optimize_knots=True, num_knots=num_knots)
        
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
'''
def check_knots(knots, x_vals):
    min_x, max_x = x_vals.min(), x_vals.max()
    return [k for k in knots if min_x <= k <= max_x]'''

def check_knots(knots, data_points):
    return [knot for knot in knots if knot in data_points]

def pspline_penalized_smoothing(df, x, y, num_knots, lambda_penalty):
    """
    Fit a penalized B-spline (P-spline) to the data.

    Parameters:
    df : pandas.DataFrame
        DataFrame containing the data.
    x : str
        Name of the column for the predictor variable.
    y : str
        Name of the column for the response variable.
    num_knots : int
        Number of knots to use for the spline.
    lambda_penalty : float
        Smoothing parameter for penalization.

    Returns:
    smoothed_values : np.ndarray
        Smoothed values of the response variable.
    c_opt : np.ndarray
        Optimized coefficients of the B-spline.
    """
    # Sort the data by the predictor variable
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
    
    # Define the knots uniformly over the range of the predictor variable
    knots = np.linspace(x_vals.min(), x_vals.max(), num_knots)[1:-1]
    
    # Initial spline fitting to obtain knots and coefficients
    t, c, k = splrep(x_vals, y_vals, t=knots, k=3)
    
    # Define the objective function for penalization
    def objective(c):
        spline = BSpline(t, c, k, extrapolate=False)
        residuals = y_vals - spline(x_vals)
        penalty = lambda_penalty * np.sum(np.diff(c, 2)**2)
        return np.sum(residuals**2) + penalty
    
    # Optimize the spline coefficients with the penalty term
    result = minimize(objective, c)
    c_opt = result.x
    
    # Generate the smoothed values using the optimized coefficients
    bspline = BSpline(t, c_opt, k, extrapolate=False)
    smoothed_values = bspline(x_vals)
    
    return smoothed_values, c_opt

def apply_pspline_smoothing(df, x, y, num_knots, lambda_penalty):
    """
    Apply P-spline smoothing to a grouped DataFrame.

    Parameters:
    df : pandas.DataFrame
        DataFrame containing the data.
    x : str
        Name of the column for the predictor variable.
    y : str
        Name of the column for the response variable.
    num_knots : int
        Number of knots to use for the spline.
    lambda_penalty : float
        Smoothing parameter for penalization.

    Returns:
    smoothed_values : pd.Series
        Series of smoothed values of the response variable.
    coefficients : pd.DataFrame
        DataFrame of optimized coefficients of the B-spline for each group.
    """
    # Apply the P-spline smoothing function to each group in the DataFrame
    smoothed_values = df.groupby('RENAME_ID', group_keys=False).apply(
        lambda group: pd.Series(pspline_penalized_smoothing(group, x, y, num_knots, lambda_penalty)[0], index=group.index)
    )
    coefficients = df.groupby('RENAME_ID').apply(
        lambda group: pd.Series(pspline_penalized_smoothing(group, x, y, num_knots, lambda_penalty)[1])
    )
    
    return smoothed_values, coefficients

def expand_coefficients(df, coeffs, prefix):
    coeffs_df = coeffs.reset_index()
    coeffs_df.columns = [f'{prefix}_coeff_{col}' if col != 'RENAME_ID' else 'RENAME_ID' for col in coeffs_df.columns]
    coeffs_df = coeffs_df.iloc[:, :-4]  # Drop the last four columns
    return pd.merge(df, coeffs_df, on='RENAME_ID', how='left')