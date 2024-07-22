from scipy.interpolate import splrep, BSpline, make_lsq_spline
import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error


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