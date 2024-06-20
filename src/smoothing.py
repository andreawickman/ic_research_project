from scipy.interpolate import splrep, splev, BSpline

def bspline_smoothing(df, x, y, smoothing_factor=1, k = 3, t = None):
    '''
    Input: Noisy dataset 
    Returns: B-Spline Smoothed response variable y
    '''
    t, c, k = splrep(df[x], df[y], s=smoothing_factor, t = t)
    bspline =  BSpline(t, c, k, extrapolate=False)

    return bspline(df[x])