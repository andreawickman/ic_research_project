import numpy as np

def first_derivative(df, x, y):
    return np.gradient(df[y], df[x])

def second_derivative(df, x, y):
    first_deriv = np.gradient(df[y], df[x])
    return np.gradient(first_deriv, df[x])