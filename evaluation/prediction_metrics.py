import numpy as np
from sklearn.metrics import r2_score

def rmse(y_true, y_pred):
    """Root Mean Squared Error"""
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.sqrt(np.mean((y_true - y_pred) ** 2))

def mae(y_true, y_pred):
    """Mean Absolute Error"""
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs(y_true - y_pred))

def mse(y_true, y_pred):
    """Mean Squared Error"""
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean((y_true - y_pred) ** 2)

def r2(y_true, y_pred):
    """R-squared score"""
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return r2_score(y_true, y_pred)

def nasa_rul_score(y_true, y_pred):
    """
    NASA RUL scoring function (used for C-MAPSS)
    Penalizes late predictions more than early ones
    """
    score = 0.0
    for t, p in zip(y_true, y_pred):
        diff = p - t
        if diff < 0:
            score += np.exp(-diff / 13) - 1
        else:
            score += np.exp(diff / 10) - 1
    return score
