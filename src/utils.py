import numpy as np 
import pandas as pd

def compute_vector_change(v1, v2):
    """Computes the angular difference between two vectors."""
    dot = np.dot(v1, v2)
    norm = np.linalg.norm(v1) * np.linalg.norm(v2)
    if norm == 0:
        return 0
    return np.degrees(np.arccos(np.clip(dot / norm, -1.0, 1.0)))

def moving_average(series, window = 3):
    """Smooth out frame-level noise"""
    return series.rolling(window=window, min_periods=1, center = True).mean()

def get_accel_vectors(df):
    """Compute acceleration vector (ax, ay) from a +dir"""
    df["ax"] = df["a"] * np.cos(np.deg2rad(df["dir"]))
    df["ay"] = df["a"] * np.sin(np.deg2rad(df["dir"]))
    return df

