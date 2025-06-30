import numpy as np

def PK(t, D0=1.0, alpha=1 / 60 / 8, texp=3 * 60):
    return D0 * (t < texp) + D0 * np.exp(-alpha * (t - texp)) * (t >= texp)
