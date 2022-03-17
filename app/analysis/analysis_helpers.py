from collections.abc import Callable
import numpy as np
from numpy.typing import NDArray
from .vectorize_mp import compExp


def calculate_r2(model: NDArray, exp: NDArray):
    mean_exp = np.mean(exp)
    num = np.sum((exp - model)**2)
    den = np.sum((exp - mean_exp)**2)
    try:
        r2 = 1 - num / den
    except:
        r2 = np.nan
    return r2


def norm_ramp_hold(t: NDArray):
    t[t >= 1] = 1
    return t


def interpolation_signal(time, signal) -> Callable:
    t = time
    sig0 = signal[1:]
    sig1 = signal[0:-1]
    t0 = t[1:]
    t1 = t[0:-1]

    def lap_transform(s):
        try:
            e0 = -1 * s * t0
            e1 = -1 * s * t1
            b = np.exp(e1) - np.exp(e0)
            y = sum(((sig0 - sig1) / (s**2 * (t0 - t1))) * b)
        except:
            b = compExp(e1) - compExp(e0)
            y = sum(((sig0 - sig1) / (s**2 * (t0 - t1))) * b)
        return y

    return lap_transform