from typing import Tuple
from numpy.typing import NDArray
import numpy as np
import scipy.special as special
import scipy.optimize as optimize
from mpmath import besseli, fabs, arg
from .vectorize_mp import *


def experimental_to_bode(freq, time, strain, stress, mode='c'):
    mode = mode
    s = freq * 2 * np.pi * 1j
    t = time
    eps0 = strain[1:]
    eps1 = strain[0:-1]
    phi0 = stress[1:]
    phi1 = stress[0:-1]
    t0 = t[1:]
    t1 = t[0:-1]
    phase = []
    gain = []
    for x in s:
        e0 = -1 * x * t0
        e1 = -1 * x * t1
        b = np.exp(e1) - np.exp(e0)
        tempStrain = ((eps0 - eps1) / (x**2 * (t0 - t1))) * b
        tempStress = ((phi0 - phi1) / (x**2 * (t0 - t1))) * b
        tempEps = sum(tempStrain)
        tempPhi = sum(tempStress)
        if mode == 'c':
            a = tempEps / tempPhi
        else:
            a = tempPhi / tempEps
        gain.append(20 * np.log(float(fabs(a))))
        # gain.append(float(fabs(a)))
        d = arg(a) * 180.0 / np.pi
        phase.append(float(d))
    return gain, phase


def ucc_ramp_hold_model(time, params, root, rampTime):
    Al, Bl, Cl, th = params
    rampMask = (time <= rampTime)
    time1 = time[rampMask] / th
    t = time / th
    summed1 = []
    summed2 = []
    root1 = root
    r2 = np.float64(root1**2)
    t0 = rampTime / th
    for x in t:
        if x <= t0:
            r1 = np.exp(np.float64(-1 * x * r2))
            r1 = np.nan_to_num(r1)
            d = (r2 * ((r2 / (2 * Al)) + (Al / (2.0)) - 1))
            b = 1 / d
            temp = r1 * b
            y = np.sum(temp)
            summed1.append(y)
        else:
            r11 = (
                np.exp(np.float64(-1 * x * r2)) -
                np.exp(np.float64(t0 * r2)) * np.exp(np.float64(-1 * x * r2)))
            r1 = np.nan_to_num(r11)
            d = (r2 * ((r2 / (2 * Al)) + (Al / (2.0)) - (1)))
            b = 1 / d
            y = np.sum(r1 * b)
            summed2.append(y)
    summed1 = np.array(summed1)
    summed2 = np.array(summed2)
    phi1 = ((1 - Bl) /
            (4 * (Al - 2) * t0)) + (time1 / t0) + ((Bl - 1) *
                                                   (1 -
                                                    (2 / Al)) / t0) * summed1
    phi2 = 1 + ((Bl - 1) * (1 - (2 / Al)) / t0) * summed2
    modelPhi = np.append(phi1, phi2)
    return modelPhi


def ucc_cle_3param(
        mode,
        rad,
        freq,
        params,
        Ey,
        imagin=1) -> (list[list[int]] or list[list[int], list[int]]):
    Eyc = Ey
    H_t, lam2, kr = params[0], params[1], params[2]
    H_c = Ey + 2 * lam2**2 / (H_t + lam2)
    th = rad**2 / (H_t * kr)
    A = 1 - (lam2 / H_t)
    B = (2 * H_c - 3 * lam2 + H_t) / (2.0 * Eyc)
    C = -1 * A * (2 * lam2 - H_t - H_c) / Eyc
    s = freq * 2 * np.pi * th * imagin

    if np.imag(s[0]) != 0:
        ss = compSqrt(s)
        phi = (B * vBesseli(0, ss) * ss - C * vBesseli(1, ss))
        eps = (vBesseli(0, ss) * ss - A * vBesseli(1, ss))
        if mode == 'c':
            g = vDiv(eps, phi)
        else:
            g = vDiv(phi, eps)
        return [
            20 * np.log(np.asfarray(vFabs(g))),  ## Gain (dB)
            np.asfarray(vArg(g) * 180.0 / np.pi)  ## phase (deg)
        ]
    else:
        ss = np.sqrt(s)
        phi = (B * special.iv(0, ss) * ss - C * special.iv(1, ss))
        eps = (special.iv(0, ss) * ss - A * special.iv(1, ss))
        if mode == 'c':
            g = np.divide(eps, phi)
        else:
            g = np.divide(phi, eps)
        return [g]


def ucc_cle_4param(
        mode,
        rad,
        freq,
        params,
        imagin=1) -> (list[list[int]] or list[list[int], list[int]]):
    H_c, H_t, lam2, kr = params
    th = rad**2 / (H_t * kr)
    Eyc = H_c - (2 * lam2**2) / (H_t + lam2)
    A = 1 - (lam2 / H_t)
    B = (2 * H_c - 3 * lam2 + H_t) / (2.0 * Eyc)
    C = -1 * A * (2 * lam2 - H_t - H_c) / Eyc
    s = freq * 2 * np.pi * th * imagin

    if np.imag(s[0]) != 0:
        ss = compSqrt(s)
        phi = (B * vBesseli(0, ss) * ss - C * vBesseli(1, ss))
        eps = (vBesseli(0, ss) * ss - A * vBesseli(1, ss))
        if mode == 'c':
            g = vDiv(eps, phi)
        else:
            g = vDiv(phi, eps)
        return [
            20 * np.log(np.asfarray(vFabs(g))),  ## Gain (dB)
            np.asfarray(vArg(g) * 180.0 / np.pi)  ## phase (deg)
        ]
    else:
        ss = np.sqrt(s)
        phi = (B * special.iv(0, ss) * ss - C * special.iv(1, ss))
        eps = (special.iv(0, ss) * ss - A * special.iv(1, ss))
        if mode == 'c':
            g = np.divide(eps, phi)
        else:
            g = np.divide(phi, eps)
        return [g]


def create_dynamic_modulus(rad, Ey, H_t, lam2, kr):
    H_c = Ey + 2 * lam2**2 / (H_t + lam2)
    th = rad**2 / (H_t * kr)
    A = 1 - (lam2 / H_t)
    B = (2 * H_c - 3 * lam2 + H_t) / (2.0 * Ey)
    C = -1 * A * (2 * lam2 - H_t - H_c) / Ey

    def calc_value(s):
        try:
            ss = np.sqrt(s * th)
            num = (B * besseli(0, ss) * ss - C * besseli(1, ss))
            den = (besseli(0, ss) * ss - A * besseli(1, ss))
            a = num / den
        except:
            ss = compSqrt(s * th)
            num = (B * besseli(0, ss) * ss - C * besseli(1, ss))
            den = (besseli(0, ss) * ss - A * besseli(1, ss))
            a = num / den
        return a

    return calc_value


def calculate_biphasic_params(Ey, rad, x0):
    Ht, lam2, k = x0[0], x0[1], x0[2]
    Hc = Ey + 2 * lam2**2 / (Ht + lam2)
    A = 1 - (lam2 / Ht)
    B = (2 * Hc - 3 * lam2 + Ht) / (2.0 * Ey)
    C = -1 * A * (2 * lam2 - Ht - Hc) / Ey
    th = rad**2 / (Ht * k)
    return [A, B, C, th]


def obj_func_3param(x0, time, Ey):
    num1 = 20
    Ht, lam2, k = x0[0], x0[1], x0[2]
    Hc = Ey + 2 * lam2**2 / (Ht + lam2)
    if (lam2 > min(Hc, Ht)) | (lam2 < -0.5 * min(Hc, Ht)) | (
            np.abs(Hc - (Ey + (2 * lam2**2) / (Ht + lam2))) > 1000):
        res = 1e3
    else:
        p = self._calc_p(x0)
        roots = self._roots(num1, p)[1:]
        modelPhi = self._modelRamp(time, p, roots)
        # c3 = (np.dot(self.nstress, modelPhi)) ** 2 / (np.dot(self.nstress, self.nstress) * np.dot(modelPhi, modelPhi))
        res = sum((modelPhi - self.nstress)**2)
        # o = 1 - c3
    return res


def impulse(df, rootList, p):
    Al, Bl, Cl, th = p
    t = df.Time / th
    summed = []
    num1 = 100
    num2 = 100
    root1 = rootList[1:num1]
    root2 = rootList[1:num2]
    r2 = np.float64(root1**2)
    r22 = np.float64(root2**2)
    for x in t:
        if x < .1:
            r1 = np.exp(np.float64(-1 * x * r2))
            b = 1 / ((1 / (2 * Al)) + (Al / (r2 * 2.0)) - (1 / r2))
            y = np.sum(r1 * b)
            summed.append(y)
        else:
            r1 = np.exp(np.float64(-1 * x * r22))
            b = 1 / ((1 / (2 * Al)) + (Al / (r22 * 2.0)) - (1 / r22))
            y = np.sum(r1 * b)
            summed.append(y)
    summed = np.array(summed)
    phi1 = (((Bl - 1) * ((1 - (2 / Al)))) / th) * summed
    return phi1


def calcScale(p: Tuple[int, int, int], roots: NDArray):
    Al, Bl, Cl = p[0], p[1], p[2]
    num = 1000
    if len(roots) < num:
        num = len(roots)
    root = roots[1:num]
    r2 = np.float64(root**2)
    r1 = 1
    b = 1 / ((r2 / (2 * Al)) + (Al / (2.0)) - (1))
    y = np.sum(r1 * b)
    return 1 - ((Bl - 1) * (1 - (2.0 / Al)) * y)


def A(Ht, lam2):
    return 1 - (lam2 / Ht)


def B(Hc, Ht, lam2):
    Eyc = Hc - 2 * lam2**2 / (Ht + lam2)
    return (2 * Hc - 3 * lam2 + Ht) / (2.0 * Eyc)


def C(Hc, Ht, lam2):
    Eyc = Hc - 2 * lam2**2 / (Ht + lam2)
    return -1 * A(Ht, lam2) * (2 * lam2 - Ht - Hc) / Eyc


def roots(n: int, A: int):

    def f(r):
        return r * special.jn(0, (r)) - A * special.jn(1, (r))

    count = 1
    rL = [0]
    aL = [.01]
    bL = [.3]
    while count < n + 1:
        freq = np.linspace(bL[count - 1] + .01, bL[count - 1] + 10, 100)
        values = f(freq)
        for i in range(1, len(freq)):
            if (values[i - 1] * values[i]) < 0:
                a = freq[i - 1]
                b = freq[i]
                break
            else:
                continue
        aL.append(a)
        bL.append(b)
        rL.append(optimize.bisect(f, a, b))
        count += 1

    return np.asarray(rL)