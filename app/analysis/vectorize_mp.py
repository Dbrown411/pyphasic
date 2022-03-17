import mpmath as mp
import numpy as np

##vectorize some functions (effect on performance untested)
compSqrt = np.frompyfunc(mp.sqrt, 1, 1)
compExp = np.frompyfunc(mp.exp, 1, 1)
vBesseli = np.frompyfunc(mp.besseli, 2, 1)
vDiv = np.frompyfunc(mp.fdiv, 2, 1)
vFabs = np.frompyfunc(mp.fabs, 1, 1)
vArg = np.frompyfunc(mp.arg, 1, 1)
