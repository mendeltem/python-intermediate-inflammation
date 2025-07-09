import numpy as np
import matplotlib.pyplot as plt
import os

from inflammation.models import daily_max, daily_mean, daily_min, load_csv

data=np.loadtxt(fname="data/inflammation-01.csv", delimiter=",")

print(data)
print(data.shape)

daily_min_var = daily_min(data)
daily_max_var = daily_max(data)
daily_mean_var = daily_mean(data)

print("Daily Min:", daily_min_var)
print("Daily Max:", daily_max_var)          
print("Daily Mean:", daily_mean_var)

import numpy.testing as npt

test_input = np.array([[2,0],[4,0]])
test_result = np.array([3,0])

npt.assert_array_equal(daily_mean(test_input), test_result )

