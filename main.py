from utils.descriptive_stats import *
from utils.helper import *
from stats.basic import *
import numpy as np
from ML.supervised.regression.linear_regression import linear_regression

if __name__ == "__main__":
    data1 = data(np.array([1,2,2,4,5,6,6,6,9,10]))
    pmf = gen_pmf(data1)
    pmf.plot()
    get_cdf = gen_cdf(data1)
    get_cdf.plot()
    x = np.array([1,2,3,4,5,6,7,8,9,10])
    y = np.array([1,2,3,3.5,5,6,7.6,8,9.2,10.55])
    model = linear_regression(x=x, y=y)
    model.plot()

