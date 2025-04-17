from utils import descriptive_stats as ds
import numpy as np
import scipy.stats as stats

#TODO: implement different types of percentile

def percentile(data: ds.data, x: int):
    stats.percentileofscore(data.get_data(), x)

def L1_norm(**kwargs):
    """
    L1 norm of a vector
    :param kwargs: x: vector
    :return: L1 norm of the vector
    """
    x = kwargs.get('x')
    return np.sum(np.abs(x))

def L2_norm()
