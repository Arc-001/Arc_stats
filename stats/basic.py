from utils import descriptive_stats as ds
import numpy as np


#TODO: make your own exceptions
def cov(data1 :ds.data, data2: ds.data):
    if not(len(data1) == len(data2)):
        raise()
    
    #TODO: check the data sanity
    

    data1_mean = data1.get_data() - (data1.mean())
    data2_mean = data2.get_data() - (data2.mean())

    return np.sum(data1_mean * data2_mean)/len(data2_mean)

