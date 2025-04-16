import utils.descriptive_stats as ds
import numpy as np


#TODO: make your own exceptions
def cov(data1 :ds.data, data2: ds.data):
    if not(data1.length() == data2.length()):
        raise()
    
    #TODO: check the data sanity
    

    data1_mean = data1.get_data() - (data1.mean())
    data2_mean = data2.get_data() - (data2.mean())

    return np.sum(data1_mean * data2_mean)/(data1.length()-1)

def corr(data1: ds.data, data2: ds.data):
    return (cov(data1, data2))/(data1.sd()*data2.sd())

