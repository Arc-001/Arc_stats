import utils.descriptive_stats as ds
import numpy as np
import stats.discrete.probablity_mass_function as discrete

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

def gen_pmf(data: ds.data)-> discrete.pmf:
    temp = np.sort(data.get_data())
    x = np.unique(temp)
    prev = temp[0]
    idx_ = 0
    freq = np.array([0])
    for idx, value in enumerate(temp):
        if value == prev:
            freq[idx_]+=1
        else:
            freq = np.append(freq, 1)
            idx_+=1
            prev = value

    prob = freq/freq.sum()

    return discrete.pmf(x=x, probability=prob)

def gen_cdf(data: ds.data)-> discrete.pmf:
    return discrete.cdf(gen_pmf(data))


    


