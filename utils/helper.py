from utils import descriptive_stats as ds
import numpy as np
import scipy.stats as stats

#TODO: implement different types of percentile

def percentile(data: ds.data, x: int):
    stats.percentileofscore(data.get_data(), x)
