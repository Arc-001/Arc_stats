import numpy as np 
import scipy as sp


class data:
    def __init__(self, x1):
        self.x = x1

    def length(self):
        return len(self.x)

    #TODO: resolve datatypes and exceptions
    def __mul__(self, data2):
        return (self.x * data2.get_data())

    def get_data(self):
        return self.x

    def mean(self):
        return np.mean(self.x)

    def var(self):
        mean = self.mean()
        return (np.sum(np.square(self.x - self.mean())))/(len(self.x)-1)
    
    def sd(self):
        return np.sqrt(self.var())
    

