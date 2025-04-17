import numpy as np 
import scipy as sp
import pandas as pd


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
    
class data_box:
    def __init__(self, data:pd.DataFrame):
        self.data = data
        self.col = data.columns
        self.row = data.index
        self.shape = (self.col, self.row)
    
    def transpose(self):
        return self.data.T
    
    def sum_col(self):
        return self.data.sum(axis=0)
    
    def sum_row(self):
        return self.data.sum(axis=1)
    
    def determinant(self):
        return np.linalg.det(self.data)
    

    
    def __mul__(self, data2):
        return self.data * data2.get_data()
    
    def __add__(self, data2):
        return self.data + data2.get_data()
    
    def __sub__(self, data2):
        return self.data - data2.get_data()
    
    def __truediv__(self, data2):
        return self.data / data2.get_data()
    
    def __floordiv__(self, data2):
        return self.data // data2.get_data()
    
    
class contengency_table:
    def __init__(self, f_xy:data_box, x:np.array, y:np.array):
        '''
        Args:
            f_xy (m,n) has the joint probablity
            x: (m,) is the first variable
            y: (n,) is the second variable
        '''
        self.f_xy = f_xy
        self.col = f_xy.col
        self.row = f_xy.row
        self.shape = (self.col, self.row)
        self.x = x
        self.y = y

    def marginal_x(self):
        return self.f_xy.sum_col()
    
    def marginal_y(self):
        return self.f_xy.sum_row()
    
    def check_independence(self, threshold=0.05):
        f_x = self.marginal_x()
        f_y = self.marginal_y()
        f_xy_ind = pd.DataFrame(np.outer(f_x, f_y).T)
        diff_f = np.abs(self.f_xy.data - f_xy_ind)
        diff = diff_f.sum().sum()
        if diff < threshold:
            return True
        else:
            return False


