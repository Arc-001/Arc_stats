import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt

class pmf:
    def __init__(self, **kwargs):
        #-----------------------------------------------------------------#
        '''
        args:
        keyword arguments:
            x : list of x lables
            probability : list of probabilities for each lable/x
        '''
        #------------------------------------------------------------------#
        self.x = np.array(kwargs['x'])
        self.p = np.array(kwargs['probability'])
        # if not self.validate():
        #     raise ValueError("Invalid PMF")
        self.sort()
        self.len = len(self.x)
    

    def validate(self) -> bool:
        return (len(self.p) == len(self.x)) and (np.sum(self.p) == 1)

    def expectation(self) -> np.float64:
        return np.sum(self.x * self.p)
    
    def moment(self, k) -> np.float64:
        return np.sum(self.x**k * self.p)
    
    def func_expectation(self, func) -> np.float64:
        return np.sum(func(self.x) * self.p)
    
    def var(self) -> np.float64:
        return self.moment(2) - self.expectation()**2
    
    def SD(self) -> np.float64:
        return np.sqrt(self.var())
    
    def plot(self, **kwargs):
        plt.bar(self.x, self.p, **kwargs)
        plt.xlabel('x')
        plt.ylabel('p(x)/P(X=x)/f(x)')
        plt.xticks(self.x)
        plt.yticks(self.p)
        plt.grid()
        plt.title('Probability Mass Function')
        plt.show()

    #------------------------------------------#
    # Internal Sort for the PMF for effeciency #
    # -----------------------------------------#

    def sort(self):
        #sorts the index based on the x values
        idx = np.argsort(self.x)

        #sorts the x and p values based on the index
        self.x = self.x[idx]
        self.p = self.p[idx]

class cdf(pmf):
    def __init__(self, pmf_obj: pmf):
        self.x = pmf_obj.x
        self.p = pmf_obj.p
        self.len = pmf_obj.len
        self.cdf = np.cumsum(self.p)

    def F(self, x):
        # F(x) = P(X <= x)
        # returns the value of cdf at x
        if x < self.x[0]:
            return 0
        elif x > self.x[-1]:
            return 1
        else:
            idx = np.searchsorted(self.x, x)
            return self.cdf[idx-1]
        
    def F_interpolate(self, x):
        # F(x) = P(X <= x)
        # returns the value of cdf at x
        if x < self.x[0]:
            return 0
        elif x > self.x[-1]:
            return 1
        else:
            idx = np.searchsorted(self.x, x)
            return np.interp(x, self.x[idx-1:idx+1], self.cdf[idx-1:idx+1])
        
    

    def plot(self, **kwargs):
        plt.plot(self.x, self.cdf, **kwargs)
        plt.xlabel('x')
        plt.ylabel('F(x)')
        plt.xticks(self.x)
        plt.yticks(self.cdf)
        plt.title('Cumulative Distribution Function')
        plt.show()

def bernoulli(p) ->pmf:
    #-----------------------------------------------------------------#
    '''
    args:
        p: probability of success
    '''
    #------------------------------------------------------------------#
    x = [0,1]
    p = [1-p,p]

    return pmf(x=x, probability=p)


def binomial(n, p) -> pmf:
    #-----------------------------------------------------------------#
    '''
    args:
        n: number of trials
        p: probability of success
    '''
    #------------------------------------------------------------------#
    x = np.arange(0, n+1)
    p = stats.binom.pmf(x, n, p)
    return pmf(x=x, probability=p)

def geometric(p) -> pmf:
    #-----------------------------------------------------------------#
    '''
    args:
        p: probability of success
    '''
    #------------------------------------------------------------------#
    x = np.arange(0, 100)
    p = stats.geom.pmf(x, p)
    return pmf(x=x, probability=p)

def poisson(lam) -> pmf:
    #-----------------------------------------------------------------#
    '''
    args:
        lam: rate of success

    same as binomial but with n tending to infinity and p tending to 0
    '''
    #------------------------------------------------------------------#
    x = np.arange(0, 100)
    p = stats.poisson.pmf(x, lam)
    return pmf(x=x, probability=p)


def hypergeometric(N, K, n) -> pmf:
    #-----------------------------------------------------------------#
    '''
    args:
        N: population size
        K: number of success in population
        n: number of draws
    '''
    #--------------------------------------------------------------#
    x = np.arange(0, n+1)
    p = stats.hypergeom.pmf(x, N, K, n)
    return pmf(x=x, probability=p)

def multinomial(n, p) -> pmf:
    #--------------------------------------------------------------#
    '''
    args:
        n: number of trials
        p: probability of success
    '''
    #------------------------------------------------------------------#
    x = np.arange(0, n+1)
    p = stats.multinomial.pmf(x, n, p)
    return pmf(x=x, probability=p)

def negative_binomial(n, p) -> pmf:
    #-----------------------------------------------------------------#
    '''
    args:
        n: number of successes
        p: probability of success
    '''
    #------------------------------------------------------------#
    x = np.arange(0, 100)
    p = stats.nbinom.pmf(x, n, p)
    return pmf(x=x, probability=p)


def uniform(a, b)-> pmf:
    #------------------------------------------------------------#
    '''
    args:
        a: lower bound
        b: upper bound
    '''
    #-----------------------------------------------------------------#
    x = np.arange(a, b+1)
    p = np.ones(len(x)) / len(x)
    return pmf(x=x, probability=p)