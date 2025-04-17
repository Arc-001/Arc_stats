import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
from utils import descriptive_stats as ds

class linear_regression:
    def __init__(self, **kwargs):
        self.x = np.array(kwargs['x']).reshape(-1, 1)
        self.y = np.array(kwargs['y']).reshape(-1, 1)
        self.model = LinearRegression()
        self.model.fit(self.x, self.y)
        self.intercept = self.model.intercept_[0]
        self.slope = self.model.coef_[0][0]
        self.mse = mean_squared_error(self.y, self.model.predict(self.x))
        self.r2 = r2_score(self.y, self.model.predict(self.x))

    def fit(self):
        self.model.fit(self.x, self.y)
        self.intercept = self.model.intercept_[0]
        self.slope = self.model.coef_[0][0]
        self.mse = mean_squared_error(self.y, self.model.predict(self.x))
        self.r2 = r2_score(self.y, self.model.predict(self.x))
        return self.intercept, self.slope, self.mse, self.r2

    def predict_arr(self, x):
        x = np.array(x).reshape(-1, 1)
        return self.model.predict(x)
    
    def predict(self, x):
        x = np.array(x).reshape(-1, 1)
        return self.model.predict(x)[0][0]
    

    def plot(self):
        plt.scatter(self.x, self.y, color='red',marker= "x", label='Data Points')
        plt.plot(self.x, self.model.predict(self.x), color='red', label='Regression Line')
        plt.xlabel('X')
        plt.ylabel('Y')
        # plt.xticks(self.x)
        # plt.yticks(self.y)
        plt.grid()
        plt.text(0.5, 0.9, f'Intercept: {self.intercept:.2f}, Slope: {self.slope:.2f}, MSE: {self.mse:.2f}, R^2: {self.r2:.2f}', ha='center', va='center', transform=plt.gca().transAxes)
        plt.xlim(self.x.min(), self.x.max())
        plt.ylim(self.y.min(), self.y.max())
        plt.title('Linear Regression')
        plt.legend()
        plt.show()
        return plt

