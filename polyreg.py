import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from random import random, seed 
from sklearn.pipeline import make_pipeline
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, mean_squared_log_error

def PolynomialRegression(degree=2, **kwargs):
    return make_pipeline(PolynomialFeatures(degree),
                         LinearRegression(**kwargs))

x=np.random.rand(100,1)

y=5*x*x+0.1*np.random.randn(100,1) 
                            
X_test=np.linspace(0,1,100)[:, None]

model = PolynomialRegression(2)

model.fit(x,y)

y_test = model.predict(X_test)

r2 = r2_score(model.predict(x),y)

mse = mean_squared_error(model.predict(x), y)

print ("Ridge Mean Square Error: ",mse)
print ("Ridge R2-score: ",r2 )                                                                                                                
print('Mean absolute error: %.2f' % mean_absolute_error(model.predict(x), y))
print ("------------------------------------")
plt.plot(X_test.ravel(), y_test, color='C1',label='y_model')
plt.plot(x.ravel(), y, 'ro', label='y', color='C7')
plt.xlabel(r'$x$') #Setter navn på x-akse
plt.ylabel(r'$y$') 
plt.title(r'Linear Regression - 2.order polynomial')

plt.legend()
plt.show()
