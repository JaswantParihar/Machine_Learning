import matplotlib.pyplot as plt
import numpy as np
from sklearn import linear_model
from sklearn.metrics import mean_squared_error

x=np.array([[1],[2],[3]])
x_train=x
x_test=x

y_train=np.array([3,2,4])
y_test=np.array([3,2,4])

model=linear_model.LinearRegression()
model.fit(x_train,y_train)
y_predicted=model.predict(x_test)

print("Mean squared error is:", mean_squared_error(y_test,y_predicted))
print('Weights:',model.coef_)
print('Intercept:',model.intercept_)

plt.scatter(x_test,y_test)
plt.plot(x_test,y_predicted)

plt.show()
