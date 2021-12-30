import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets,linear_model
from sklearn.metrics import mean_squared_error

diabetes=datasets.load_diabetes()

# print(diabetes.keys()) # We print it to know which keys is present in dataset
#dict_keys(['data', 'target', 'frame', 'DESCR', 'feature_names', 'data_filename', 'target_filename', 'data_module'])
#these keys are printed, now we use data key

diabetes_x=diabetes.data[:,np.newaxis,2]
#for more feature replace "diabetes_x=diabetes.data[:,np.newaxis,2]" this with "diabetes_x=diabetes.data" this and plot lines that are started with plt
#print(diabetes_x)
diabetes_x_train=diabetes_x[:-30]
diabetes_x_test=diabetes_x[-30:]

diabetes_y_train=diabetes.target[:-30]
diabetes_y_test=diabetes.target[-30:]

model=linear_model.LinearRegression()
model.fit(diabetes_x_train,diabetes_y_train)
diabetes_y_predicted=model.predict(diabetes_x_test)

print("Mean squared error is:",mean_squared_error(diabetes_y_test,diabetes_y_predicted))
print("Weights:",model.coef_)
print("Intercept:",model.intercept_)

plt.scatter(diabetes_x_test,diabetes_y_test)
plt.plot(diabetes_x_test,diabetes_y_predicted)

plt.show()

