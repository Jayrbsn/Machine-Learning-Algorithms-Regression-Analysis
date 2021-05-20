
#Resource: https://pythonprogramming.net/how-to-program-best-fit-line-slope-machine-learning-tutorial/
#Resource: https://www.chegg.com/homework-help/questions-and-answers/create-python-file-called-linearregressionpy-task-use-diabetes-dataset-mentioned-perform-l-q53202693
#Resource: https://stackoverflow.com/questions/43118846/writing-a-best-fit-line-function-in-python
import matplotlib.pyplot as plt
import numpy as np 
from sklearn.datasets import load_diabetes

#Create function to replace the linear_model.LinearRegression() function
#Function takes in data for the x and y axis
#The formula for m comes from one of the resources found at the top of this document
#The means for the two arrays are found using the .mean() function from the numpy library
def best_fit(d_X, y):
    mean_of_x = np.mean(d_X)
    mean_of_y = np.mean(y)
    m = np.dot(np.transpose(d_X - mean_of_x), y - mean_of_y) / np.dot(np.transpose(d_X-mean_of_x), d_X-mean_of_x)
    b = mean_of_y - m * mean_of_x
    
    return [m,b]

#Copy code from PDF example to load the data
#Make small modification so that only the last 20 observations are used for testing and the rest for training
d = load_diabetes()
d_X = d.data[:, np.newaxis, 2]
y = d.target
dx_train = d_X[::]
dy_train = d.target[::]
dx_test = d_X[-20:]
dy_test = d.target[-20:]


#Call best_fit function to calculate m and b
#Then use regression formula to make prediction
m, b = best_fit(dx_train, dy_train)
prediction_line = m * d_X + b

#Create the scatter graph and legend
plt.scatter(dx_train, dy_train, c='r', label='training')
plt.scatter(dx_test, dy_test, c='g', label='testing')
plt.plot(d_X, prediction_line, c='b')
plt.legend()
plt.show()