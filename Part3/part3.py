import numpy as np
from sklearn.neural_network import MLPRegressor
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def my_function(input):
    res = [ 2*i[0]+i[1] for i in input]
    return res


# create random input and output data for the 2D function
x_train = np.random.randint(100, size=(200,2))
y_train = my_function(x_train)

x_test = np.random.randint(200, size=(300,2))
y_test = my_function(x_test)


hidden_layer = (100)
num_of_iteraion = 1000
train_network= MLPRegressor(hidden_layer_sizes=hidden_layer, max_iter=num_of_iteraion).fit(x_train, y_train)

y_tree = train_network.predict(x_test)

def colculate_mistakes(our_answer,correct_Answer):
    l = len(our_answer)
    sum = 0
    for i in range(0,l):
        sum = sum+ abs(our_answer[i]-correct_Answer[i])
    return sum/l

mistake = colculate_mistakes(y_test,y_tree)
print('colculated mistake: ' + str(mistake))
