import random
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPRegressor
import math


def myFuction(input_list):
    return [math.sin(i*math.pi)+i**2+round(random.uniform(0,1),2) for i in input_list]

def colculate_mistakes(our_answer,correct_Answer):
    l = len(our_answer)
    sum = 0
    for i in range(0,l):
        sum = sum+ abs(our_answer[i]-correct_Answer[i])
    return sum/l

def train_test_network(train_range,test_range,train_point,test_point,hidden_layer,num_of_iteration):
    train_x = np.linspace(train_range[0], train_range[1], num=train_point).reshape(-1, 1)
    train_y = np.array(myFuction(train_x)).reshape(-1, 1)

    train_network = MLPRegressor(hidden_layer_sizes=hidden_layer,
                                 max_iter=num_of_iteration,
                                 random_state=0,
                                 shuffle=True).fit(train_x,train_y.ravel())
    
    # hidden_layer_sizes : array-like of shape(n_layers - 2,), default=(100,)
    #     The ith element represents the number of neurons in the ith
    #     hidden layer.
    test_x = np.linspace(test_range[0], test_range[1], num=test_point).reshape(-1, 1)
    test_y = np.array(myFuction(test_x)).reshape(-1, 1)

    tree_y = train_network.predict(test_x)

    # colculate mistake


    fig, ax = plt.subplots()
    test_plt,  = plt.plot(test_x, tree_y,color = 'g', label='Test')
    expected_plt,  = plt.plot(test_x, test_y,color='b', label='Expected_result')
    train_plt, = plt.plot(train_x, train_y,color ='r', label='Train', linestyle=':',linewidth=2)
    
    mistake = colculate_mistakes(test_y,tree_y)
    plt.title('colculated mistake: ' + str(round(mistake[0],4)))
    ax.set_xlabel('\n Train Info : number of points : '+str(train_point)+"    Train Range : ("+ str(train_range[0])+ ","+str(train_range[1])+")"+
                  '\n Test  Info : number of points : '+str(test_point)+"    Test  Range : ("+ str(test_range[0])+ ","+str(test_range[1])+")"+
                  '\n Number of Hidden layer : '+str(hidden_layer[0])+
                  '\n Number of Iteration : '+str(num_of_iteration)) 
    plt.tight_layout()
    plt.show()

train_test_network((-100,100),(-300,300),100,500,(10,),1000)