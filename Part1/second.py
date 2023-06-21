import math
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import zero_one_score

# This is our chosen function
def SimpleFunction(input_list):
    return [math.sin(i*math.pi/100) for i in input_list]

# give us correct input and out put in specified range
def correct_input_output(num_of_points,min,max):
    my_X = np.linspace(min, max, num=num_of_points).reshape(-1, 1)
    my_Y = np.array(SimpleFunction(my_X)).reshape(-1, 1)
    return my_X,my_Y

# x,y = Training(10,-10,20)
# print(x)
# print(y)

def Train_Network(hidden_layer,num_of_iteration,num_of_points,min,max):
    x_input,y_input = correct_input_output(num_of_points=num_of_points,min=min,max=max)
    network = MLPRegressor(hidden_layer_sizes=hidden_layer,
                           max_iter=num_of_iteration,
                           activation="logistic",
                           random_state=0,shuffle=True).fit(x_input,y_input.ravel())
    return network


def colculate_mistakes(our_answer,correct_Answer):
    l = len(our_answer)
    sum = 0
    for i in range(0,l):
        sum = sum+ abs(our_answer[i]-correct_Answer[i])
    return sum/l

def Train_Test_Network(hidden_layer,num_of_iteration,train_info,test_info):
    x_train,y_train = correct_input_output(train_info[0],train_info[1],train_info[2])
    x_test ,y_test = correct_input_output(test_info[0],test_info[1],test_info[2])
    train_network  = Train_Network(hidden_layer=hidden_layer,
                                   num_of_iteration=num_of_iteration,
                                   num_of_points=train_info[0],
                                   min=train_info[1],max=train_info[2])
    
    y_tree_output = train_network.predict(x_test)

    fig, ax = plt.subplots()
    test_plt,  = plt.plot(x_test, y_tree_output,color = 'g', label='Test')
    expected_plt,  = plt.plot(x_test, y_test,color='b', label='Expected_result')
    train_plt, = plt.plot(x_train, y_train,color ='r', label='Train', linestyle=':',linewidth=2)
    mistake = colculate_mistakes(y_test,y_tree_output)
    plt.title('colculated mistake: ' + str(round(mistake[0],4)))
    ax.set_xlabel('\n Train Info : number of points : '+str(train_info[0])+"    Train Range : ("+ str(train_info[1])+ ","+str(train_info[2])+")"+
                  '\n Test  Info : number of points : '+str(test_info[0])+"    Test  Range : ("+ str(test_info[1])+ ","+str(test_info[2])+")"+
                  '\n Number of Hidden layer : '+str(hidden_layer[0])+
                  '\n Number of Iteration : '+str(num_of_iteration))

    # Adding legend, which helps us recognize the curve according to it's color
    # plt.legend()
    plt.tight_layout()
    plt.show()

# For number of hidden layer
# Train_Test_Network((10,),3000,(200,-100,100),(600,-300,300))
# Train_Test_Network((20,),2000,(200,-100,100),(600,-300,300))
# Train_Test_Network((30,),2000,(200,-100,100),(600,-300,300))
# Train_Test_Network((1000,),2000,(200,-100,100),(600,-300,300))
# Train_Test_Network((9,),2000,(200,-100,100),(600,-300,300))


# # for train range
# Train_Test_Network((1000,),2000,(200,-150,150),(600,-300,300))
# Train_Test_Network((1000,),2000,(200,-100,100),(600,-300,300))
# Train_Test_Network((1000,),2000,(200,-200,200),(600,-300,300))

# # for test range
# Train_Test_Network((1000,),2000,(200,-200,200),(600,-300,300))
# Train_Test_Network((1000,),2000,(200,-200,200),(600,-400,400))
# Train_Test_Network((1000,),2000,(200,-200,200),(600,-500,500))

# # for train point
# Train_Test_Network((1000,),2000,(150,-100,100),(600,-300,300))
# Train_Test_Network((1000,),2000,(200,-100,100),(600,-300,300))
# Train_Test_Network((1000,),2000,(250,-100,100),(600,-300,300))

# # for test point
# Train_Test_Network((1000,),2000,(200,-200,200),(500,-400,400))
# Train_Test_Network((1000,),2000,(200,-200,200),(600,-400,400))
# Train_Test_Network((1000,),2000,(200,-200,200),(700,-400,400))

# # number of iteration
# Train_Test_Network((1000,),1500,(200,-100,100),(600,-300,300))
# Train_Test_Network((1000,),2000,(200,-100,100),(600,-300,300))
# Train_Test_Network((1000,),3000,(200,-100,100),(600,-300,300))















