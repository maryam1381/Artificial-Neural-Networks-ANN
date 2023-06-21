import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPRegressor

def colculate_mistakes(our_answer,correct_Answer):
    l = len(our_answer)
    sum = 0
    for i in range(0,l):
        sum = sum+ abs(our_answer[i]-correct_Answer[i])
    return sum/l

x = [2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29]
y = [12,13,13.5,13.5,14,14,14,13,12.5,13,14,15,14,13,11,10,10.5,10,14,11,9.5,8.5,8,8,7,7,7.5,8]
mydata = [(2,12),(3,13),(4,13.5),(5,13.5),(6,14),(7,14),(8,14),(9,13),(10,12.5),(11,13),
          (12,14),(13,15),(14,14),(15,13),(16,11),(17,10),(18,10.5),(19,10),(20,14),
          (21,11),(22,9.5),(23,8.5),(24,8),(25,8),(26,7),(27,7),(28,7.5),(29,8)]


hidden_layer=(10,100,10)
num_of_iteration=1000
train = []
test = []

for i in range(len(mydata)):
    if i % 4 == 0:
        test.append(mydata[i])
    else:
        train.append(mydata[i])


# convert it to 2d array
train_x = np.array([data[0] for data in train]).reshape(-1,1)
train_y = np.array([data[1] for data in train]).reshape(-1,1)
test_x = np.array([data[0] for data in test]).reshape(-1,1)
test_y = np.array([data[1] for data in test]).reshape(-1,1)

train_network = MLPRegressor(hidden_layer_sizes=hidden_layer,
                                max_iter=num_of_iteration,
                                random_state=0,
                                shuffle=True).fit(train_x,train_y.ravel())

tree_y = train_network.predict(test_x)

fig, ax = plt.subplots()
test_plt,  = plt.plot(test_x, tree_y,color = 'g', label='Test')
expected_plt,  = plt.plot(test_x, test_y,color='b', label='Expected_result')
train_plt, = plt.plot(train_x, train_y,color ='r', label='Train', linestyle=':',linewidth=2)

mistake = colculate_mistakes(test_y,tree_y)
plt.title('colculated mistake: ' + str(round(mistake[0],4)))
ax.set_xlabel('\n Train Info : number of points : '+str(len(train_x))+"    Train : "+ str(train)+
                '\n Test  Info : number of points : '+str(len(test_x))+"    Test : "+ str(test)+ 
                '\n Number of Hidden layer : '+str(hidden_layer)+
                '\n Number of Iteration : '+str(num_of_iteration)) 
plt.tight_layout()
fig.set_figwidth(19)
plt.show()
