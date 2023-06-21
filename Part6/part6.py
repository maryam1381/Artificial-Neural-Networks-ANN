
import os
import cv2
import numpy as np
from os import listdir
from os.path import isfile, join
from sklearn.neural_network import MLPClassifier,MLPRegressor
from sklearn.model_selection import cross_validate

train_path ="D:\\term6\AI\Project2\Part6\images\\train"
train_noise_path = "D:\\term6\AI\Project2\Part6\images\\train_noise"
train_files = [f for f in listdir(train_path) if isfile(join(train_path, f))]
# print(train_files)
train_x = []
train_y = []
for trainfile in train_files:
    img = cv2.imread(train_path+"\\"+str(trainfile),0)
    img_np = np.array(img)
    img_np = img_np.reshape(256)
    # img_np2 = np.reshape(img, [256, 1]) #256 D 

    noise_img = cv2.imread(train_noise_path+"\\"+str(trainfile),0)
    noise_img_np = np.array(img)
    noise_img_np = img_np.reshape(256)


    train_y.append(img_np)
    train_x.append(noise_img_np)
    # print(img_np) 

train_x = np.array(train_x)
train_y = np.array(train_y)


test_path ="D:\\term6\AI\Project2\Part6\images\\test"
test_noise_path = "D:\\term6\AI\Project2\Part6\images\\test_noise"
test_files = [f for f in listdir(test_path) if isfile(join(test_path, f))]
# print(test_files)
test_x = []
test_y = []
for testfile in test_files:
    img = cv2.imread(test_path+"\\"+str(testfile),0)
    img_np = np.array(img)
    img_np = img_np.reshape(256)
    # img_np2 = np.reshape(img, [256, 1]) #256 D 

    noise_img = cv2.imread(test_noise_path+"\\"+str(testfile),0)
    noise_img_np = np.array(img)
    noise_img_np = img_np.reshape(256)


    test_y.append(img_np)
    test_x.append(noise_img_np)
    # print(img_np) 

test_x = np.array(test_x)
test_y = np.array(test_y)

hidden_layer = (100,100,100,20)
# hidden_layer = (100,1000,100)
num_of_iteration = 2000

train_netwoek = MLPRegressor( hidden_layer_sizes= hidden_layer,
                                max_iter=num_of_iteration,
                                random_state=1,
                                shuffle=True).fit(train_x, train_y)

tree_images = train_netwoek.predict(test_x)

accuracy = train_netwoek.score(tree_images, test_y)
print(" accuracy = ", accuracy )
for i in range(len(tree_images)) :
    image = tree_images[i]
    my_img = np.reshape(image, [16, 16])
    cv2.imwrite("D:\\term6\AI\Project2\Part6\images\\network_predict"+ "\\" +str(i)+'_predict.png', my_img)

