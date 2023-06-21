# Import Image from wand.image module
import os
import cv2
import numpy as np
from os import listdir
from os.path import isfile, join
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import cross_validate
from wand.image import Image
	
train_path ="D:\\term6\AI\Project2\Part6\images\\train"
train_files = [f for f in listdir(train_path) if isfile(join(train_path, f))]

for trainfile in train_files:
    with Image(filename ="D:\\term6\AI\Project2\Part6\images\\train\\"+str(trainfile)) as img:
        img.noise("poisson", attenuate = 0.9)
        img.save(filename ="D:\\term6\AI\Project2\Part6\images\\train_noise1\\"+str(trainfile))


test_path ="D:\\term6\AI\Project2\Part6\images\\test"
test_files = [f for f in listdir(test_path) if isfile(join(test_path, f))]

for testfile in test_files:
    with Image(filename ="D:\\term6\AI\Project2\Part6\images\\test\\"+str(testfile)) as img:
        img.noise("poisson", attenuate = 0.9)
        img.save(filename ="D:\\term6\AI\Project2\Part6\images\\test_noise\\"+str(testfile))


