import pandas as pd
import numpy as np
import random
from numpy import arange 
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.collections as collections
import matplotlib.gridspec as gridspec
import os
import csv
import scipy.misc
from scipy.stats import bernoulli
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import cv2
from subprocess import call

# %matplotlib inline
"""
# Some useful constants

log_path = './data/driving_log.csv'
img_path = './data/IMG'

rolling_window = 3
amp_factor = 1.5
steering_offset = 0.1

###Load Driving Log data into system
data = pd.read_csv(log_path)

left = []
center = []
right = []
#steering = []
steering = data['steering']
steering = steering.astype(np.float32)
for i in range(len(data)):
    left.append(data['left'][i].split('IMG')[1])
    center.append(data['center'][i].split('IMG')[1])
    right.append(data['right'][i].split('IMG')[1])

"""
log_path = './Calvenn_data/sim_log.csv'
img_path = './Calvenn_data/dataset2/'
center = []
steering = []
#read drivinglog.csv
with open(log_path) as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        center.append(row['center'])
        #left.append(row['left'][1:])
        #right.append(row['right'][1:])
        steering.append(float(row['steering']))

play_image = center[200:-200]
play_angle = steering[200:-200]
img = cv2.imread('136385649.jpg',0)

rows,cols = img.shape

smoothed_angle = 0


i = 0
while(cv2.waitKey(10) != ord('q')):
    full_image = scipy.misc.imread(img_path + center[i])
    #image = cv2.imread(img_path + play_image[i])
    
    #large_image = scipy.misc.imresize(image[-150:], [1024, 1280])
    cv2.imshow("Sim replay", full_image)
    i += 1

cv2.destroyAllWindows()
