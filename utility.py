import errno
import json
import os
import csv
import random
import cv2
import matplotlib.pyplot as plt
import numpy as np
from numpy import arange
from sklearn.utils import shuffle
import pandas as pd
import scipy.misc
from scipy.ndimage import rotate
from scipy.stats import bernoulli

# Some useful constants

log_path = './data/driving_log.csv'
img_path = './data/IMG'
rolling_window = 3
amp_factor = 1.5
steering_offset = 0.1

model_json = 'model.json'
model_weights = 'model.h5'

###Load Driving Log data into system
data = pd.read_csv(log_path)

left = []
center = []
right = []
steering = data['steering']
steering = steering.astype(np.float32)

for i in range(len(data)):
    left.append(data['left'][i].split('IMG')[1])
    center.append(data['center'][i].split('IMG')[1])
    right.append(data['right'][i].split('IMG')[1])

###Apply 30 frame moving average to steering inputs 
df = pd.DataFrame(steering)
smooth = df.rolling(rolling_window,center=True).mean()

###Also apply the steering offset to left or right camera images:
smooth_l = df.rolling(rolling_window,center=True).mean()+steering_offset
smooth_r = df.rolling(rolling_window,center=True).mean()-steering_offset

###Merge Left, Center and Right Camera image together...
X_all = np.concatenate((center[200:8000], left[200:8000], right[200:8000]), axis = 0)
y_all = np.concatenate((smooth[200:8000], smooth_l[200:8000], smooth_r[200:8000]), axis = 0)

###Shuffle all samples
X_all, y_all = shuffle(X_all, y_all)

### Split training set and validation set
split_point = 128*150   # 128*150=19200

X_train = X_all[:int(split_point)]
y_train = y_all[:int(split_point)]

X_val = X_all[-int(len(X_all)-split_point):]
y_val = y_all[-int(len(X_all)-split_point):]

num_train_images = len(X_train)
num_val_images = len(X_val)

### Define some helper functions: 

def lookahead_crop(image, top_percent=0.35, bottom_percent=0.1):
    
    #Inspired by human driver. 
    #Look farther can smooth out the steering wheel angle. 
    #Crops an image based on driver lookahead requirment, aim for the apex point or vannish point.  
       
    rows,cols,ch = image.shape
    image = image[int(rows*top_percent):int(rows-rows*bottom_percent),:]

    return image

def horizontal_crop(image, angle):

    #random pan the camera from left to right in a small pixel shift, between -24 to 24 pixels, compensate the steering angle 
    
    rows,cols,ch = image.shape
    width = int(cols*0.8)
    
    x_var = int(np.random.uniform(-24,24))
        
    crop_img = image[0:rows,int(cols/2-width/2+x_var):int(cols/2+width/2+x_var)]
    
    angle_factor = 0.002 # degree per each shifted pixel
    adj_angle = angle + angle_factor*x_var
    
    return crop_img, adj_angle


def resize(image, new_dim):
    
    return scipy.misc.imresize(image, new_dim)


def random_flip(image, steering_angle, flipping_prob=0.5):
    # Source: https://github.com/upul/behavioral_cloning    

    head = bernoulli.rvs(flipping_prob)
    if head:
        return np.fliplr(image), -1 * steering_angle
    else:
        return image, steering_angle


def flip_image(image):
    # Source: https://chatbotslife.com/learning-human-driving-behavior-using-nvidias-neural-network-model-and-image-augmentation-80399360efee#.gix474ksk

    img_flip = cv2.flip(image,1)
    #angle_flip = -angle
    return img_flip


def dark_image(image):
    # Source: https://chatbotslife.com/learning-human-driving-behavior-using-nvidias-neural-network-model-and-image-augmentation-80399360efee#.gix474ksk

    img = cv2.cvtColor(image,cv2.COLOR_RGB2HSV)
    img[:,:,2] = img[:,:,2]*np.random.uniform(0.1,1.2)
    dst = cv2.cvtColor(img,cv2.COLOR_HSV2RGB)
    return dst


def mask_image(image):
    # refer to Udacity Self Driving Car Project 1: 
    # Applies an image mask.
    # region_of_interest(image, vertices):
    
    rows,cols,ch = image.shape
    ax = int(cols*(np.random.uniform(-0.5,0.5)))
    bx = int(ax+cols*np.random.uniform(-0.5,0.5))
    cx = int(np.random.uniform(0, 80))
    dx = int(cols-cx)
    p = (np.random.uniform(-0.5,0.5))
    #vertices = np.array([[(p*cols,rows),(ax,int(p*rows)), (bx, int(p*rows)), (cols*(1+p),rows)]], dtype=np.int32)
    vertices = np.array([[(dx,rows),(ax,int(p*rows)), (bx, int(p*rows)), (cx,rows)]], dtype=np.int32)
       
    shadow = np.random.randint(1, 200)
    mask = np.full_like(image, shadow)
        
    #defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(image.shape) > 2:
        channel_count = image.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255
        
    #filling pixels inside the polygon defined by "vertices" with the fill color    
    cv2.fillPoly(mask, vertices, ignore_mask_color)
    
    #returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(image, mask)
    
    return masked_image


def random_gamma(image):
    
    # Source: http://www.pyimagesearch.com/2015/10/05/opencv-gamma-correction/

    gamma = np.random.uniform(0.4, 1.5)
    inv_gamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** inv_gamma) * 255
                      for i in np.arange(0, 256)]).astype("uint8")

    # apply gamma correction using the lookup table
    return cv2.LUT(image, table)


def random_shear(image, steering_angle, shear_range=200):

    # Source: https://medium.com/@ksakmann/behavioral-cloning-make-a-car-drive-like-yourself-dc6021152713#.7k8vfppvk

    rows, cols, ch = image.shape
    dx = np.random.randint(-shear_range, shear_range + 1)
    random_point = [cols / 2 + dx, rows / 2]
    pts1 = np.float32([[0, rows], [cols, rows], [cols / 2, rows / 2]])
    pts2 = np.float32([[0, rows], [cols, rows], random_point])
    dsteering = dx / (rows / 2) * 360 / (2 * np.pi * 25.0) / 6.0
    M = cv2.getAffineTransform(pts1, pts2)
    image = cv2.warpAffine(image, M, (cols, rows), borderMode=1)
    steering_angle += dsteering

    return image, steering_angle


def pipeline(image, steering_angle, top_crop_percent=0.35, bottom_crop_percent=0.1,
                       resize_dim=(66, 200), do_shear_prob=0.5):
    
    # Current image augement pipeline

    head = bernoulli.rvs(do_shear_prob)

    if head == 1:
        image, steering_angle = random_shear(image, steering_angle)


    image = dark_image(image)

    image = mask_image(image)

    image = lookahead_crop(image, top_crop_percent, bottom_crop_percent)

    image, steering_angle = horizontal_crop(image, steering_angle)

    image, steering_angle = random_flip(image, steering_angle)

    image = resize(image, resize_dim)

    return image, steering_angle



### Loop through the dataset, generate batch image files names and steering angles.

tp =0 # set training batch point at zero at begining

def get_train_batch_data(batch_size=64):
    global tp
    X_batch = X_train[tp:tp+batch_size]
    y_batch = y_train[tp:tp+batch_size]
    if tp + batch_size >= len(X_train):
        tp = 0
    else:
        tp += batch_size
    return X_batch, y_batch


vp = 0  #set validation batch pointer at zero at begining

def get_val_batch_data(batch_size=64):
    global vp
    X_batch = X_val[vp:vp+batch_size]
    y_batch = y_val[vp:vp+batch_size]
    if vp + batch_size >= len(X_val):
        vp = 0
    else:
        vp += batch_size
    return X_batch, y_batch



def generate_train_batch(batch_size=64):

    # Generate train batch data on the fly, yield array file format.

    while True:
        X_batch = []
        y_batch = []
        #images = get_train_image_files(batch_size)
        files = get_train_batch_data(batch_size)
        raw_angle = files[1]
        i = 0       
        for img_file in files[0]:
            #Read all images and angles in the batch and go through image process pipeline
            raw_image = scipy.misc.imread(img_path + img_file)
            new_image, new_angle = pipeline(raw_image, amp_factor*raw_angle[i])
            X_batch.append(new_image)
            y_batch.append(new_angle)
            i += 1

        yield np.array(X_batch), np.array(y_batch)

def generate_val_batch(batch_size=64):

    # Generate validation batch date on the fly, yield array file format.

    while True:
        X_batch = []
        y_batch = []
        #images = get_val_image_files(batch_size)
        files = get_val_batch_data(batch_size)
        raw_angle = files[1]
        i = 0
        for img_file in files[0]:
            #Read all images and angles in the batch and go through image process pipeline
            raw_image = scipy.misc.imread(img_path + img_file)
            #new_image, new_angle = pipeline(raw_image, amp_factor*raw_angle[i])
            X_batch.append(resize(raw_image,(66, 200)))
            y_batch.append(amp_factor*raw_angle[i])

            #X_batch.append(new_image)
            #y_batch.append(new_angle)
            i += 1

        yield np.array(X_batch), np.array(y_batch)

def save_model(model, model_name='model.json', weights_name='model.h5'):

    # Save the model into the hard disk
    # https://github.com/upul/behavioral_cloning

    silent_delete(model_name)
    silent_delete(weights_name)

    json_string = model.to_json()
    with open(model_name, 'w') as outfile:
        json.dump(json_string, outfile)

    model.save_weights(weights_name)


def silent_delete(file):
    
    # This method delete the given file from the file system if it is exist.
    # Source: http://stackoverflow.com/questions/10840533/most-pythonic-way-to-delete-a-file-which-may-not-exist

    try:
        os.remove(file)

    except OSError as error:
        if error.errno != errno.ENOENT:
            raise
