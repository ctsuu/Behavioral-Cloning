#import pandas as pd
import numpy as np
#from subprocess import call
from numpy import arange 
#import matplotlib
#import matplotlib.pyplot as plt
#import matplotlib.image as mpimg
#import matplotlib.collections as collections
#import matplotlib.gridspec as gridspec
import os
import csv
import scipy.misc
#from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import cv2
#from keras.layers import Dense,Input,Activation,Dropout,Conv2D,Convolution2D,MaxPooling2D,Flatten,Lambda,ELU
#from keras.optimizers import Adam
#from keras.models import model_from_json
#from keras.models import Sequential, load_model
#from keras.callbacks import EarlyStopping
#import json
#import h5py


#matplotlib.style.use('ggplot')


### Define some help functions: 
def flip_image(img):
    img_flip = cv2.flip(img,1)
    #angle_flip = -angle
    return img_flip

def dark_image(img):
    img = cv2.cvtColor(img,cv2.COLOR_RGB2HSV)
    img[:,:,2] = img[:,:,2]*np.random.uniform(0.1,1.2)
    dst = cv2.cvtColor(img,cv2.COLOR_HSV2RGB)
    return dst

def trans_image(img,steer,trans_range):
    # Translation
    tr_x = np.random.uniform()-0.5
    steer_ang = steer + tr_x/trans_range
    tr_y = np.random.uniform()-0.5
    rows,cols,ch = img.shape
    Trans_M = np.float32([[1,0,tr_x*140],[0,1,tr_y*50]])
    image_tr = cv2.warpAffine(img,Trans_M,(cols,rows))
    return image_tr,steer_ang

def mask_image(img):
    """
    Applies an image mask.
    region_of_interest(img, vertices):
    Only keeps the region of the image defined by the polygon
    formed from `vertices`. The rest of the image is set to black.
    """
    #defining a blank mask to start with
    #mask = np.zeros_like(img)
    #image1 = cv2.cvtColor(img,cv2.COLOR_RGB2HSV)
    #image = mpimg.imread('data/'+X_all[index])
    rows,cols,ch = img.shape
    ax = int(cols*(np.random.uniform(-0.5,0.5)))
    #+y_all[np.random.uniform(100,8000)])
    bx = int(ax+cols*np.random.uniform(-0.5,0.5))
    #cx = int(cols*np.random.uniform(-1,1)+cols*np.random.uniform(-0.5,0.5))
    #dx = int(cx+cols*np.random.uniform())
    #ax = 100+y_all[np.random.uniform(100,8000)]
    #bx = 200
    cx = int(np.random.uniform(0, 160))
    dx = int(cols-cx)
    p = (np.random.uniform(-0.5,0.5))
    #vertices = np.array([[(p*cols,rows),(ax,int(p*rows)), (bx, int(p*rows)), (cols*(1+p),rows)]], dtype=np.int32)
    vertices = np.array([[(dx,rows),(ax,int(p*rows)), (bx, int(p*rows)), (cx,rows)]], dtype=np.int32)
       
    shadow = np.random.randint(1, 128)
    mask = np.full_like(img, shadow)
        
    #defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255
        
    #filling pixels inside the polygon defined by "vertices" with the fill color    
    cv2.fillPoly(mask, vertices, ignore_mask_color)
    
    #returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img, mask)
    
    return masked_image

def crop_image(img, angle):
    center_row = 80
    center_col = 160
    width = 200
    height = 66
    
    x_var = int(np.random.uniform(-20,20))
    y_var = int(np.random.uniform(-10,10))
    
    crop_img = img[int(center_row-height/2+y_var):int(center_row+height/2+y_var),
                  int(center_col-width/2+x_var):int(center_col+width/2+x_var)]
    
    angle_factor = 0.002 # degree per each shifted pixel
    adj_angle = angle + angle_factor*y_var
    
    return crop_img, adj_angle
    
def plot_random_image(n_row,n_col,data_dir,X,y):

    plt.figure(figsize = (2.5*n_col,1.2*n_row))
    gs1 = gridspec.GridSpec(n_row,n_col)
    gs1.update(wspace=0.1, hspace=0.2) # set the spacing between axes. 

    for i in range(n_row*n_col):
        ax1 = plt.subplot(gs1[i])
        index = np.random.randint(1,len(y))
        img = mpimg.imread(data_dir+X[index])
        dark_img = dark_image(img)
        masked_img = mask_image(dark_img)
        cropped_img,adj_angle = crop_image(masked_img,y[index])
        if i%2 ==1:
            plt.imshow(flip_image(cropped_img))
            plt.title(str(np.round(adj_angle,5)),fontsize=8)
            plt.axis('off')
        if i%2 ==0:
            plt.imshow(cropped_img)
            plt.title(str(np.round(adj_angle,5)),fontsize=8)
            plt.axis('off')
                
    plt.show()

def get_image_files_loop(batch_size, num_of_img):
    """
    The simulator records three images (namely: left, center, and right) at a given time
    However, when we are picking images for training we randomly (with equal probability)
    one of these three images and its steering angle.
    :param batch_size:
        Size of the image batch
    :return:
        An list of selected (image files names, respective steering angles)
    """
    #data = pd.read_csv(DRIVING_LOG_FILE)
    #num_of_img = len(data)
    rnd_indices = np.random.randint(0, num_of_img, batch_size)

    img_batch = []
    for index in rnd_indices:
        img_batch.append((X_all[index],y_all[index]))

    return img_batch


"""
def load_batch_images(batch_size=128):
    global train_batch_pointer
    x_out = []
    y_out = []
    for i in range(0, batch_size):
        img =mpimg.imread(data_dir+train_xs[(train_batch_pointer + i) % num_train_images])
        angle = train_ys[(train_batch_pointer + i) % num_train_images]
        dark_img = dark_image(img)
        masked_img = mask_image(dark_img)
        cropped_img,adj_angle = crop_image(masked_img,angle)
        if i%2 ==1:
            x_out.append(flip_image(cropped_img)/255.0)
            #x_out.append(flip_image(masked_img)/255.0)
            y_out.append([-adj_angle])
        if i%2 ==0:
            x_out.append((cropped_img)/255.0)
            #x_out.append(scipy.misc.imresize(cropped_img, [66, 200]) / 255.0)
            y_out.append([adj_angle])
    train_batch_pointer += batch_size
    return x_out, y_out

"""

def LoadValBatchGenerator(batch_size):
    global val_batch_pointer
    x_out = []
    y_out = []
    for i in range(0, batch_size):
        img = scipy.misc.imread(img_path+val_xs[(val_batch_pointer + i) % num_val_images])
        angle = val_ys[(val_batch_pointer + i) % num_val_images]
        dark_img = dark_image(img)
        masked_img = mask_image(dark_img)
        cropped_img,adj_angle = crop_image(masked_img,angle)
        if i%2 ==1:
            x_out.append(flip_image(cropped_img)/255.0)
            y_out.append([-adj_angle])
        if i%2 ==0:
            x_out.append((cropped_img)/255.0)
            #x_out.append(scipy.misc.imresize(cropped_img, [66, 200]) / 255.0)
            y_out.append([adj_angle])
    val_batch_pointer += batch_size
    return np.array(x_out).astype('float32'), np.array(y_out).astype('float32')

def save_model(model, model_name='model.json', weights_name='model.h5'):
    """
    Save the model into the hard disk
    :param model:
        Keras model to be saved
    :param model_name:
        The name of the model file
    :param weights_name:
        The name of the weight file
    :return:
        None
    """
    os.remove(model_name)
    os.remove(weights_name)

    json_string = model.to_json()
    with open(model_name, 'w') as outfile:
        json.dump(json_string, outfile)

    model.save_weights(weights_name)


data_dir = './data/'
img_path = './data/IMG/'
log_path = './data/driving_log.csv'
data_csv = 'driving_log.csv'
model_name = 'model.json'
weights_name = 'model.h5'

center = []
left = []
right = []
steering = []
throttle = []
brake = []
speed = []

#read drivinglog.csv
with open(log_path) as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        center.append(row['center'].split('IMG')[1])
        left.append(row['left'].split('IMG')[1])
        right.append(row['right'].split('IMG')[1])
        steering.append(float(row['steering']))
        throttle.append(float(row['throttle']))
        brake.append(float(row['brake']))
        speed.append(float(row['speed']))

#Apply 60 frame moving average to steering records 
smooth = []
smooth_l = []
smooth_r = []
moving_range = 60
moving_avg = arange(-1,1,2/moving_range) 
print ('Apply '+str(len(moving_avg))+' frame moving average to steering records')

for i in range(len(steering)):
    j = i%moving_range
    moving_avg[j] = steering[i]
    smooth.append(np.mean(moving_avg)*2)
    smooth_l.append(np.mean(moving_avg)*2+0.1)
    smooth_r.append(np.mean(moving_avg)*2-0.1)

print('Merge Left, Center and Right Camera image together...')

X_all = np.concatenate((center, left, right), axis = 0)
y_all = np.concatenate((smooth, smooth_l, smooth_r), axis = 0)

print('Merge completed. Number of total samples', len(y_all))

"""
### 
f = np.arange(0.0, len(y_all), 1)

fig, ax = plt.subplots(figsize =(12,3))
ax.set_title('steering input by section')
ax.plot(f, y_all, color='black')
ax.axhline(0, color='black', lw=1)
plt.show()
"""

#X_all, y_all = shuffle(X_all, y_all)


"""
#reading in an image
index = np.random.randint(1,len(X_all))
img = mpimg.imread(img_path+X_all[index])
plt.show(img) 
"""
batch_size = 256

#points to the end of the last batch
train_batch_pointer = 0
val_batch_pointer = 0


train_xs = X_all[:int(len(X_all) * 0.8)]
train_ys = y_all[:int(len(y_all) * 0.8)]

val_xs = X_all[-int(len(X_all) * 0.2):]
val_ys = y_all[-int(len(y_all) * 0.2):]

num_train_images = len(train_xs)
num_val_images = len(val_xs)

print ('Number of raw images in Training Set: '+str(num_train_images)+' images')
print ('Number of raw images in Validation Set: '+str(num_val_images)+' images')

X_val, y_val = LoadValBatchGenerator(256)

print (get_image_files_loop(batch_size,num_train_images))

