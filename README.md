# Behavioral Cloning: 
##The taste of self-driving Car in a Simulator

### Overview

The objective: Use Deep Learning to Clone Driving Behavior

I personly tried drive the car in provided simulator with keyboard. It took me a while to get used to the track 1, and saved  some images for exporation. Udacity also provided prefessional driver records for better results. I have broken down the project scope into the following sections:

- Explorting the data
- Training methold
- Deep learning Model
- Driving fine tune
- Lessions Learned
- Future Work

First, let's talk about the computer setup. I have run the training cycle and driving cycle on CPU. It is quite hard on the machine. I have Dell T3500 workstation with 12G ram and 8 core CPU run on Linux Ubuntu 14.04. 

### Dependencies

This project requires **Python 3.5** and the following Python libraries installed:

- [Keras](https://keras.io/)
- [NumPy](http://www.numpy.org/)
- [SciPy](https://www.scipy.org/)
- [TensorFlow](http://tensorflow.org)
- [Pandas](http://pandas.pydata.org/)
- [OpenCV](http://opencv.org/)
- [Matplotlib](http://matplotlib.org/) (Optional)
- [Jupyter](http://jupyter.org/) (Optional)

Run this command at the terminal prompt to install [OpenCV](http://opencv.org/). Useful for image processing:

- `conda install -c https://conda.anaconda.org/menpo opencv3`

### How to Run the Model

This repository comes with trained model which you can directly test using the following command.

- `python drive.py model.json`

## Implementation

### Explorting the data

First of all, we may want to see what is captured in the data set. For every given time stamp, there are 3 images recorded from left, center, and right cameras. 
<p align="center">
 <img src="./image/3_images.png" width="800">
</p>
One steering angle value, throttle, brake, speed also recorded in the driving_log.csv file. 
First plot is my keyboard driving inputs.
<p align="center">
 <img src="./image/keyboard_driving_dataset_steering_ 200_1600.png" width="800">
</p>
Second plot is Udacity provided driving inputs.
<p align="center">
 <img src="./image/udacity_raw_training_steering_200_1600" width="800">
</p>
Next plot is Udacity Open Source Car Challenge Two real human driving inputs.
<p align="center">
 <img src="./image/udacity_ch2_final_evalution_steering_200_1600_30f_moving_avg.png" width="800">
</p>
Next plot is Nvidia paper real human driving inputs.
<p align="center">
 <img src="./image/nvidia_training_steering_200_1600_30f_moving_avg.png" width="800">
</p>

The first thing I want to do is to apply a moving average to the input and smooth out the steering action. The green line shows 30 frams averaged outputs. Because I never turn the steering wheel like keyboard driver does. But something happed after. 

### Training methold

There are total 24108 images (8036 images per camera), along with 8036 steering records in the Udacity dataset. Compare to Nvidia sample set 40,000 images one channel, this is very small, may not have enough image to generalize the model weight in order to pass the training course or other unseen courses. Image Augmentation Technique is recommand by Udacity and other student's post. I tried many of them, and I choose the following to form the image processing pipeline. 
- Random adjust the image brighnest/darkness
- Flip the image and steering angle together
- Random mask some area of the image, create shadow effect
- Random shear
- Lookahead, crop the image center portation out
- Horizontal crop the image center portation out, with shifted steering angle

### Data Processing Pipeline

I use Pandas module to read driving_log.csv, and saved all image file names and steering inputs into lists. I applied 5 frame moving average to the steering angle, and centered the result. The left image assigned the steering angle with right turn offset, the right image assigned with left turn offset. 
![udacity_raw_training_steering_200_2000_5f](https://cloud.githubusercontent.com/assets/22917810/22216774/a084e848-e15d-11e6-8c46-b53881ca2a17.png)

### Image Processing Pipeline
A combination of augmentation tools collected from https://github.com/upul, https://medium.com/@mohankarthik/cloning-a-car-to-mimic-human-driving-5c2f7e8d8aff#.kot5rcn4b, https://chatbotslife.com/learning-human-driving-behavior-using-nvidias-neural-network-model-and-image-augmentation-80399360efee#.gix474ksk, Udacity class room and slack channels. 
```
def pipeline(image, steering_angle, top_crop_percent=0.35, bottom_crop_percent=0.1,
                       resize_dim=(64, 64), do_shear_prob=0.9):
    head = bernoulli.rvs(do_shear_prob)
    if head == 1:
        image, steering_angle = random_shear(image, steering_angle)
    image = lookahead_crop(image, top_crop_percent, bottom_crop_percent)
    image = dark_image(image)
    image = mask_image(image)
    image, steering_angle = random_flip(image, steering_angle)
    image = resize(image, resize_dim)
    return image, steering_angle
```
The lookahead_crop is my idea inspired by human driver. When kids start learning how to ride bicycle, one important technique is when you zigzag the handle bar too much, lookup, lookahead, you will find much easier to balance and smooth out. Same thing as driving a car or race car, if you find hard to follow the track, or the lane, look farther, look into the apex, or the vanish point, you will find everything will smooth out. I applied this idea into the code, During training cycle, top_crop_percent is 35%, bottom_crop_percent is 10%. In the driving cycle, the image crop is moving up a few percent, to 30% and 12%. I can fine tune it without retrain the model again. 
<p align="center">
 <img src="./image/image_pipeline_output.png" width="800">
</p>

I see lot of students using random shear operation, so I think I will give it a try. I selected 90% of the images will go through the random shearing process. 

<p align="center">
 <img src="./images/sheared.png">
</p>
Random darkness function is to adjust the brightness of a image, to make new image look like under a shade. 
Random mask_img function will mask random area of a picture, the new image will look like under a shade and other difficulty visual conditions. The idea is inspired by Udacity project 1. 
```
def mask_image(img):
    """
    Applies an image mask.
    region_of_interest(img, vertices):
    """
    rows,cols,ch = img.shape
    ax = int(cols*(np.random.uniform(-0.5,0.5)))
    bx = int(ax+cols*np.random.uniform(-0.5,0.5))
    cx = int(np.random.uniform(0, 80))
    dx = int(cols-cx)
    p = (np.random.uniform(-0.5,0.5))
    vertices = np.array([[(dx,rows),(ax,int(p*rows)), (bx, int(p*rows)), (cx,rows)]], dtype=np.int32)
    shadow = np.random.randint(1, 200)
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
```
The flip function is simple flip the image and steering angle at 50/50 chance. I will consider flip all the images just to double the dataset size. 

According to the model input size(64x64), all images need resize to fit. 
There are more benfit to make a square input, it will explained in the notebook and driving section.

### Network Architecture 

NVIDIA's End to End Learning for Self-Driving Cars Network Architecture is a well know model that work for this kind of project. paper. In this project, I kept all the feature filters for each layer, and use the same stride and padding sittings as Nvidia paper. And transfer it into Keras environment. My approach is try to keep track of what kind of structures are going to work, which one are not going to work.  

Eventually, I added Lambda layer to normalize the input image, follow by a color space conversion layer.
I don't see a big differecy or benefit about this color space layer yet. 
I do added a MaxPooling layer after first Convolutional layer to see if there is any benefit.  
The full network is look like this: 
```
model = Sequential()
# First Normalize layer, credit to comma ai model
model.add(Lambda(lambda x: x / 127.5 - 1.0, input_shape=(66, 200, 3)))
# Color space conversion layer, credit to Vivek's model
model.add(Convolution2D(3, 1, 1, border_mode='same', name='color_conv'))
# Classic five convolutional, Nvidia model and additional maxpooling layers
model.add(Convolution2D(24, 5, 5, border_mode='valid', activation='relu', subsample=(2, 2)))
model.add(Convolution2D(36, 5, 5, border_mode='valid', activation='relu', subsample=(2, 2)))
model.add(Convolution2D(48, 5, 5, border_mode='valid', activation='relu', subsample=(2, 2)))
model.add(Convolution2D(64, 3, 3, border_mode='valid', activation='relu', subsample=(1, 1)))
model.add(Convolution2D(64, 3, 3, border_mode='valid', activation='relu', subsample=(1, 1)))
model.add(Flatten())
# Next, five fully connected layers
model.add(Dense(1164, activation='relu'))
model.add(Dropout(keep_prob))
model.add(Dense(100, activation='relu'))
model.add(Dense(50, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(1))

model.summary()

model.compile(optimizer=Adam(learning_rate), loss="mse" )
```
There are 1,595,523 total/trainable parameters came out of this model. Nvidia paper stated only 250k parameters. 

Resize the input image to 64x64 can save one third of the CPU training time, from 190s to 120s. 

### Training Method
As requested, fit_generator function is used to generate images while training the model. 
I implementated two generators, one for training batch, one for validation batch:
```
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
```


Batch size of both `train_gen` and `validation_gen` was 64. We used 20032 images per training epoch. It is to be noted that these images are generated on the fly using the document processing pipeline described above. In addition to that, we used 6400 images (also generated on the fly) for validation. We used `Adam` optimizer with `1e-4` learning rate. Finally, when it comes to the number of training epochs we tried several possibilities such as `5`, `8`, `1`0, `2`5 and `50`. However, `8` works well on both training and validation tracks. 

## Results
In the initial stage of the project, I used a dataset generated by myself. That dataset was small and recorded while navigating the car using the laptop keyboard. However, the model built using that dataset was not good enough to autonomously navigate the car in the simulator. However, later I used the dataset published by the Udacity. The model developed using that dataset (with the help of augmented data) works well on both tracks as shown in following videos. 

#### Training Track
[![training_track](https://img.youtube.com/vi/nSKA_SbiXYI/0.jpg)](https://www.youtube.com/watch?v=nSKA_SbiXYI)

#### Validation Track
[![validation_track](https://img.youtube.com/vi/ufoyhOf5RFw/0.jpg)](https://www.youtube.com/watch?v=ufoyhOf5RFw)

## Conclusions and Future Directions
In this project, we were working on a regression problem in the context of self-driving cars. In the initial phase, we mainly focused on finding a suitable network architecture and trained a model using our own dataset. According to Mean Square Error (**MSE**) our model worked well. However, it didn't perform as expected when we test the model using the simulator. So it was a clear indication that MSE is not a good metrics to assess the performance this project. 

In the next phase of the project, we started to use a new dataset (actually, it was the dataset published by Udacity). Additionally, we didn't fully rely on MSE when building our final model. Also, we use relatively small number of training epochs (namely `8` epochs). Data augmentation and new dataset work surprisingly well and our final model showed superb performance on both tracks. 

When it comes to extensions and future directions, I would like to highlight followings.

* Train a model in real road conditions. For this, we might need to find a new simulator.
* Experiment with other possible data augmentation techniques.
* When we are driving a car, our actions such as changing steering angles and applying brakes are not just based on instantaneous driving decisions. In fact, curent driving decision is based on what was traffic/road condition in fast few seconds. Hence, it would be really interesting to seee how Recurrent Neural Network (**RNN**) model such as **LSTM** and **GRU** perform this problem.
* Finally, training a (deep) reinforcement agent would also be an interesting additional project.
