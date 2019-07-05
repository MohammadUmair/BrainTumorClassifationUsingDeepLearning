
# coding: utf-8

# In[14]:

import keras 
import numpy as np

get_ipython().magic('matplotlib inline')
#Magic Command - Details :https://stackoverflow.com/questions/19410042/how-to-make-ipython-notebook-matplotlib-plot-inline

import matplotlib.pyplot as plt
plt.rcParams['image.cmap'] = 'gray' #To show all images in the grayscale



# In[15]:

#I think, I should use this one : 
#https://keras.io/getting-started/faq/#how-can-i-obtain-reproducible-results-using-keras-during-development
#instead of below code

np.random.seed(100) #Important to generate the same training result again


# # For reading the medical images, I have used the nibabel package.
# ## For nibabel installation : http://nipy.org/nibabel/installation.html

# In[16]:

import os
import nibabel
from nibabel.testing import data_path

cwd = os.getcwd()
print(cwd)


train_img = os.path.join(cwd, 'lupus001_T1_reg+bias.nii.gz')
train_img = nibabel.load(train_img)

ground_truth_img = os.path.join(cwd, 'lupus001_T1_reg+bias+brain.nii.gz')
ground_truth_img =  nibabel.load(ground_truth_img)


# In[17]:

#Check the size of images
print('Train img : ', train_img.shape);
print('Test img :', ground_truth_img.shape);
    


# # Conversion from Nifti to Numpy
# The images are still still in nifti format
# Therefore, it is required to convert them to numpy array.

# In[18]:

print(type(train_img))
print(type(ground_truth_img))

train_img = train_img.get_data()
ground_truth_img = ground_truth_img.get_data()

print(type(train_img))
print(type(ground_truth_img))


# In[19]:

print(np.max(train_img))
print(np.max(ground_truth_img))


#  Display the middle Slice and its mask 

# In[20]:

middle_slice = round(train_img.shape[2]/2)
print('Show the output of Slice # ', middle_slice)


slice = train_img[:,:,middle_slice]
plt.figure()
plt.imshow(slice)

slice_mask = ground_truth_img[:,:,middle_slice]
plt.figure()
plt.imshow(slice_mask)





# # Construction of Training images and its Labels (Here GroundTruth images)

# In[21]:

Num_of_train_examples = 10

train_images = train_img[:,:,middle_slice-Num_of_train_examples:middle_slice+Num_of_train_examples]
ground_truth_images = ground_truth_img[:,:,middle_slice-Num_of_train_examples:middle_slice+Num_of_train_examples]

print(train_images.shape)
print(ground_truth_images.shape)


# # Preparing the image shapes for the input
# 

# In[22]:


print('Old shape:')
print(train_images.shape)
print(ground_truth_images.shape)

#The number of examples should comes first (Num of examples, , ) therefore, their transpose is required. 
train_images = train_images.transpose(2,0,1)
ground_truth_images = ground_truth_images.transpose(2,0,1)


print('New shape:')
print(train_images.shape)
print(ground_truth_images.shape)




# In[ ]:

# Used these in different experiements. No need of them. 
'''
new_train_images = np.zeros((10, 256, 256,3))
new_train_images[:,:,:,0] = train_images
new_train_images[:,:,:,1] = train_images
new_train_images[:,:,:,2] = train_images
print(new_train_images.shape)

new_ground_truth_images = np.zeros((10, 256, 256,3))
new_ground_truth_images[:,:,:,0] = ground_truth_images
new_ground_truth_images[:,:,:,1] = ground_truth_images
new_ground_truth_images[:,:,:,2] = ground_truth_images
print(new_ground_truth_images.shape)
'''


# # Creating the binary Masks from ground turth images
# 

# In[23]:

ground_truth_images[ground_truth_images>0] = 1 
ground_truth_images[ground_truth_images<=0] = 0

print(ground_truth_images.shape) #Confirming their size
plt.imshow(ground_truth_images[0,:,:].reshape(256,256)) #Showing one sample of binary mask
#from scipy.misc import imresize


# # Here is the VGG trained model
# ## For now, it is not required.

# In[ ]:


'''
from keras.applications.vgg16 import VGG16
from keras.layers import Input



#It will download it first. The last parameter shows that we don't want its last layers (FC layers)
# so that we can add our layers and train on it.
#
# Network Image: https://blog.keras.io/img/imgclf/vgg16_original.png
#Pretraining example : https://blog.keras.io/building-powerful-image-classification-models-using-very-little-data.html

trained_model = VGG16(input_tensor=Input(shape=(256, 256, 1))  , weights='imagenet', include_top=False)

'''



# In[8]:

#trained_model.layers[0] #Just did to see what it shows 


# # Constructing our CNN Model here
# 
#  In each layer, we wil use the output of previous layer. Except, in first layer, we use the shape of our input image (256, 256,1). If we are using pretrained model, then its output will be the input of first layer
# 
# I tried to make a network like this : V-Net Network: http://mattmacy.io/vnet.pytorch/images/diagram.png (Paper: https://arxiv.org/abs/1606.04797) which itself inspired from U-Net: https://arxiv.org/abs/1505.04597
# 
# They used such structures so that images don't lose their information during Conv layers.  You will find the add() function below for the same purpose. If you read that github code of SelfDriving Car again, you will notice concatenate there for the same purpose: https://github.com/markstrefford/didi-sdc-challenge-2017/blob/master/nn/nn.py#L149
# 
# 

# In[29]:


from keras.layers import Dense, Flatten, Activation, Input
from keras.layers.convolutional import Conv2D, Conv2DTranspose,UpSampling2D
from keras.layers.normalization import BatchNormalization
from keras.layers.merge import add


#Initializing the model 
from keras.models import Sequential 
model = Sequential() #We are creating an empty sequential model here


#The data should be in the form of (Num of examples, height, width, color channel) 
# For constructing the model, these both commands are not required, so they can be moved up/down,
#they are required during the training which comes later. 
train_images = train_images.reshape(-1,256,256,1) 
ground_truth_images = ground_truth_images.reshape(-1,256,256,1)

print(train_images.shape)

##########################
#We are required to tell the first layer that our input image would of the following shape
#Incase of pre-trained model, we will use its output : trained_model.output
#################################

Inputs = Input(shape=(256,256,1))


########### Layer 1 ############
Conv1 = Conv2D( filters=8, kernel_size=[3,3], strides=[1,1], padding='same', kernel_initializer='he_normal',activation='sigmoid')(Inputs)


########### Layer 2 ############
Conv2 = Conv2D( filters=8, kernel_size=[2,2], strides=[2,2], padding='same', kernel_initializer='he_normal',activation='sigmoid') (Conv1)

########### Layer 3 ############
Conv3 = Conv2D( filters=16, kernel_size=[3,3], strides=[1,1], padding='same', kernel_initializer='he_normal',activation='sigmoid') (Conv2)

########### Layer 4 ############
Conv4 = Conv2D( filters=16, kernel_size=[2,2], strides=[2,2], padding='same', kernel_initializer='he_normal',activation='sigmoid') (Conv3)

########### Layer 5 ############
Conv5 = Conv2D( filters=32, kernel_size=[3,3], strides=[1,1], padding='same', kernel_initializer='he_normal',activation='sigmoid') (Conv4)

########### Layer 6 ############
Conv6 = Conv2D( filters=32, kernel_size=[3,3], strides=[1,1], padding='same', kernel_initializer='he_normal',activation='sigmoid') (Conv5)

########### Layer 7 ############
Conv7 =  add([Conv2DTranspose( filters=16, kernel_size=[2,2], strides=[2,2], padding='same', kernel_initializer='he_normal',activation='sigmoid') (Conv6), Conv3])

########### Layer 8 ############
Conv8 = Conv2D( filters=8, kernel_size=[3,3], strides=[1,1], padding='same', kernel_initializer='he_normal',activation='sigmoid') (Conv7)

########### Layer 9 ############
Conv9 = add( [Conv2DTranspose( filters=8, kernel_size=[2,2], strides=[2,2], padding='same', kernel_initializer='he_normal',activation='sigmoid') (Conv8), Conv1])

########### Layer 10 ############
Conv10 = Conv2D( filters=1, kernel_size=[1,1], strides=[1,1], padding='same', kernel_initializer='he_normal',activation='sigmoid') (Conv9)






# ### Here we are instantiating our model
# Here inputs are the input of our first layer and output is the last layer
# For pre_trained models, input will be the input of trained model and output will remain same. 

# In[43]:


from keras.models import Model

model = Model(inputs=Inputs, output=Conv10) 

#It shows the summary of our model.

#Here None means it is not fixed and can be any. 

#The last columns shows how many parameters are involved and will be learned during training. 
#For example 3x3 filter has 9 params. 

#For experiemnt, you can load the pre_trained model and then pass its input and output
#to the Model function and check its summary


print(model.summary())


# # Loss functions

# In[44]:

import keras.backend as K
smooth=1

#Initially Iou taken from : https://github.com/markstrefford/didi-sdc-challenge-2017/blob/master/nn/nn.py#L56
#Then I changed it for improved version

def IOU_calc(y_true, y_pred):
    V = K.flatten(y_true)
    X = K.flatten(y_pred)
    # inter=tf.reduce_sum(tf.mul(logits,trn_labels))
    intersection = K.sum(X * V)
    # union=tf.reduce_sum(tf.sub(usum=tf.add(logits,trn_labels),intersection=tf.mul(logits,trn_labels)))
    union = K.sum( (X + V) - (X * V) )
#    union = K.sum(usum - intersection)
    #IoU = (2. * intersection + smooth) / (union + smooth)
    IoU = intersection / union
    return IoU


def IOU_calc_loss(y_true, y_pred):
    return 1-IOU_calc(y_true, y_pred)


# In[45]:

#copied from : https://github.com/markstrefford/didi-sdc-challenge-2017/blob/master/nn/nn.py#L56
#who has copied from: # from https://github.com/jocicmarko/ultrasound-nerve-segmentation/blob/master/train.py
def dice_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


def dice_coef_loss(y_true, y_pred):
    return -dice_coef(y_true, y_pred)


# # Training the data

# In[46]:

#First, we select the optimization algorithms such as sgd, adam, rmsprop and so on.
#For more optimizers: https://keras.io/optimizers/

#Then choose the loss function and Metrics. We can create our own. 
#Fore loss functions : https://keras.io/losses/
#For Metrics(Optional): https://keras.io/metrics/ 

#Once we are done with it, we compile with them. 
#The network will not train if not compiled


from keras import optimizers

#The lowe the learning rate, the slower it will converge.
adam = optimizers.adam(lr=7e-3)  #Higher run rate may now converge at all. Right now, it is at higher side. 


model.compile(optimizer=adam,
              loss=IOU_calc_loss,
              metrics=[IOU_calc])


# ## We can also start from where we left yesterday
# 
# Now, I will load the my trained model weights. 
# 
# Make sure, you comment this code so that you could try the above constructed model from scratch i.e. random weights. 

# In[47]:

model.load_weights('20_slices_trained')


# In[48]:

#For parameters details : https://keras.io/models/model/#fit
#For further training, run fit function repeatedly

model.fit(train_images,ground_truth_images, batch_size=20, epochs=10,shuffle=True)





# In[283]:

#######################################################################################
#################      We can save our weights for later use       ####################
#################       or further training in future              ####################
######################################################################################
#model.save_weights('20_slices_trained')


# # Testing our Model

# In[286]:

#Normally we input our test set in it so that could predict the results
# As you know, here we want to predict train images, therefore, passing train images. 
#You may experiment with different input

masks =model.predict(train_images)

#It shape shows that we have the brain mask of all 20 slices. 


# In[276]:

masks.shape


# In[277]:

#Input Slice
plt.imshow(train_images[1,:,:,:].reshape(256,256))


# In[278]:

#Predicted/Generated Brain Mask
   
plt.imshow(masks[1,:,:,0].reshape(256,256))


# In[279]:

#Predicted Brain (Mask) Slice
   
plt.imshow(np.multiply(masks[1],train_images[1]) .reshape(256,256))


# In[266]:

# The actual ground_truth Mask 

plt.imshow(ground_truth_images[0].reshape(256,256))


# # Displaying all the images at once
# 
# I should have done at the top of this notebook too when I took input from file

# In[268]:

plt.figure(figsize=(20,10))
columns = 10
for i, image in enumerate(train_images):
    plt.subplot(len(train_images) / columns + 1, columns, i + 1)
    plt.imshow(image.reshape(256,256))


# In[267]:

plt.figure(figsize=(20,10))
columns = 10
for i, image in enumerate(ground_truth_images):
    plt.subplot(len(ground_truth_images) / columns + 1, columns, i + 1)
    plt.imshow(image.reshape(256,256))


# In[287]:

plt.figure(figsize=(20,10))
columns = 10
for i, image in enumerate(masks):
    plt.subplot(len(masks) / columns + 1, columns, i + 1)
    plt.imshow(image.reshape(256,256))


# In[288]:

plt.figure(figsize=(20,10))
columns = 10
for i, image in enumerate(np.multiply(masks, train_images)):
    plt.subplot(len(masks) / columns + 1, columns, i + 1)
    plt.imshow(image.reshape(256,256))


# In[272]:

plt.figure(figsize=(20,10))
columns = 10
for i, image in enumerate(np.multiply(ground_truth_images, train_images)):
    plt.subplot(len(masks) / columns + 1, columns, i + 1)
    plt.imshow(image.reshape(256,256))


# #  

# ## Just confirming whether slices are different because all the brain masks look similar. 

# In[255]:

plt.figure(figsize=(20,10))
columns = 10
for i, image in enumerate(train_img[100:150,:,:]):
    plt.subplot(len(train_img[100:150,:,:]) / columns + 1, columns, i + 1)
    plt.imshow(image.reshape(256,256))


# In[256 ]:
history = model.fit(train_images,ground_truth_images,validation_split=0.33, epochs=150, batch_size=20, verbose=0)
# list all data in history
print(history.history.keys())
# summarize history for accuracy
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()



# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[300]:

import keras
print(keras.__version__)


# In[ ]:



