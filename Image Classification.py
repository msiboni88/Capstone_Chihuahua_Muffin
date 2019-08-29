#!/usr/bin/env python
# coding: utf-8

# In[22]:


# Import libraries and modules
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

from skimage import io

import pandas as pd

# For reproducibility
np.random.seed(42)

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.utils import np_utils
from keras.datasets import mnist
from keras.callbacks import ReduceLROnPlateau, Callback

import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

import seaborn as sns


# In[2]:


from keras.applications import VGG16

vgg_conv = VGG16(weights='imagenet',
                  include_top=False,
                  input_shape=(150, 150, 3))


# In[4]:


# Loading dataset
chihuahua_file = np.load('./files/c_array_150_3.npy')
muffin_file = np.load('./files/m_array_150_3.npy')

# Loading test set
test_array = np.load('./files/test_array_150_3.npy')

# combining images to one dataset
files = np.append(chihuahua_file, muffin_file, axis = 3)

# Creating file of output values, 
# chihuahua is positive class muffin is negative class
y = np.append(np.ones(chihuahua_file.shape[3]),
              np.zeros(muffin_file.shape[3]))

# Reshaping X such that it can be input in to neural net 
X = np.transpose(files, (3, 0, 1, 2))
test_array = np.transpose(test_array, (3, 0, 1, 2))

X = X.astype('float32')
test_array = test_array.astype('float32')


# Setting up callback for early stopping and reducing learning rates
class myCallback(Callback):
    def on_epoch_end(self, epoch, logs={}):
        if(logs.get('acc')>.90):
            print("\nReached 90% accuracy so cancelling training!")
            self.model.stop_training = True            
callbacks=myCallback()

reduce_lr = ReduceLROnPlateau(monitor='loss', factor=0.3, verbose=1,
                              patience=2, min_lr=0.00000001)


# In[ ]:


# Train Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.07, stratify=y)



# CNN - no transfer learning, no image data gen 
# Starting at 8 because lower performing similar models have been removed from notebook
model8 = Sequential()

model8.add(Conv2D(
    filters = 40,
    kernel_size = (18),
    activation = 'relu',
    input_shape = (150, 150, 3)
))
model8.add(AveragePooling2D(pool_size = (3)))
model8.add(Dropout(0.25))

model8.add(Conv2D(20,
                     kernel_size=6,
                     activation='relu'))
model8.add(AveragePooling2D(pool_size=3))
model8.add(Dropout(0.25))

model8.add(Flatten())

model8.add(Dense(256, activation='relu'))
model8.add(Dropout(0.3))
model8.add(Dense(64, activation='relu'))
model8.add(Dropout(0.3))
model8.add(Dense(32, activation='relu'))
model8.add(Dropout(0.3))
model8.add(Dense(1, activation='sigmoid'))

model8.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
history8 = model8.fit(X_train, y_train,
                 batch_size=256,
                 epochs=20,
                 verbose=1,
                 validation_data=(X_test, y_test),
                 callbacks=[reduce_lr, callbacks]
                 )


# In[ ]:


# Save model to file
model8.save('./files/model_aws28.HDF5')

# Run predictions for meme images
print(model8.predict(test_array))

# loss scores
train_loss8 = history8.history['loss']
test_loss8 = history8.history['val_loss']

# Generate line plot of training, testing loss over epochs.
plt.figure(figsize=(12, 8))

plt.plot(train_loss8, label='Training Loss', color='#185fad')
plt.plot(test_loss8, label='Testing Loss', color='orange')

plt.title('Training and Testing Loss by Epoch', fontsize = 25)
plt.xlabel('Epoch', fontsize = 18)
plt.ylabel('Categorical Crossentropy', fontsize = 18)

plt.legend(fontsize = 18)
# Save plot to file
plt.savefig('./files/loss28.png')
plt.show();

# model accuracy scores
train_acc8 = history8.history['acc']
test_acc8 = history8.history['val_acc']

# Generate line plot of training, testing loss over epochs.
plt.figure(figsize=(12, 8))

plt.plot(train_acc8, label='Training Accuracy', color='#185fad')
plt.plot(test_acc8, label='Validation Accuracy', color='orange')

plt.title('Training and Validation Accuracy by Epoch', fontsize = 25)
plt.xlabel('Epoch', fontsize = 18)
plt.ylabel('Categorical Crossentropy', fontsize = 18)

plt.legend(fontsize = 18)
# Save plot to file
plt.savefig('./files/acc28.png')
plt.show();

#saving y values to file
np.save('./files/y.npy', y)

np.save('./files/test_loss28.npy', test_loss8)
np.save('./files/train_loss28.npy', train_loss8)
np.save('./files/test_acc28.npy', test_acc8)
np.save('./files/train_acc28.npy', train_acc8)

# Run predictions for full data set
probs = model8.predict(X)
preds = (probs > 0.5).astype(int)

np.save('./files/probs28.npy', probs)
np.save('./files/preds28.npy', preds)

# plot the confusion matrix
confusion_mtx = confusion_matrix(y, preds) 
f,ax = plt.subplots(figsize=(8, 8))
sns.heatmap(confusion_mtx, annot=True, linewidths=0.01,cmap="copper")
plt.xlabel("Prediction", fontsize=20)
plt.xticks([0,1], ['Blueberry Muffin', 'Chihuahua'], fontsize=16)
plt.ylabel("Actual", fontsize=20)
plt.xticks([0,1], ['Blueberry Muffin', 'Chihuahua'], fontsize=16)
plt.title("Confusion Matrix", fontsize=24)
plt.savefig('./files/cm28.png')


# In[ ]:


# transfer learning CNN
vgg_conv = VGG16(weights='imagenet',
                  include_top=False,
                  input_shape=(150, 150, 3))

for layer in vgg_conv.layers[:-2]:
    layer.trainable = False

reduce_lr = ReduceLROnPlateau(monitor='loss', factor=0.3, verbose=1,
                              patience=2, min_lr=0.00000001)

model8 = Sequential()

model8.add(vgg_conv)

model8.add(Flatten())

model8.add(Dense(256, activation='relu'))
model8.add(Dropout(0.3))
model8.add(Dense(64, activation='relu'))
model8.add(Dropout(0.3))
model8.add(Dense(1, activation='sigmoid'))


model8.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
history8 = model8.fit(X_train, y_train,
                 batch_size=128,
                 epochs=15,
                 verbose=1,
                 validation_data=(X_test, y_test),
                 callbacks=[reduce_lr, callbacks]
                 )


# In[ ]:


# Save model to file
model8.save('./files/model_aws29.HDF5')

# Run predictions for meme images
print(model8.predict(test_array))

# loss scores
train_loss8 = history8.history['loss']
test_loss8 = history8.history['val_loss']

# Generate line plot of training, testing loss over epochs.
plt.figure(figsize=(12, 8))

plt.plot(train_loss8, label='Training Loss', color='#185fad')
plt.plot(test_loss8, label='Testing Loss', color='orange')

plt.title('Training and Testing Loss by Epoch', fontsize = 25)
plt.xlabel('Epoch', fontsize = 18)
plt.ylabel('Categorical Crossentropy', fontsize = 18)

plt.legend(fontsize = 18)
# Save plot to file
plt.savefig('./files/loss29.png')
plt.show();

# model accuracy scores
train_acc8 = history8.history['acc']
test_acc8 = history8.history['val_acc']

# Generate line plot of training, testing loss over epochs.
plt.figure(figsize=(12, 8))

plt.plot(train_acc8, label='Training Accuracy', color='#185fad')
plt.plot(test_acc8, label='Validation Accuracy', color='orange')

plt.title('Training and Validation Accuracy by Epoch', fontsize = 25)
plt.xlabel('Epoch', fontsize = 18)
plt.ylabel('Categorical Crossentropy', fontsize = 18)

plt.legend(fontsize = 18)
# Save plot to file
plt.savefig('./files/acc29.png')
plt.show();

np.save('./files/test_loss29.npy', test_loss8)
np.save('./files/train_loss29.npy', train_loss8)
np.save('./files/test_acc29.npy', test_acc8)
np.save('./files/train_acc29.npy', train_acc8)

# Run predictions for full data set
probs = model8.predict(X)
preds = (probs > 0.5).astype(int)

np.save('./files/probs29.npy', probs)
np.save('./files/preds29.npy', preds)

# plot the confusion matrix
confusion_mtx = confusion_matrix(y, preds) 
f,ax = plt.subplots(figsize=(8, 8))
sns.heatmap(confusion_mtx, annot=True, linewidths=0.01,cmap="copper")
plt.xlabel("Prediction", fontsize=20)
plt.xticks([0,1], ['Blueberry Muffin', 'Chihuahua'], fontsize=16)
plt.ylabel("Actual", fontsize=20)
plt.xticks([0,1], ['Blueberry Muffin', 'Chihuahua'], fontsize=16)
plt.title("Confusion Matrix", fontsize=24)
plt.savefig('./files/cm29.png')


# In[ ]:


# getting IDs for 13 missclassified images
df = pd.DataFrame(probs)
df['preds']=preds
df['y']=y
df[df.preds != df.y]


# In[ ]:


# Showing missclassified images
io.imshow(X[408,:,:,:])
io.show()
io.imshow(X[753,:,:,:])
io.show()
io.imshow(X[909,:,:,:])
io.show()
io.imshow(X[929,:,:,:])
io.show()
io.imshow(X[1669,:,:,:])
io.show()
io.imshow(X[1888,:,:,:])
io.show()
io.imshow(X[1949,:,:,:])
io.show()
io.imshow(X[2232,:,:,:])
io.show()
io.imshow(X[2650,:,:,:])
io.show()
io.imshow(X[3047,:,:,:])
io.show()
io.imshow(X[3347,:,:,:])
io.show()
io.imshow(X[3358,:,:,:])
io.show()
io.imshow(X[3535,:,:,:])
io.show()


# In[ ]:


# ImageDataGenerator CNN

X = X.reshape(-1,150,150,3)
test_array = test_array.reshape(-1,150,150,3)

datagen = ImageDataGenerator(
    rotation_range=35,
    fill_mode='nearest',
    brightness=
    width_shift_range=0.15,
    height_shift_range=0.15,
    horizontal_flip=True,
    validation_split=0.08
)  

datagen.fit(X)

train_generator = datagen.flow(X_train, y_train, batch_size=256, subset="training")

validation_generator = datagen.flow(X_train, y_train, batch_size=256, subset="validation")



model8 = Sequential()

model8.add(Conv2D(
    filters = 40,
    kernel_size = (20),
    activation = 'relu',
    input_shape = (150, 150, 3)
))
model8.add(AveragePooling2D(pool_size = (3)))
model8.add(Dropout(0.25))

model8.add(Conv2D(20,
                     kernel_size=6,
                     activation='relu'))
model8.add(AveragePooling2D(pool_size=3))
model8.add(Dropout(0.25))

model8.add(Flatten())

model8.add(Dense(256, activation='relu'))
model8.add(Dropout(0.3))
model8.add(Dense(64, activation='relu'))
model8.add(Dropout(0.3))
model8.add(Dense(32, activation='relu'))
model8.add(Dropout(0.3))
model8.add(Dense(1, activation='sigmoid'))

model8.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
history8 = model8.fit_generator(generator=train_generator,
                 callbacks=[reduce_lr, callbacks],
                 epochs=15,
                 verbose=1,
                 validation_data=validation_generator,
                 steps_per_epoch=25,
                 workers=20, 
                 validation_steps = 5
                 )


# In[ ]:


# Save model to file
model8.save('./files/model_aws30.HDF5')

# Run predictions for meme images
print(model8.predict(test_array))

# loss scores
train_loss8 = history8.history['loss']
test_loss8 = history8.history['val_loss']

# Generate line plot of training, testing loss over epochs.
plt.figure(figsize=(12, 8))

plt.plot(train_loss8, label='Training Loss', color='#185fad')
plt.plot(test_loss8, label='Testing Loss', color='orange')

plt.title('Training and Testing Loss by Epoch', fontsize = 25)
plt.xlabel('Epoch', fontsize = 18)
plt.ylabel('Categorical Crossentropy', fontsize = 18)

plt.legend(fontsize = 18)
# Save plot to file
plt.savefig('./files/loss30.png')
plt.show();

# model accuracy scores
train_acc8 = history8.history['acc']
test_acc8 = history8.history['val_acc']

# Generate line plot of training, testing loss over epochs.
plt.figure(figsize=(12, 8))

plt.plot(train_acc8, label='Training Accuracy', color='#185fad')
plt.plot(test_acc8, label='Validation Accuracy', color='orange')

plt.title('Training and Validation Accuracy by Epoch', fontsize = 25)
plt.xlabel('Epoch', fontsize = 18)
plt.ylabel('Categorical Crossentropy', fontsize = 18)

plt.legend(fontsize = 18)
# Save plot to file
plt.savefig('./files/acc30.png')
plt.show();

np.save('./files/test_loss30.npy', test_loss8)
np.save('./files/train_loss30.npy', train_loss8)
np.save('./files/test_acc30.npy', test_acc8)
np.save('./files/train_acc30.npy', train_acc8)

# Run predictions for full data set
probs = model8.predict(X)
preds = (probs > 0.5).astype(int)

np.save('./files/probs30.npy', probs)
np.save('./files/preds30.npy', preds)

# plot the confusion matrix
confusion_mtx = confusion_matrix(y, preds) 
f,ax = plt.subplots(figsize=(8, 8))
sns.heatmap(confusion_mtx, annot=True, linewidths=0.01,cmap="copper")
plt.xlabel("Prediction", fontsize=20)
plt.xticks([0,1], ['Blueberry Muffin', 'Chihuahua'], fontsize=16)
plt.ylabel("Actual", fontsize=20)
plt.xticks([0,1], ['Blueberry Muffin', 'Chihuahua'], fontsize=16)
plt.title("Confusion Matrix", fontsize=24)
plt.savefig('./files/cm30.png')


# In[ ]:


# Image Data Generator and Transfer learning

X = X.reshape(-1,150,150,3)
test_array = test_array.reshape(-1,150,150,3)

datagen = ImageDataGenerator(
    rotation_range=15,
    fill_mode='nearest',
    width_shift_range=0.1,
    shear_range=0.01,
    height_shift_range=0.1,
    horizontal_flip=True,
    validation_split=0.1
)  

datagen.fit(X)

train_generator = datagen.flow(X_train, y_train, batch_size=256, subset="training")

validation_generator = datagen.flow(X_train, y_train, batch_size=256, subset="validation")


vgg_conv = VGG16(weights='imagenet',
                  include_top=False,
                  input_shape=(150, 150, 3))

for layer in vgg_conv.layers[:-2]:
    layer.trainable = False

model8 = Sequential()

model8.add(vgg_conv)

model8.add(Flatten())

model8.add(Dense(256, activation='relu'))
model8.add(Dropout(0.3))
model8.add(Dense(64, activation='relu'))
model8.add(Dropout(0.3))
model8.add(Dense(32, activation='relu'))
model8.add(Dropout(0.3))
model8.add(Dense(1, activation='sigmoid'))

model8.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
history8 = model8.fit_generator(generator=train_generator,
                 callbacks=[reduce_lr, callbacks],
                 epochs=20,
                 verbose=1,
                 validation_data=validation_generator,
                 steps_per_epoch=50,
                 workers=20, 
                 validation_steps = 20
                 )


# In[ ]:


# Save model to file
model8.save('./files/model_aws31.HDF5')

# Run predictions for meme images
print(model8.predict(test_array))

# loss scores
train_loss8 = history8.history['loss']
test_loss8 = history8.history['val_loss']

# Generate line plot of training, testing loss over epochs.
plt.figure(figsize=(12, 8))

plt.plot(train_loss8, label='Training Loss', color='#185fad')
plt.plot(test_loss8, label='Testing Loss', color='orange')

plt.title('Training and Testing Loss by Epoch', fontsize = 25)
plt.xlabel('Epoch', fontsize = 18)
plt.ylabel('Categorical Crossentropy', fontsize = 18)

plt.legend(fontsize = 18)
# Save plot to file
plt.savefig('./files/loss31.png')
plt.show();

# model accuracy scores
train_acc8 = history8.history['acc']
test_acc8 = history8.history['val_acc']

# Generate line plot of training, testing loss over epochs.
plt.figure(figsize=(12, 8))

plt.plot(train_acc8, label='Training Accuracy', color='#185fad')
plt.plot(test_acc8, label='Validation Accuracy', color='orange')

plt.title('Training and Validation Accuracy by Epoch', fontsize = 25)
plt.xlabel('Epoch', fontsize = 18)
plt.ylabel('Categorical Crossentropy', fontsize = 18)

plt.legend(fontsize = 18)
# Save plot to file
plt.savefig('./files/acc31.png')
plt.show();

np.save('./files/test_loss31.npy', test_loss8)
np.save('./files/train_loss31.npy', train_loss8)
np.save('./files/test_acc31.npy', test_acc8)
np.save('./files/train_acc31.npy', train_acc8)

# Run predictions for full data set
probs = model8.predict(X)
preds = (probs > 0.5).astype(int)

np.save('./files/probs31.npy', probs)
np.save('./files/preds31.npy', preds)

# plot the confusion matrix
confusion_mtx = confusion_matrix(y, preds) 
f,ax = plt.subplots(figsize=(8, 8))
sns.heatmap(confusion_mtx, annot=True, linewidths=0.01,cmap="copper")
plt.xlabel("Prediction", fontsize=20)
plt.xticks([0,1], ['Blueberry Muffin', 'Chihuahua'], fontsize=16)
plt.ylabel("Actual", fontsize=20)
plt.xticks([0,1], ['Blueberry Muffin', 'Chihuahua'], fontsize=16)
plt.title("Confusion Matrix", fontsize=24)
plt.savefig('./files/cm31.png')

