import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras import applications
from keras.utils import to_categorical
from sklearn.metrics import mean_squared_error, accuracy_score, f1_score, confusion_matrix, classification_report
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd

# https://stackoverflow.com/questions/50928329/getting-x-test-y-test-from-generator-in-keras/50930515#50930515
from platform import python_version_tuple

if python_version_tuple()[0] == '3':
    xrange = range
    izip = zip
    imap = map
else:
    from itertools import izip, imap

import numpy as np


datagen = ImageDataGenerator(
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest')

model = Sequential()
model.add(Conv2D(32, (3, 3), input_shape=(150, 150, 1)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(1))
model.add(Activation('sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])


batch_size = 16

train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)


# this is the augmentation configuration we will use for testing:
# only rescaling
test_datagen = ImageDataGenerator(rescale=1./255)

# this is a generator that will read pictures found in
# subfolers of 'data/train', and indefinitely generate
# batches of augmented image data
train_generator = train_datagen.flow_from_directory(
        'chest_xray/chest_xray/train',  # this is the target directory
        target_size=(150, 150),  # all images will be resized to 150x150
        color_mode = "grayscale",
        batch_size=batch_size,
        class_mode='binary')  # since we use binary_crossentropy loss, we need binary labels

test_generator = test_datagen.flow_from_directory(
        'chest_xray/chest_xray/test',
        target_size=(150, 150),
        color_mode = "grayscale",
        batch_size=624,
        class_mode='binary')

# this is a similar generator, for validation data
validation_generator = test_datagen.flow_from_directory(
        'chest_xray/chest_xray/val',
        target_size=(150, 150),
        color_mode = "grayscale",
        batch_size=batch_size,
        class_mode='binary')

history = model.fit_generator(
        train_generator,
        steps_per_epoch=2000 // batch_size,
        epochs=50,
        validation_data=validation_generator,
        validation_steps=800 // batch_size)
model.save_weights('first_try.h5')  # always save your weights after training or during training


X_test, y_test = test_generator.next()

y_pred = model.predict(X_test)
y_pred = np.rint(y_pred) # CNN gives out a procentage for how confident it is that it detects pheumonia this rounds that of to 1 or 0.
# MSE = mean_squared_error(y_pred, y_test)
# print(f'RMSE = {MSE**(1/2)}')
print(f'F1-score: {f1_score(y_test, y_pred)}')
print(f'accuracy: {accuracy_score(y_test, y_pred)}')
print(confusion_matrix(y_test, y_pred))

# pd.DataFrame(history.history['loss']).plot()
# plt.grid(True)
# plt.gca().set_ylim(0, 2)
# plt.xlabel('Epoch')
# plt.title('Learning curves')
# plt.show()
