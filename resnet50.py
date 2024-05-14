from tensorflow.keras.layers import Conv2D, Dense, Dropout, Flatten, MaxPooling2D, BatchNormalization, Activation
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from tensorflow.keras.models import Sequential
from tensorflow.keras.applications import ResNet50, resnet
from tensorflow.keras import optimizers
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.metrics import AUC
from tensorflow.keras.models import load_model
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import PIL
import os
import cv2

train_path = "archive/Data/train"
test_path = "archive/Data/test"
valid_path = "archive/Data/valid"

image_shape = (305,430,3)
N_CLASSES = 4
BATCH_SIZE = 32

train_datagen = ImageDataGenerator(dtype='float32', rescale= 1./255.,featurewise_center=False,
        featurewise_std_normalization=False,
        samplewise_std_normalization=False,
        zca_whitening=False,
        rotation_range=4,
        zoom_range = 0.2,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=False,
        vertical_flip=False)
train_generator = train_datagen.flow_from_directory(train_path,
                                                   batch_size = BATCH_SIZE,
                                                   target_size = (305,430),
                                                   class_mode = 'categorical')

valid_datagen = ImageDataGenerator(dtype='float32', rescale= 1.0/255.)
valid_generator = valid_datagen.flow_from_directory(valid_path,
                                                   batch_size = BATCH_SIZE,
                                                   target_size = (305,430),
                                                   class_mode = 'categorical')

test_datagen = ImageDataGenerator(dtype='float32', rescale = 1.0/255.0)
test_generator = test_datagen.flow_from_directory(test_path,
                                                   batch_size = BATCH_SIZE,
                                                   target_size = (305,430),
                                                   class_mode = 'categorical')

res_model = ResNet50(include_top = False, pooling = "avg", weights = "imagenet", input_shape = (image_shape))

for layer in res_model.layers:
    if "conv5" not in layer.name:
        layer.trainable = False

model = Sequential()
model.add(res_model)
model.add(Dropout(0.3))
model.add(Flatten())
model.add(BatchNormalization())
model.add(Dropout(0.3))
model.add(Dense(N_CLASSES, activation='softmax'))
model.summary()

optimizer = optimizers.Adam(learning_rate= 0.00001)
model.compile(optimizer=optimizer, loss = 'categorical_crossentropy', metrics = ['acc', AUC()])

checkpointer = ModelCheckpoint(filepath='check(85).hdf5',
                            monitor='val_loss', verbose = 1,
                            save_best_only=True)
early_stopping = EarlyStopping(verbose=1, patience=15)

hist = model.fit(train_generator,
                    steps_per_epoch = 20,
                    epochs = 85,
                    verbose = 1,
                    validation_data = valid_generator
                    ,callbacks = [checkpointer
                                  # , early_stopping
                                  ]
                    )

# model.save_weights("resnet_weights4(85)_nonstop.h5")
# model.save("resnet4_nonstop.h5")

# model = load_model("resnet4_nonstop.h5")

# model.summary()
# model.load_weights("resnet_weights4(85)_nonstop.h5")
# print(model.weights)

result = model.evaluate(test_generator)

plt.plot(hist.history['acc'], label = 'train',)
plt.plot(hist.history['val_acc'], label = 'val')

plt.legend(loc = 'right')
plt.xlabel('epochs')
plt.ylabel('accuracy')
plt.show()

plt.plot(hist.history['loss'], label = 'train',)
plt.plot(hist.history['val_loss'], label = 'val')

plt.legend(loc = 'right')
plt.xlabel('epochs')
plt.ylabel('loss')
plt.show()



























