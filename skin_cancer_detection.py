# Importing Dependencies

import tensorflow as tf
from tensorflow import keras
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
import glob

#Loading Dataset

from google.colab import drive
drive.mount('/content/drive')

# Unzipping The Dataset

!unzip /content/drive/MyDrive/Datasets/archive.zip

# Visualising Images

for image in glob.glob('/content/data/test/benign/*'): 
  img = plt.imread(image)
  plt.imshow(img)
  plt.show()

# Image Preprocessing & Image Augmentation

train_dir = '/content/data/train/'
#Data Augmentation
train_datagen = ImageDataGenerator(
                                   rescale=1/255,
                                   rotation_range = 40,
                                   width_shift_range = 0.2,
                                   height_shift_range = 0.2,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True
                                   
)
#train_generator will assign binary labels to the images
train_generator = train_datagen.flow_from_directory(
                                        train_dir,
                                        target_size =(224,224),
                                        class_mode = 'binary',
                                        batch_size = 293,                  
)

validate_dir = '/content/data/test/'
#Data Augmentation
validate_datagen = ImageDataGenerator(
                                   rescale=1/255,
                                   rotation_range = 40,
                                   width_shift_range = 0.2,
                                   height_shift_range = 0.2,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True
                                   
)
#validation_generator will assign binary labels to the images
validation_generator = validate_datagen.flow_from_directory(
                                        validate_dir,
                                        target_size = (224,224),
                                        class_mode = 'binary',
                                        batch_size = 60
                                        
)

#Making The Model - CNNs

model = keras.Sequential([keras.layers.Conv2D(16,(3,3), activation="relu",input_shape=(224,224,3)),
                          keras.layers.MaxPool2D((2,2)),
                          keras.layers.Conv2D(32,(3,3), activation="relu"),
                          keras.layers.MaxPool2D((2,2)),
                          keras.layers.Conv2D(64,(3,3), activation="relu"),
                          keras.layers.MaxPool2D((2,2)),
                          keras.layers.Flatten(),
                          keras.layers.Dense(512, activation="relu"),
                          keras.layers.Dense(1, activation="sigmoid")
])

model.summary()
model.compile(loss="binary_crossentropy", optimizer=Adam(1e-3), metrics=['accuracy'])
model.fit(train_generator, epochs=10,steps_per_epoch=9, validation_data = validation_generator, validation_steps=11)
