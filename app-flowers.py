# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.optimizers import Adam

# Helper libraries
#import matplotlib.pyplot as plt
import numpy as np
from os import listdir
from os.path import join
import cv2
import pandas
import os
import random
import pathlib

# Set the path of the input folder

dataset = "https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz"
directory = '/home/codespace/.keras/datasets/flower_photos/flower_photos/'
data = pathlib.Path(directory)

#data = '/home/codespace/.keras/datasets/flower_photos'
print(data)
folders = os.listdir(data)
print(folders)

# Import the images and resize them to a 128*128 size
# Also generate the corresponding labels

image_names = []
train_labels = []
train_images = []

size = 64,64

for folder in folders:
    if not os.path.isdir(os.path.join(data, folder)):
        continue
    for file in os.listdir(os.path.join(data,folder)):
        if file.endswith("jpg"):
            image_names.append(os.path.join(data,folder,file))
            train_labels.append(folder)
            img = cv2.imread(os.path.join(data,folder,file))
            im = cv2.resize(img,size)
            train_images.append(im)
        else:
            continue

# Transform the image array to a numpy type

train = np.array(train_images)
print(train.shape)

# Reduce the RGB values between 0 and 1
train = train.astype('float32') / 255.0
# Extract the labels
label_dummies = pandas.get_dummies(train_labels)
labels =  label_dummies.values.argmax(1)

print(pandas.unique(train_labels))
print(pandas.unique(labels))

# Shuffle the labels and images randomly for better results

union_list = list(zip(train, labels))
random.shuffle(union_list)
train,labels = zip(*union_list)

# Convert the shuffled list to numpy array type

train = np.array(train)
labels = np.array(labels)


# Develop a sequential model using tensorflow keras
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(64,64,3)),
    keras.layers.Dense(128, activation=tf.nn.tanh),
    keras.layers.Dense(5, activation=tf.nn.softmax)
])

print(model.summary())
w, b = model.layers[1].get_weights()
print(w.shape)
print(b.shape)

w, b = model.layers[2].get_weights()
print(w.shape)
print(b.shape)

model.compile(optimizer=Adam(learning_rate=0.001),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(train,labels, epochs=50)
w, b = model.layers[1].get_weights()
print(w.shape)
print(w)
print(b.shape)
print(b)

w, b = model.layers[2].get_weights()
print(w.shape)
print(w)
print(b.shape)
print(b)

img_height = 64
img_width = 64
import cv2

test_folder = os.path.join(str(data), 'tulips')

# Filtramos solo archivos .jpg para evitar errores
available_images = [f for f in os.listdir(test_folder) if f.endswith('.jpg')]
first_image_name = available_images[0] # Tomamos la primera que exista

path_to_test = os.path.join(test_folder, first_image_name)
print(f"Probando predicción con: {path_to_test}")

image = cv2.imread(path_to_test)

#img_path = os.path.join(str(data), 'roses', '10090824183_d02c613f10_m.jpg')
#image = cv2.imread(img_path)
#image = cv2.imread('/root/.keras/datasets/flower_photos/tulips/100930342_92e8746431_n.jpg')
#image = cv2.imread('/home/adsoft/.keras/datasets/flower_photos/sunflowers/1008566138_6927679c8a.jpg')


image_resized = cv2.resize(image, (img_height, img_width))
image = np.expand_dims(image_resized, axis=0)


# Make predictions
image_pred = model.predict(image)

print(image_pred)
# Produce a human-readable output label\
classes_labels = pandas.unique(train_labels)


image_output_class = classes_labels[np.argmax(image_pred)]
print(classes_labels)
print(np.argmax(image_pred))

print("The predicted class is", image_output_class)

export_path = 'flowers-model/1/'
tf.saved_model.save(model, os.path.join('./',export_path))
