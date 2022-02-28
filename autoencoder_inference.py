import os
import pandas as pd
from PIL import Image, ImageOps
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import time

import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator,load_img, img_to_array
from tensorflow.keras.metrics import categorical_crossentropy
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D,GlobalAveragePooling2D
from tensorflow.keras.layers import Activation, Dropout, BatchNormalization, Flatten, Dense, LeakyReLU
from tensorflow.keras.optimizers import Adam, SGD, RMSprop
from tensorflow.keras import backend as K
from tensorflow.keras import layers

import sklearn
from sklearn.model_selection import train_test_split

from natsort import natsorted

import cv2
import shutil
import glob

def dataset_collection_func(normal_class, abnormal_classes_list, abnormal_ratio):

    abnormal_classes_array = np.array(abnormal_classes_list)

    (train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.cifar10.load_data()
    
    print(train_labels)
    train_labels = np.squeeze(train_labels)
    test_labels = np.squeeze(test_labels)

    print("train_images.shape:", train_images.shape)

    print(train_images[0].shape)

    print(train_images[[1,2,3]].shape)

    print("train_labels.shape:", train_labels.shape)
    one_class_idx = np.where(train_labels == normal_class)
    print("one_class_idx:", one_class_idx)
    one_class_train_images = train_images[one_class_idx]
    one_class_train_labels = train_labels[one_class_idx]
    one_class_train_labels[one_class_train_labels==normal_class] = 0

    print("1 one_class_train_images.shape:", one_class_train_images.shape)

    other_class_idx = np.where(train_labels != normal_class)
    other_class_train_images = train_images[other_class_idx]
    other_class_train_labels = train_labels[other_class_idx]

    print("2 other_class_train_images.shape:", other_class_train_images.shape)

    other_class_train_should_idx = np.where((other_class_train_labels == abnormal_classes_array[0])
                                            | (other_class_train_labels == abnormal_classes_array[1])
                                            | (other_class_train_labels == abnormal_classes_array[2])
                                            | (other_class_train_labels == abnormal_classes_array[3])
                                            | (other_class_train_labels == abnormal_classes_array[4]))
    other_class_train_images = other_class_train_images[other_class_train_should_idx]
    other_class_train_labels = other_class_train_labels[other_class_train_should_idx]

    print("3 other_class_train_images.shape:", other_class_train_images.shape)

    np.random.seed(1234) # set seed
    idx = np.random.permutation(len(other_class_train_labels))
    other_class_train_images = other_class_train_images[[idx]]
    other_class_train_labels = other_class_train_labels[[idx]]

    print("4 other_class_train_images.shape:", other_class_train_images.shape)

    print("other_class_train_labels:", other_class_train_labels)

    other_class_train_labels[other_class_train_labels !=normal_class] = 1

    train_one_class_len = len(one_class_train_labels)

    train_other_class_should_len = int(train_one_class_len * abnormal_ratio)

    other_class_train_images = other_class_train_images[:train_other_class_should_len]
    other_class_train_labels = other_class_train_labels[:train_other_class_should_len]

    print("5 other_class_train_images.shape:", other_class_train_images.shape)

    print("majority train number:", len(one_class_train_labels))
    print("minority train number:", len(other_class_train_labels))
    # print("other_class_train_labels:", other_class_train_labels)

    train_images = np.concatenate((one_class_train_images,other_class_train_images),axis=0)
    train_labels = np.concatenate((one_class_train_labels,other_class_train_labels),axis=0)

    test_labels[test_labels!=normal_class] = 11
    test_labels[test_labels==normal_class] = 0
    test_labels[test_labels==11] = 1
    
    print("train_images.shape:", train_images.shape)

    return train_images, train_labels, test_images, test_labels

train_images, train_labels, test_images, test_labels = dataset_collection_func(normal_class = 5, abnormal_classes_list = [6,7,8,9,0], abnormal_ratio = 0.01)

normal_idx = np.where(train_labels==0)[0]
train_images_normal = train_images[normal_idx]

image_shape = (32,32,3)

img_dim = image_shape[0]

latent_dim = 128

model_encoder = tf.keras.Sequential(
    [
        tf.keras.layers.InputLayer(input_shape=(32, 32, 3)),
        tf.keras.layers.Conv2D(
            filters=32, kernel_size=5, strides=(1, 1), padding='same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.LeakyReLU(alpha=0.01),
        tf.keras.layers.MaxPool2D(),

        tf.keras.layers.Conv2D(
            filters=64, kernel_size=5, strides=(1, 1), padding='same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.LeakyReLU(alpha=0.01),
        tf.keras.layers.MaxPool2D(),

        tf.keras.layers.Conv2D(
            filters=128, kernel_size=5, strides=(1, 1), padding='same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.LeakyReLU(alpha=0.01),
        tf.keras.layers.MaxPool2D(),

        tf.keras.layers.Flatten(),

        # No activation
        tf.keras.layers.Dense(latent_dim),
    ]
)

model_decoder = tf.keras.Sequential(
    [
        tf.keras.layers.InputLayer(input_shape=(latent_dim,)),
        tf.keras.layers.Dense(units=4*4*64),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.LeakyReLU(alpha=0.01),
        tf.keras.layers.Reshape(target_shape=(4, 4, 64)),

        tf.keras.layers.Conv2DTranspose(
            filters=128, kernel_size=5, strides=1, padding='same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.LeakyReLU(alpha=0.01),
        tf.keras.layers.UpSampling2D(),

        tf.keras.layers.Conv2DTranspose(
            filters=64, kernel_size=5, strides=1, padding='same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.LeakyReLU(alpha=0.01),
        tf.keras.layers.UpSampling2D(),

        tf.keras.layers.Conv2DTranspose(
            filters=32, kernel_size=5, strides=1, padding='same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.LeakyReLU(alpha=0.01),
        tf.keras.layers.UpSampling2D(),

        # No activation
        tf.keras.layers.Conv2DTranspose(
            filters=3, kernel_size=3, strides=1, padding='same', activation='sigmoid'),
    ]
)

model = tf.keras.Sequential()
model.add(tf.keras.layers.InputLayer(input_shape=image_shape))
model.add(model_encoder)
model.add(model_decoder)

learning_rate = 1e-3
# optimizer = tf.keras.optimizers.Adam(learning_rate)
decayed_learning_rate = tf.keras.optimizers.schedules.ExponentialDecay(learning_rate, 1000, 0.96)
optimizer = tf.keras.optimizers.Adam(decayed_learning_rate)

checkpoint_dir = os.path.join(base_dir, 'autoencoder_checkpoints')

checkpoint = tf.train.Checkpoint(optimizer=optimizer,
                                 model=model)

ckpt_manager = tf.train.CheckpointManager(checkpoint, checkpoint_dir, max_to_keep=10)

checkpoint.restore(ckpt_manager.latest_checkpoint) # restore the lastest checkpoints

def generate_and_save_images(model, test_sample):
    predictions = model(test_sample)

    # print(predictions.shape)

    fig = plt.figure(figsize=(1, 2))

    test_sample = test_sample * 255
    test_sample = test_sample.astype(np.uint8)

    predictions = np.array(predictions)
    predictions = predictions * 255
    predictions = predictions.astype(np.uint8)

    print(predictions.shape)

    plt.subplot(1, 2, 1)
    plt.imshow(test_sample[0, :, :, :])
    plt.subplot(1, 2, 2)
    plt.imshow(predictions[0, :, :, :])

    plt.show()

def inference(images, i):

    img_array = images[i]

    print(img_array.dtype)

    img_array = img_array.astype(np.float32)
    img_array = img_array / 255.

    img_array = np.array(img_array)
    img_array = np.expand_dims(img_array, axis=0)

    generate_and_save_images(model, img_array)

inference(train_images_normal, 5)
