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

c = np.load(os.path.join(base_dir, "c.npy"))

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

half_model = Model(inputs=model.layers[0].input, outputs=model.layers[0].output)

learning_rate = 1e-3
optimizer = tf.keras.optimizers.Adam(learning_rate)

checkpoint_dir = os.path.join(base_dir, 'deep_sad_checkpoints')
checkpoint = tf.train.Checkpoint(optimizer=optimizer,
                                 model=half_model)

ckpt_manager = tf.train.CheckpointManager(checkpoint, checkpoint_dir, max_to_keep=1)

def compute_loss(labels, z, c, eps=1e-4):

    c_tf = tf.Variable(c, dtype=tf.float32, trainable=False)

    dist = tf.math.reduce_sum((z - c_tf) ** 2, axis=1)

    # print("dist:", dist)

    euclidean_norm = tf.math.sqrt(dist)

    labels = tf.squeeze(labels)

    normal_idx = tf.where(labels == 0)
    abnormal_idx = tf.where(labels == 1)

    normal_idx = tf.squeeze(normal_idx)
    abnormal_idx = tf.squeeze(abnormal_idx)

    normal_labels = 1 - labels
    normal_labels = tf.cast(normal_labels, dtype=tf.float32)
    normal_dist_loss = normal_labels * dist

    abnormal_labels = labels
    abnormal_labels = tf.cast(abnormal_labels, dtype=tf.float32)
    
    abnormal_dist_loss = abnormal_labels * (1. / (dist + eps))

    loss = tf.math.reduce_mean(normal_dist_loss + abnormal_dist_loss)

    print("loss:", loss)

    return loss

def my_metrics(y_true, y_pred):
    y_true = np.squeeze(y_true)
    y_pred = np.squeeze(y_pred)

    y_pred = np.where(y_pred >= 0.5, 1, 0)

    TP, TN, FP, FN = 0, 0, 0, 0
    for prediction, y in zip(y_pred, y_true):

        if(prediction == y):
            if(prediction == 1): # {'No': 0, 'Yes': 1}
                TP += 1
            else:
                TN += 1
        else:
            if(prediction == 1):
                FP += 1
            else:
                FN += 1

    precision = TP/(TP+FP+1.0e-4)

    recall = TP/(TP+FN+1.0e-4)

    f_measure = (2. * precision * recall)/(precision + recall + 1.0e-4)

    accuracy = (TP + TN) / (TP + TN + FP + FN+1.0e-4)

    # print("TP:", TP)
    # print("TN:", TN)
    # print("FP:", FP)
    # print("FN:", FN)

    # print("precision:", precision)
    # print("recall:", recall)
    # print("f_measure:", f_measure)
    # print("accuracy:", accuracy)

    return np.array([TP, TN, FP, FN, precision, recall, f_measure, accuracy])

def train_step(inputs, labels, c, optimizer):
    # print("training......")

    with tf.GradientTape() as tape:
        z = half_model(inputs, training=True)
        loss = compute_loss(labels, z, c)
        # print(loss)

    gradients = tape.gradient(loss, half_model.trainable_variables)

    optimizer.apply_gradients(zip(gradients, half_model.trainable_variables))

    return np.array(loss), optimizer

def get_R(dist, nu):
    return np.quantile(dist, 1 - nu)

def get_R_based_on_c(train_images, train_labels, nu, c, eps = 1e-4):

    latent_features_list = []

    for index in range(len(train_images)):
        if(train_labels[index] == 0):
            image = train_images[index]
            image = np.expand_dims(image, axis=0)

            image = image.astype(np.float32)
            image = image / 255.

            latent_features = half_model(image, training=False)
            latent_features = np.array(latent_features)

            latent_features_list.append(latent_features[0])

    latent_features_list = np.array(latent_features_list)

    normal_dist = np.sum((latent_features_list - c) ** 2, axis=1)

    normal_euclidean_norm = np.sqrt(normal_dist)

    R = get_R(normal_euclidean_norm, nu)

    # R = tf.Variable(R, dtype=tf.float32, trainable=False)

    return R

def customized_R_evaluation(train_images, train_labels, batch_images, batch_labels, nu, c):

    R = get_R_based_on_c(train_images, train_labels, nu, c)

    print("R:", R)

    predictions = []

    scores_list = []
    dist_list = []

    for i in range(len(batch_labels)):
        img_path = batch_images[i]

        img_array = batch_images[i]

        img_array = np.expand_dims(img_array, axis=0)

        img_array = img_array.astype(np.float32)

        img_array = img_array / 255.

        z = half_model(img_array, training=False)

        dist = np.sum((z[0] - c) ** 2)
        dist = np.sqrt(dist)

        dist_list.append(dist)

        scores = R - dist

        scores_list.append(scores)

        if(scores >= 0):
            predictions.append(0)
        else:
            predictions.append(1)

    labels = np.array(batch_labels)
    predictions = np.array(predictions)

    metric_results = my_metrics(labels, predictions)

    scores_list = np.array(scores_list)
    dist_list = np.array(dist_list)

    auc_roc = sklearn.metrics.roc_auc_score(labels, dist_list)

    print("TP:", metric_results[0])
    print("TN:", metric_results[1])
    print("FP:", metric_results[2])
    print("FN:", metric_results[3])

    print("precision:", metric_results[4])
    print("recall:", metric_results[5])
    print("f_measure:", metric_results[6])
    print("accuracy:", metric_results[7])
    print("auc_roc:", auc_roc)

    return metric_results, auc_roc

def train(train_images, train_labels, test_images, test_labels, epochs, BATCH_SIZE, nu):

    global learning_rate
    global optimizer

    print("train_images.shape in main train:", train_images.shape)
    best_auc_roc = 0

    for epoch in range(epochs):
        start = time.time()

        idx = np.random.permutation(len(train_images))
        train_images = train_images[idx]
        train_labels = train_labels[idx]

        print("train epoch = ",epoch)
        for index in range(0, len(train_images)-BATCH_SIZE, BATCH_SIZE):
            label_batch = []

            for i in range(BATCH_SIZE):

                img_array = train_images[index+i]
                # img_array = np.expand_dims(img_array, axis=-1)

                img_array = img_array.astype(np.float32)
                img_array = img_array / 255.

                # data augmentation
                img_array = tf.keras.preprocessing.image.random_rotation(img_array, 0.2)
                img_array = tf.keras.preprocessing.image.random_shift(img_array, 0.1, 0.1)
                img_array = tf.keras.preprocessing.image.random_shear(img_array, 0.1)
                img_array = tf.keras.preprocessing.image.random_zoom(img_array, (0.7,1))

                img_array = np.array(img_array)
                img_array = np.expand_dims(img_array, axis=0)

                if(i == 0):
                    image_batch = img_array
                else:
                    image_batch = np.concatenate((image_batch, img_array), axis=0)

                label_batch.append(train_labels[index+i])
            
            label_batch = np.array(label_batch)
            label_batch = np.expand_dims(label_batch, axis=-1)

            loss, optimizer = train_step(image_batch, label_batch, c, optimizer)

            # print("training Loss: ", loss)

        print("saveing model")
        ckpt_manager.save()

        f = open("epoch_num.txt", "w")
        f.write(str(epoch))
        f.close()

        if(epoch % 10 == 0):

            idx_val = np.random.permutation(len(test_labels))
            test_images, test_labels = test_images[idx_val], test_labels[idx_val]

            metric_results, auc_roc = customized_R_evaluation(train_images, train_labels, test_images, test_labels, nu, c)

            if(auc_roc > best_auc_roc):
                best_auc_roc = auc_roc

                f = open("best_auc_roc_cifar10.txt", "w")
                f.write(str(best_auc_roc))
                f.close()

        if(epoch == 200):
            learning_rate = learning_rate * 0.1
            optimizer = tf.keras.optimizers.Adam(learning_rate)

        if(epoch == 250):
            learning_rate = learning_rate * 0.1
            optimizer = tf.keras.optimizers.Adam(learning_rate)

    print("saveing model")
    ckpt_manager.save()

epochs = 300 + 1
BATCH_SIZE = 256
nu = 0.05 # it should be a fine-tuned hyper-parameter
train(train_images, train_labels, test_images, test_labels, epochs, BATCH_SIZE, nu)
