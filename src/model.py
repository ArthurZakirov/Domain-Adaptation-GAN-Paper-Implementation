import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


import numpy as np
import matplotlib.pyplot as plt


dropout = 0.5
conv_activation = "sigmoid"
clf_activation = "relu"
disc_activation = "relu"
input_shape = (28, 28, 1)
hidden_len = 512
feature_len = 256
disc_len = 1024
hidden_depth = 10
kernel_size = 3
num_conv_layers = 3
num_classes = 10


class FeatureExtractor(tf.keras.layers.Layer):
    def __init__(self):
        super(FeatureExtractor, self).__init__()
        self.feature_extractor = keras.Sequential(
            [
                layers.Conv2D(
                    filters=16, kernel_size=(2, 2), activation="relu"
                ),
                layers.Dropout(dropout),
                layers.MaxPool2D(pool_size=(2, 2)),
                layers.Conv2D(
                    filters=32, kernel_size=(2, 2), activation="relu"
                ),
                layers.Dropout(dropout),
                layers.MaxPool2D(pool_size=(2, 2)),
                layers.Conv2D(
                    filters=64, kernel_size=(2, 2), activation="relu"
                ),
                layers.Dropout(dropout),
                layers.MaxPool2D(pool_size=(2, 2)),
                layers.Flatten(),
                layers.Dense(256, activation="relu"),
                layers.Dropout(dropout),
            ]
        )

    def call(self, x):
        return self.feature_extractor(x)


class Classifier(tf.keras.layers.Layer):
    def __init__(self):
        super(Classifier, self).__init__()
        self.MLP = keras.Sequential(
            [
                layers.Dense(feature_len, activation=clf_activation),
                layers.Dense(num_classes, activation="softmax"),
            ]
        )

    def call(self, x):
        return self.MLP(x)


class Discriminator(tf.keras.layers.Layer):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.MLP = keras.Sequential(
            [
                layers.Dense(disc_len, activation=disc_activation),
                layers.Dense(disc_len, activation=disc_activation),
                layers.Dense(1, activation="sigmoid"),
            ]
        )

    def call(self, x):
        return self.MLP(x)


class Backbone(tf.keras.Model):
    def __init__(self):
        super(Backbone, self).__init__()
        self.f = FeatureExtractor()
        self.clf = Classifier()

    def call(self, x):
        return self.clf(self.f(x))
