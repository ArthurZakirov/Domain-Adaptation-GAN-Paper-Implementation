import h5py

import tensorflow as tf
from tensorflow.data import Dataset
import numpy as np

MNIST_M_PATH = "mnistm.h5"
BATCH_SIZE = 64
CHANNELS = 3
NUM_SAMPLES = 10000
VAL_SET = 0.2


def prepare_data():
    # Load MNIST Data (Source)
    (
        (mnist_train_x, mnist_train_y),
        (mnist_test_x, mnist_test_y),
    ) = tf.keras.datasets.mnist.load_data()

    # Convert to 3 Channel and One_hot labels
    mnist_train_x, mnist_test_x = (
        mnist_train_x.reshape((60000, 28, 28, 1)),
        mnist_test_x.reshape((10000, 28, 28, 1)),
    )
    mnist_train_x, mnist_test_x = (
        mnist_train_x[:NUM_SAMPLES],
        mnist_test_x[: int(NUM_SAMPLES * VAL_SET)],
    )
    mnist_train_y, mnist_test_y = (
        mnist_train_y[:NUM_SAMPLES],
        mnist_test_y[: int(NUM_SAMPLES * VAL_SET)],
    )
    mnist_train_x, mnist_test_x = mnist_train_x / 255.0, mnist_test_x / 255.0
    mnist_train_x, mnist_test_x = (
        mnist_train_x.astype("float32"),
        mnist_test_x.astype("float32"),
    )

    mnist_train_x = np.repeat(mnist_train_x, CHANNELS, axis=3)
    mnist_test_x = np.repeat(mnist_test_x, CHANNELS, axis=3)
    mnist_train_y = tf.one_hot(mnist_train_y, depth=10)
    mnist_test_y = tf.one_hot(mnist_test_y, depth=10)

    # Load MNIST-M [Target]

    with h5py.File(MNIST_M_PATH, "r") as mnist_m:
        mnist_m_train_x, mnist_m_test_x = (
            mnist_m["train"]["X"][()],
            mnist_m["test"]["X"][()],
        )

    mnist_m_train_x, mnist_m_test_x = (
        mnist_m_train_x[:NUM_SAMPLES],
        mnist_m_test_x[: int(NUM_SAMPLES * VAL_SET)],
    )
    mnist_m_train_x, mnist_m_test_x = (
        mnist_m_train_x / 255.0,
        mnist_m_test_x / 255.0,
    )
    mnist_m_train_x, mnist_m_test_x = (
        mnist_m_train_x.astype("float32"),
        mnist_m_test_x.astype("float32"),
    )
    mnist_m_train_y, mnist_m_test_y = mnist_train_y, mnist_test_y

    ds_stage_1_train = Dataset.from_tensor_slices(
        (mnist_train_x, mnist_train_y)
    ).batch(BATCH_SIZE)
    ds_stage_1_test = Dataset.from_tensor_slices(
        (mnist_test_x, mnist_test_y)
    ).batch(BATCH_SIZE)
    ds_stage_2_train = Dataset.from_tensor_slices(
        (mnist_train_x, mnist_train_y, mnist_m_train_x)
    ).batch(BATCH_SIZE)
    ds_stage_2_test = Dataset.from_tensor_slices(
        (mnist_test_x, mnist_test_y, mnist_m_test_x)
    ).batch(BATCH_SIZE)

    return ds_stage_1_train, ds_stage_1_test, ds_stage_2_train, ds_stage_2_test
