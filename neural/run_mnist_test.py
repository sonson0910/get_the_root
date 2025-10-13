import sys
import site
import tensorflow as tf
from keras.datasets import mnist

print("PYTHON:", sys.executable)
print("SITE:", ";".join(site.getsitepackages()))
print("TF VERSION:", tf.__version__)

(x_train, y_train), (x_test, y_test) = mnist.load_data()
print("MNIST shapes:", x_train.shape, y_train.shape, x_test.shape, y_test.shape)
