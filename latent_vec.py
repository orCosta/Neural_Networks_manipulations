#############################################################################################################
# Load the mnist AE model and scatter the latent vector samples
#############################################################################################################
from __future__ import division, print_function, absolute_import
import tensorflow as tf
import numpy as np
from scipy import linalg as LA
import matplotlib.pyplot as plt


# PCA:
def PCA(data, dims_rescaled_data=2):
    """
    returns: data transformed in 2 dims/columns + regenerated original data
    pass in: data as 2D NumPy array
    """
    m, n = data.shape
    data -= data.mean(axis=0)
    R = np.cov(data, rowvar=False)
    evals, evecs = LA.eigh(R)
    # sort eigenvalue in decreasing order
    idx = np.argsort(evals)[::-1]
    evecs = evecs[:,idx]
    # sort eigenvectors according to same index
    evals = evals[idx]
    evecs = evecs[:, :dims_rescaled_data]
    return np.dot(evecs.T, data.T).T, evals, evecs


from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

size_l1 = 256
size_l2 = 100
size_input = 784

# Input:
X = tf.placeholder("float", [None, size_input])

def weight_variable(shape):
    """weight_variable generates a weight variable of a given shape."""
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    """bias_variable generates a bias variable of a given shape."""
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def encoder(x):
    # Encoder 2 hidden FC layers with sigmoid activation:
    w_fc1 = weight_variable([size_input, size_l1])
    b_fc1 = bias_variable([size_l1])
    fc_1 = tf.nn.sigmoid(tf.add(tf.matmul(x, w_fc1), b_fc1))

    w_fc2 = weight_variable([size_l1, size_l2])
    b_fc2 = bias_variable([size_l2])
    fc_2 = tf.nn.sigmoid(tf.add(tf.matmul(fc_1, w_fc2), b_fc2))
    return fc_2


def decoder(x):
    # Decoder 2 hidden FC layers with sigmoid activation:
    w_fc1 = weight_variable([size_l2, size_l1])
    b_fc1 = bias_variable([size_l1])
    fc_1 = tf.nn.sigmoid(tf.add(tf.matmul(x, w_fc1), b_fc1))

    w_fc2 = weight_variable([size_l1, size_input])
    b_fc2 = bias_variable([size_input])
    fc_2 = tf.nn.sigmoid(tf.add(tf.matmul(fc_1, w_fc2), b_fc2))
    return fc_2


# AE Model
latent_vec = encoder(X)
decoder_op = decoder(latent_vec)

# Prediction
y_ = decoder_op
y = X

init = tf.global_variables_initializer()
saver = tf.train.Saver()

with tf.Session() as sess:
    saver.restore(sess, "model1.ckpt")
    print("Model restored")
    n = 2000
    batch_x, _ = mnist.test.next_batch(n)
    h = sess.run(latent_vec, feed_dict={X: batch_x})
    # np.save('mnist_latent_vec.npy', h)
    labels = np.argmax(_, axis=1)
    data, e1, e2 = PCA(h)
    plt.scatter(data[:, 0], data[:, 1], c=labels)
    plt.show()



