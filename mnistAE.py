from __future__ import division, print_function, absolute_import
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# Import MNIST data
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

def weight_variable(shape):
    """weight_variable generates a weight variable of a given shape."""
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    """bias_variable generates a bias variable of a given shape."""
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

size_l1 = 256
size_l2 = 100
size_input = 784

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


def AE():
    learning_rate = 0.01
    num_steps = 10000
    batch_size = 200
    display_step = 1000

    # Input
    X = tf.placeholder("float", [None, size_input])

    # AE Model
    latent_vec = encoder(X)
    decoder_op = decoder(latent_vec)

    # Prediction
    y_ = decoder_op
    y = X

    # Loss - MSE
    loss = tf.reduce_mean(tf.pow(y - y_, 2))
    optimizer = tf.train.RMSPropOptimizer(learning_rate).minimize(loss)

    init = tf.global_variables_initializer()
# ======================================================
# ====================== Training ======================
# ======================================================
    saver = tf.train.Saver()
    with tf.Session() as sess:
        sess.run(init)
        for i in range(1, num_steps+1):
            # next batch
            batch_x, _ = mnist.train.next_batch(batch_size)

            _, l = sess.run([optimizer, loss], feed_dict={X: batch_x})
            if i % display_step == 0 or i == 1:
                print('Step %i: Minibatch Loss: %f' % (i, l))

        # Saving the model
        save_path = saver.save(sess, "model1.ckpt")
        print("Model saved in path: %s" % save_path)

        # Test
        canvas = np.empty((28, 56))
        batch_x, _ = mnist.test.next_batch(1)
        r = sess.run(decoder_op, feed_dict={X: batch_x})
        canvas[0: 28, 0: 28] = batch_x[0].reshape([28, 28])
        canvas[0: 28, 28:56] = r[0].reshape([28, 28])

        plt.figure()
        plt.title("Example: in => out")
        plt.imshow(canvas, origin="upper", cmap="gray")
        plt.show()


def denoising_AE():
    learning_rate = 0.01
    num_steps = 10000
    batch_size = 200
    display_step = 1000
    noise_level = 1.5

    # Input
    X = tf.placeholder("float", [None, size_input])
    y = tf.placeholder("float", [None, size_input])

    # AE Model
    latent_vec = encoder(X)
    decoder_op = decoder(latent_vec)

    # Prediction
    y_ = decoder_op

    # Loss - MSE
    loss = tf.reduce_mean(tf.pow(y - y_, 2))
    optimizer = tf.train.RMSPropOptimizer(learning_rate).minimize(loss)

    init = tf.global_variables_initializer()
# ======================================================
# ====================== Training ======================
# ======================================================
    saver = tf.train.Saver()
    with tf.Session() as sess:
        sess.run(init)
        for i in range(1, num_steps+1):
            # next batch
            batch_x, _ = mnist.train.next_batch(batch_size)
            batch_x_noisy = batch_x + noise_level * np.random.normal(loc=0.0, scale=1.0, size=batch_x.shape)

            _, l = sess.run([optimizer, loss], feed_dict={X: batch_x_noisy, y: batch_x})
            if i % display_step == 0 or i == 1:
                print('Step %i: Minibatch Loss: %f' % (i, l))

        # Saving the model
        save_path = saver.save(sess, "model2.ckpt")
        print("Model saved in path: %s" % save_path)

        # Test
        canvas = np.empty((28, 56))
        batch_x, _ = mnist.test.next_batch(1)
        r = sess.run(decoder_op, feed_dict={X: batch_x})
        canvas[0: 28, 0: 28] = batch_x[0].reshape([28, 28])
        canvas[0: 28, 28:56] = r[0].reshape([28, 28])

        plt.figure()
        plt.title("Example: in => out")
        plt.imshow(canvas, origin="upper", cmap="gray")
        plt.show()

if __name__ == '__main__':
    denoising_AE()
    AE()
