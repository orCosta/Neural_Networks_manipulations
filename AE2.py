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
    return tf.Variable(initial, name='N_1')

def bias_variable(shape):
    """bias_variable generates a bias variable of a given shape."""
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial, name='N_2')

size_l1 = 256
size_l2 = 100
size_input = 784

def encoder(x):
    # Encoder 2 hidden FC layers with sigmoid activation:
    w_fc1 = weight_variable([size_input, size_l1])
    b_fc1 = bias_variable([size_l1])
    fc_1 = tf.nn.sigmoid(tf.add(tf.matmul(x, w_fc1), b_fc1), name="FC1")

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



def Conv2d(input, output_dim=64, kernel=(5, 5), strides=(2, 2), stddev=0.2, name='conv_2d'):
    with tf.variable_scope(name):
        W = tf.get_variable('Conv2dW', [kernel[0], kernel[1], input.get_shape()[-1], output_dim],
                            initializer=tf.truncated_normal_initializer(stddev=stddev))
        b = tf.get_variable('Conv2db', [output_dim], initializer=tf.zeros_initializer())

        return tf.nn.conv2d(input, W, strides=[1, strides[0], strides[1], 1], padding='SAME') + b


def Deconv2d(input, output_dim, batch_size, kernel=(5, 5), strides=(2, 2), stddev=0.2, name='deconv_2d'):
    with tf.variable_scope(name):
        W = tf.get_variable('Deconv2dW', [kernel[0], kernel[1], output_dim, input.get_shape()[-1]],
                            initializer=tf.truncated_normal_initializer(stddev=stddev))
        b = tf.get_variable('Deconv2db', [output_dim], initializer=tf.zeros_initializer())

        input_shape = input.get_shape().as_list()
        output_shape = [batch_size,
                        int(input_shape[1] * strides[0]),
                        int(input_shape[2] * strides[1]),
                        output_dim]

        deconv = tf.nn.conv2d_transpose(input, W, output_shape=output_shape,
                                        strides=[1, strides[0], strides[1], 1])

        return deconv + b


def Dense(input, output_dim, stddev=0.02, name='dense'):
    with tf.variable_scope(name):
        shape = input.get_shape()
        W = tf.get_variable('DenseW', [shape[1], output_dim], tf.float32,
                            tf.random_normal_initializer(stddev=stddev))
        b = tf.get_variable('Denseb', [output_dim],
                            initializer=tf.zeros_initializer())

        return tf.matmul(input, W) + b


def BatchNormalization(input, name='bn'):
    with tf.variable_scope(name):

        output_dim = input.get_shape()[-1]
        beta = tf.get_variable('BnBeta', [output_dim],
                               initializer=tf.zeros_initializer())
        gamma = tf.get_variable('BnGamma', [output_dim],
                                initializer=tf.ones_initializer())

        if len(input.get_shape()) == 2:
            mean, var = tf.nn.moments(input, [0])
        else:
            mean, var = tf.nn.moments(input, [0, 1, 2])
        return tf.nn.batch_normalization(input, mean, var, beta, gamma, 1e-5)


def LeakyReLU(input, leak=0.2, name='lrelu'):
    return tf.maximum(input, leak * input)


def Discriminator(X, reuse=False, name='d'):

    with tf.variable_scope(name, reuse=reuse):

        if len(X.get_shape()) > 2:
            # X: -1, 28, 28, 1
            D_conv1 = Conv2d(X, output_dim=64, name='D_conv1')
        else:
            D_reshaped = tf.reshape(X, [-1, 28, 28, 1])
            D_conv1 = Conv2d(D_reshaped, output_dim=64, name='D_conv1')
        D_h1 = LeakyReLU(D_conv1) # [-1, 28, 28, 64]
        D_conv2 = Conv2d(D_h1, output_dim=128, name='D_conv2')
        D_h2 = LeakyReLU(D_conv2) # [-1, 28, 28, 128]
        D_r2 = tf.reshape(D_h2, [-1, 256])
        D_h3 = LeakyReLU(D_r2) # [-1, 256]
        D_h4 = tf.nn.dropout(D_h3, 0.5)
        D_h5 = Dense(D_h4, output_dim=1, name='D_h5') # [-1, 1]
        return tf.nn.sigmoid(D_h5)

# ======================================================================================================================
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

    ###################################################################################################################
    # This next vars is not taking a part in the training session
    ###################################################################################################################
    z = tf.placeholder(tf.float32, shape=[None, 100])
    # G = Generator(z, 'G')
    G = decoder(z)
    D_real = Discriminator(X, False, 'D')
    D_fake = Discriminator(G, True, 'D')
    D_loss = -tf.reduce_mean(tf.log(D_real) - tf.log(D_fake))  # Train to judge if the data is real correctly
    G_loss = -tf.reduce_mean(tf.log(D_fake))  # Train to pass the discriminator as real data
    vars = tf.trainable_variables()
    d_params = [v for v in vars if v.name.startswith('D/')]
    # g_params = [v for v in vars if v.name.startswith('G/')]
    g_params = [v for v in vars if v.name.startswith('N')]
    D_solver = tf.train.AdamOptimizer(learning_rate=1e-4, beta1=0.1).minimize(D_loss, var_list=d_params)
    G_solver = tf.train.AdamOptimizer(learning_rate=2e-4, beta1=0.3).minimize(G_loss, var_list=g_params)
    ###################################################################################################################

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

        save_path = saver.save(sess, "./model3.ckpt")
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
    AE()
