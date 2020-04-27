# desc:    U-Net model with partial convolutions to inpaint
#          128x128x3 images
# -------------------------------------------------------------
import tensorflow as tf
from keras.layers import Conv2D, UpSampling2D, LeakyReLU, BatchNormalization, Activation
from keras.layers.merge import Concatenate

from pconv_layer import PConv2D

print('Imported model (for Places365, 128x128 images with partial convolutions)')

def generator(z, train_bn=False):

    with tf.variable_scope('G', reuse=tf.AUTO_REUSE):

        img = z[:, :, :, :-1]

        # get, invert and broadcast mask to 3 channels
        mask = z[:, :, :, -1]
        mask = 1 - mask  # implementation expects 0 for inpainted pixels
        mask = tf.expand_dims(mask, -1)
        mask = tf.concat([mask for _ in range(3)], -1)

        # ENCODER
        def encoder_layer(img_in, mask_in, filters, kernel_size, bn=True):
            conv, mask = PConv2D(filters, kernel_size, strides=2, padding='same')([img_in, mask_in])

            # add batch normalization if desired
            if bn:
                conv = BatchNormalization(name='EncBN' + str(encoder_layer.counter))(conv, training=train_bn)
            conv = Activation('relu')(conv)

            encoder_layer.counter += 1

            return conv, mask

        encoder_layer.counter = 0

        e_conv1, e_mask1 = encoder_layer(img, mask, 64, 5, bn=False)
        e_conv2, e_mask2 = encoder_layer(e_conv1, e_mask1, 128, 5)
        e_conv3, e_mask3 = encoder_layer(e_conv2, e_mask2, 256, 3)
        e_conv4, e_mask4 = encoder_layer(e_conv3, e_mask3, 256, 3)
        e_conv5, e_mask5 = encoder_layer(e_conv4, e_mask4, 256, 3)

        # DECODER
        def decoder_layer(img_in, mask_in, e_conv, e_mask, filters, kernel_size, bn=True):

            # up-sample and concatenate
            up_img = UpSampling2D(size=(2, 2))(img_in)
            up_mask = UpSampling2D(size=(2, 2))(mask_in)
            concat_img = Concatenate(axis=3)([e_conv, up_img])
            concat_mask = Concatenate(axis=3)([e_mask, up_mask])

            conv, mask = PConv2D(filters, kernel_size, padding='same')([concat_img, concat_mask])

            # add batch normalization if desired
            if bn:
                conv = BatchNormalization()(conv)
            conv = LeakyReLU(alpha=0.2)(conv)

            return conv, mask

        d_conv6, d_mask6 = decoder_layer(e_conv5, e_mask5, e_conv4, e_mask4, 256, 3)
        d_conv7, d_mask7 = decoder_layer(d_conv6, d_mask6, e_conv3, e_mask3, 256, 3)
        d_conv8, d_mask8 = decoder_layer(d_conv7, d_mask7, e_conv2, e_mask2, 128, 3)
        d_conv9, d_mask9 = decoder_layer(d_conv8, d_mask8, e_conv1, e_mask1, 64, 3)
        d_conv10, d_mask10 = decoder_layer(d_conv9, d_mask9, img, mask, 3, 3, bn=False)
        outputs = Conv2D(3, 1, activation='sigmoid', name='outputs_img')(d_conv10)

    return outputs

def global_discriminator(x):
    with tf.variable_scope('DG', reuse=tf.AUTO_REUSE):
        conv1 = tf.layers.conv2d(
            inputs=x,
            filters=32,
            kernel_size=[5, 5],
            strides=(2, 2),
            padding="same",
            activation=tf.nn.relu)

        conv2 = tf.layers.conv2d(
            inputs=conv1,
            filters=64,
            kernel_size=[5, 5],
            strides=(2, 2),
            padding="same",
            activation=tf.nn.relu)

        conv3 = tf.layers.conv2d(
            inputs=conv2,
            filters=64,
            kernel_size=[5, 5],
            strides=(2, 2),
            padding="same",
            activation=tf.nn.relu)

        conv4 = tf.layers.conv2d(
            inputs=conv3,
            filters=128,
            kernel_size=[5, 5],
            strides=(2, 2),
            padding="same",
            activation=tf.nn.relu)

        conv5 = tf.layers.conv2d(
            inputs=conv4,
            filters=128,
            kernel_size=[5, 5],
            strides=(2, 2),
            padding="same",
            activation=tf.nn.relu)

        conv5_flat = tf.layers.flatten(
            inputs=conv5)

        dense6 = tf.layers.dense(
            inputs=conv5_flat,
            units=512,
            activation=tf.nn.relu)

    return dense6

def local_discriminator(x):
    with tf.variable_scope('DL', reuse=tf.AUTO_REUSE):
        conv1 = tf.layers.conv2d(
            inputs=x,
            filters=32,
            kernel_size=[5, 5],
            strides=(2, 2),
            padding="same",
            activation=tf.nn.relu)

        conv2 = tf.layers.conv2d(
            inputs=conv1,
            filters=64,
            kernel_size=[5, 5],
            strides=(2, 2),
            padding="same",
            activation=tf.nn.relu)

        conv3 = tf.layers.conv2d(
            inputs=conv2,
            filters=128,
            kernel_size=[5, 5],
            strides=(2, 2),
            padding="same",
            activation=tf.nn.relu)

        conv4 = tf.layers.conv2d(
            inputs=conv3,
            filters=128,
            kernel_size=[5, 5],
            strides=(2, 2),
            padding="same",
            activation=tf.nn.relu)

        conv4_flat = tf.layers.flatten(
            inputs=conv4)

        dense5 = tf.layers.dense(
            inputs=conv4_flat,
            units=512,
            activation=tf.nn.relu)

    return dense5

def concatenator(global_x, local_x_left, local_x_right):
    with tf.variable_scope('C', reuse=tf.AUTO_REUSE):
        dense1 = tf.layers.dense(
            inputs=tf.concat([global_x, local_x_left, local_x_right], axis=-1),
            units=1,
            activation=tf.sigmoid)

    return dense1
