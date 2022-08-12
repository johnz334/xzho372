import tensorflow as tf
from tensorflow.python.layers import convolutional as tfl
from tensorflow.python.layers.normalization import batch_norm as batch_n
from tensorflow.python.layers.pooling import max_pool2d as max_pooling2d
def vgg_block(inputs, filters, kernel_size, name, data_format, training=False,
              batch_normalization=True, kernel_reg=0., **params):
    with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
        x = tfl.conv2d(inputs, filters, kernel_size, name='conv',
                       kernel_regularizer=tf.contrib.layers.l2_regularizer(kernel_reg),
                       data_format=data_format, **params)
        if batch_normalization:
            x = batch_n(
                    x, training=training, name='bn', fused=True,
                    axis=1 if data_format == 'channels_first' else -1)
    return x

def vgg_de_block(inputs, filters, kernel_size, name, data_format, training=False,
              batch_normalization=True, kernel_reg=0., **params):
    with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
        x = tfl.conv2d_transpose(inputs, filters, kernel_size, name='conv',
                                 strides=(2, 2),
                       kernel_regularizer=tf.contrib.layers.l2_regularizer(kernel_reg),
                       data_format=data_format, **params)
        if batch_normalization:
            x = batch_n(
                    x, training=training, name='bn', fused=True,
                    axis=1 if data_format == 'channels_first' else -1)
    return x


def vgg_backbone(inputs, **config):
    params_conv = {'padding': 'SAME', 'data_format': config['data_format'],
                   'activation': tf.nn.relu, 'batch_normalization': True,
                   'training': config['training'],
                   'kernel_reg': config.get('kernel_reg', 0.)}
    params_pool = {'padding': 'SAME', 'data_format': config['data_format']}
    params_conv1 = {'padding': 'valid', 'data_format': config['data_format'],
                   'activation': tf.nn.relu, 'batch_normalization': True,
                   'training': config['training'],
                   'kernel_reg': config.get('kernel_reg', 0.)}
    with tf.variable_scope('vgg', reuse=tf.AUTO_REUSE):
        x = vgg_block(inputs, 64, 3, 'conv1_1', **params_conv)
        x = vgg_block(x, 64, 3, 'conv1_2', **params_conv)
        x = max_pooling2d(x, 2, 2, name='pool1', **params_pool)

        x = vgg_block(x, 64, 3, 'conv2_1', **params_conv)
        x = vgg_block(x, 64, 3, 'conv2_2', **params_conv)
        x = max_pooling2d(x, 2, 2, name='pool2', **params_pool)

        x = vgg_block(x, 128, 3, 'conv3_1', **params_conv)
        x = vgg_block(x, 128, 3, 'conv3_2', **params_conv)

        x = max_pooling2d(x, 2, 2, name='pool3', **params_pool)

        x = vgg_block(x, 128, 3, 'conv4_1', **params_conv)
        identity = vgg_block(x, 128, 3, 'conv4_2', **params_conv)

        x = vgg_block(identity, 256, 3, 'conv5_1', **params_conv)
        x = vgg_block(x, 256, 3, 'conv5_2', **params_conv)
        x = max_pooling2d(x, 2, 2, name='pool4', **params_pool)

        x = vgg_de_block(x, 128, 3, 'deconv5_1', **params_conv)

        '''paddings = tf.constant([[0,0],[0,0],[1,0],[1,1]])
        x = tf.pad(x, paddings, "CONSTANT") for synthetic_shape training
        x = vgg_block(x, 128, 3, 'deconv6_1', **params_conv1)'''

        x = tf.concat([identity, x], 1)

        x = vgg_block(x, 128, 3, 'conv7_1', **params_conv)


    return x

if __name__ == '__main__':
    tensor_normal = tf.random.normal([1, 3, 120, 160], mean=0.0, stddev=1.0)
    config = {"data_format": "channels_last", 'training': True}
    y = vgg_backbone(tensor_normal, data_format = "channels_first", training = True)


    print(y.shape)

