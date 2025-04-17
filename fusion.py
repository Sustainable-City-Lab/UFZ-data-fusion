import tensorflow as tf


def fusion_m(msi, gli, bdh, poi, dsm):
    x1_mii = tf.concat([msi, bdh, dsm], axis=-1)
    x2_giu = tf.concat([gli, bdh, dsm], axis=-1)
    x3_poi = tf.concat([poi, bdh, dsm], axis=-1)

    x23 = DBI(x2_giu, x3_poi) + x2_giu
    x32 = DBI(x3_poi, x2_giu) + x3_poi

    x123 = TMA(x2_giu, x1_mii, x3_poi)

    x123_di = dilated_convolution(x123)
    x123_di = cbr1(x123_di)

    G1 = tf.keras.layers.Conv2D(filters=1, kernel_size=1, strides=1, padding='same', activation=tf.nn.sigmoid)(x23)
    G2 = tf.keras.layers.Conv2D(filters=1, kernel_size=1, strides=1, padding='same', activation=tf.nn.sigmoid)(x32)
    G_sum = G1 + G2 + 1e-8
    G1_norm = G1 / G_sum
    G2_norm = G2 / G_sum
    out = tf.concat([(1 + G1_norm) * x123_di, G2_norm * (x23 + x32)], axis=-1)
    return out


def cbr1(input):
    output = tf.keras.layers.Conv2D(filters=1, kernel_size=1, strides=1, padding='same', activation=tf.nn.relu)(input)
    return output


def cbr2(input):
    output = tf.keras.layers.Conv2D(filters=1, kernel_size=3, strides=1, padding='same', activation=tf.nn.relu)(input)
    return output


def DBI(x1, x2):
    x1 = channel_attention(x1)
    x2 = cbr2(x2)
    g = tf.keras.layers.Conv2D(filters=1, kernel_size=1, strides=1, padding='same', activation=tf.nn.sigmoid)(x2)
    x1 = x1 * g
    x1 = spatial_attention(x1)
    return x1


def TMA(x2, x1, x3):
    input_shape = x1.get_shape().as_list()
    _, height, width, channels = input_shape
    x2 = cbr2(x2)
    x1_1 = tf.reduce_mean(x1, axis=[1, 2], keepdims=True)
    x1_1 = tf.keras.layers.Dense(units=channels, activation=tf.nn.relu)(x1_1)
    x1 = x1 + x1_1
    x3 = cbr2(x3)
    x123 = tf.concat([x2, x1, x3], axis=-1)
    x123_cbr = cbr1(x123)
    out = x123 + x123_cbr
    return out


def dilated_convolution(input_tensor, dropout_rate=0.1):
    # First Downsample the input tensor
    downsampled = tf.keras.layers.Conv2D(filters=1, kernel_size=1, strides=2, activation=tf.nn.relu, padding='same')(
        input_tensor)
    downsampled = tf.keras.layers.Dropout(dropout_rate)(downsampled)

    # Second Depthwise Separable Convolution with 3x3 kernel and dilation rate 1
    conv1 = tf.keras.layers.DepthwiseConv2D(kernel_size=3, dilation_rate=1, activation=tf.nn.relu, padding='same')(
        downsampled)
    conv1 = tf.keras.layers.Conv2D(filters=1, kernel_size=1, activation=None, padding='same')(conv1)
    conv1 = tf.keras.layers.Dropout(dropout_rate)(conv1)

    # Third Depthwise Separable Convolution with 3x3 kernel and dilation rate 2
    conv2 = tf.keras.layers.DepthwiseConv2D(kernel_size=3, dilation_rate=2, activation=tf.nn.relu, padding='same')(
        downsampled)
    conv2 = tf.keras.layers.Conv2D(filters=1, kernel_size=1, activation=None, padding='same')(conv2)
    conv2 = tf.keras.layers.Dropout(dropout_rate)(conv2)

    # Fourth Depthwise Separable Convolution with 3x3 kernel and dilation rate 3
    conv3 = tf.keras.layers.DepthwiseConv2D(kernel_size=3, dilation_rate=3, activation=tf.nn.relu, padding='same')(
        downsampled)
    conv3 = tf.keras.layers.Conv2D(filters=1, kernel_size=1, activation=None, padding='same')(conv3)
    conv3 = tf.keras.layers.Dropout(dropout_rate)(conv3)

    # Upsample the feature maps back to the original size
    upsampled = tf.keras.layers.UpSampling2D(size=(1, 1))(tf.concat([conv1, conv2, conv3], axis=-1))

    # Reduce the number of channels after upsampling
    upsampled = tf.keras.layers.Conv2D(filters=1, kernel_size=1, activation=tf.nn.relu, padding='same')(upsampled)

    # Concatenate all the feature maps along the last axis (channels)
    output_tensor = tf.concat([input_tensor, upsampled], axis=-1)

    return output_tensor


def channel_attention(input_feature, ratio=8):
    channel_axis = 3
    channel_in = input_feature.get_shape()[channel_axis]
    shared_layer_one = tf.keras.layers.Dense(channel_in // ratio, activation='relu', name='shared_layer_one',
                                             kernel_initializer='he_normal', bias_initializer='zeros')
    shared_layer_two = tf.keras.layers.Dense(channel_in, name='shared_layer_two',
                                             kernel_initializer='he_normal', bias_initializer='zeros')
    avg_pool = tf.reduce_mean(input_feature, axis=[1, 2], keepdims=True)
    avg_pool = shared_layer_one(avg_pool)
    avg_pool = shared_layer_two(avg_pool)
    max_pool = tf.reduce_max(input_feature, axis=[1, 2], keepdims=True)
    max_pool = shared_layer_one(max_pool)
    max_pool = shared_layer_two(max_pool)
    scale = tf.sigmoid(avg_pool + max_pool)
    return input_feature * scale


def spatial_attention(input_feature):
    kernel_size = 7
    avg_pool = tf.reduce_mean(input_feature, axis=3, keepdims=True)
    max_pool = tf.reduce_max(input_feature, axis=3, keepdims=True)
    concat = tf.concat([avg_pool, max_pool], 3)
    conv = tf.keras.layers.Conv2D(filters=1, kernel_size=kernel_size, strides=1, padding='same', activation='sigmoid',
                            kernel_initializer='he_normal', use_bias=False)(concat)
    return input_feature * conv
