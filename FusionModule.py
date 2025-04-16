 """
 This file is the fusion module in "An enhanced day-night feature fusion method for fine-grained urban functional 
zone mapping from the SDGSAT-1 imagery"
 """
import tensorflow as tf
from tensorflow.keras import layers


class FusionModule(tf.keras.layers.Layer):
    def __init__(self, dropout_rate=0.1, **kwargs):
        super(FusionModule, self).__init__(**kwargs)
        self.conv_sigmoid = layers.Conv2D(filters=1, kernel_size=1, activation='sigmoid')
        self.spatial_conv = layers.Conv2D(filters=1, kernel_size=7, activation='sigmoid', padding='same')
        self.conv_relu_1x1 = layers.Conv2D(filters=1, kernel_size=1, activation='relu')
        self.dilated_conv_relu_1x1 = layers.Conv2D(filters=1, kernel_size=1, activation='relu')
        self.fusion_conv_1x1 = layers.Conv2D(filters=1, kernel_size=1, activation='relu')
        self.conv_relu_3x3 = layers.Conv2D(filters=1, kernel_size=3, activation='relu', padding='same')
        self.dense_layer = layers.Dense(units=1, activation='relu')
        self.dropout_rate = dropout_rate

        self.downsample_conv = layers.Conv2D(1, 1, strides=2, activation='relu', padding='same')
        self.depthwise_conv = [
            layers.DepthwiseConv2D(3, dilation_rate=i, activation='relu', padding='same')
            for i in range(1, 4)
        ]
        self.pointwise_conv = layers.Conv2D(1, 1, activation=None, padding='same')
        self.upsampling = layers.UpSampling2D(size=(2, 2))

    def cbr1(self, inputs):
        """1×1 卷积 + ReLU"""
        return self.conv_relu_1x1(inputs)

    def cbr2(self, inputs):
        """3×3 卷积 + ReLU"""
        return self.conv_relu_3x3(inputs)

    def channel_attention(self, inputs):
        """通道注意力机制"""
        channel_dim = inputs.shape[-1]
        shared_fc1 = layers.Dense(channel_dim // 8, activation='relu')
        shared_fc2 = layers.Dense(channel_dim)
        avg_pool = tf.reduce_mean(inputs, axis=[1, 2], keepdims=True)
        avg_pool = shared_fc1(avg_pool)
        avg_pool = shared_fc2(avg_pool)
        max_pool = tf.reduce_max(inputs, axis=[1, 2], keepdims=True)
        max_pool = shared_fc1(max_pool)
        max_pool = shared_fc2(max_pool)
        scale = tf.sigmoid(avg_pool + max_pool)
        return inputs * scale

    def spatial_attention(self, inputs):
        """空间注意力机制"""
        avg_pool = tf.reduce_mean(inputs, axis=3, keepdims=True)
        max_pool = tf.reduce_max(inputs, axis=3, keepdims=True)
        concat = tf.concat([avg_pool, max_pool], axis=3)
        spatial_att = self.spatial_conv(concat)
        return inputs * spatial_att

    def dilated_convolution(self, inputs):
        """空洞卷积模块"""
        downsampled = self.downsample_conv(inputs)
        downsampled = layers.Dropout(self.dropout_rate)(downsampled)
        conv_outputs = []
        for dw_conv in self.depthwise_conv:
            x = dw_conv(downsampled)
            x = self.pointwise_conv(x)
            x = layers.Dropout(self.dropout_rate)(x)
            conv_outputs.append(x)
        upsampled = self.upsampling(tf.concat(conv_outputs, axis=-1))
        upsampled = self.dilated_conv_relu_1x1(upsampled)
        return tf.concat([inputs, upsampled], axis=-1)

    def DBI(self, x1, x2):
        """双重双向交互模块"""
        x1 = self.channel_attention(x1)
        x2_processed = self.cbr2(x2)
        g = self.conv_sigmoid(x2_processed)
        x1 = x1 * g
        return self.spatial_attention(x1)

    def TMA(self, x2, x1, x3):
        """三模态聚合模块"""
        x2 = self.cbr2(x2)
        x3 = self.cbr2(x3)
        channel_avg = tf.reduce_mean(x1, axis=[1, 2], keepdims=True)
        channel_avg = self.dense_layer(channel_avg)
        x1 = x1 + channel_avg
        x123 = tf.concat([x2, x1, x3], axis=-1)
        x123_cbr = self.cbr1(x123)
        return x123 + x123_cbr

    def call(self, inputs):
        """主融合逻辑"""
        msi, gli, bdh, poi, dsm = inputs
        # 各输入拼接
        x1_mii = tf.concat([msi, bdh, dsm], axis=-1)
        x2_giu = tf.concat([gli, bdh, dsm], axis=-1)
        x3_poi = tf.concat([poi, bdh, dsm], axis=-1)

        # 注意力交互
        x23 = self.DBI(x2_giu, x3_poi) + x2_giu
        x32 = self.DBI(x3_poi, x2_giu) + x3_poi

        # 多特征融合
        x123 = self.TMA(x2_giu, x1_mii, x3_poi)
        x123_di = self.dilated_convolution(x123)
        x123_di = self.fusion_conv_1x1(x123_di)

        # 门控融合
        G1 = self.conv_relu_3x3(x23)
        G2 = self.conv_relu_3x3(x32)
        G_sum = G1 + G2 + 1e-8  # 防止除0
        G1_norm = G1 / G_sum
        G2_norm = G2 / G_sum
        out = tf.concat([(1 + G1_norm) * x123_di, G2_norm * (x23 + x32)], axis=-1)
        return out


# 使用示例
if __name__ == "__main__":
    # 创建测试输入（通过特征提取模块后，各输入的空间尺寸相同）
    msi = tf.random.normal([8, 16, 16, 64])  # (batch, h, w, channels)
    gli = tf.random.normal([8, 16, 16, 64])
    bdh = tf.random.normal([8, 16, 16, 64])
    poi = tf.random.normal([8, 16, 16, 64])
    dsm = tf.random.normal([8, 16, 16, 64])

    fusion = FusionModule(dropout_rate=0.2)
    fused_features = fusion([msi, gli, bdh, poi, dsm])
    print("融合输出形状：", fused_features.shape)
