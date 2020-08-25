import math
import tensorflow as tf

inputs_shape = (224, 224, 3)
NUM_CLASSES = 1000


# densenet
class Densenet(object):
    @staticmethod
    def densenet_bottleneck(inputs, growth_rate, drop_rate, training=None, **kwargs):
        x = tf.keras.layers.BatchNormalization()(inputs, training=training)
        x = tf.nn.relu(x)
        x = tf.keras.layers.Conv2D(filters=4 * growth_rate,
                                   kernel_size=(1, 1),
                                   strides=1,
                                   padding="same")(x)
        x = tf.keras.layers.BatchNormalization()(x, training=training)
        x = tf.nn.relu(x)
        x = tf.keras.layers.Conv2D(filters=growth_rate,
                                   kernel_size=(3, 3),
                                   strides=1,
                                   padding="same")(x)
        x = tf.keras.layers.Dropout(rate=drop_rate)(x)
        return x

    @staticmethod
    def densenet_denseblock(inputs, num_layers, growth_rate, drop_rate, training=None, **kwargs):
        features_list = []
        features_list.append(inputs)
        x = inputs
        for _ in range(num_layers):
            y = Densenet.densenet_bottleneck(x, growth_rate=growth_rate, drop_rate=drop_rate, training=training)
            features_list.append(y)
            x = tf.concat(features_list, axis=-1)
        features_list.clear()
        return x

    @staticmethod
    def densenet_transitionlayer(inputs, out_channels, training=None, **kwargs):
        x = tf.keras.layers.BatchNormalization()(inputs, training=training)
        x = tf.nn.relu(x)
        x = tf.keras.layers.Conv2D(filters=out_channels,
                                   kernel_size=(1, 1),
                                   strides=1,
                                   padding="same")(x)
        x = tf.keras.layers.MaxPool2D(pool_size=(2, 2),
                                      strides=2,
                                      padding="same")(x)
        return x

    @staticmethod
    def Densenet(num_init_features, growth_rate, block_layers, compression_rate, drop_rate, training=None, mask=None):
        inputs = tf.keras.layers.Input(shape=inputs_shape)
        x = tf.keras.layers.Conv2D(filters=num_init_features,
                                   kernel_size=(7, 7),
                                   strides=2,
                                   padding="same")(inputs)
        x = tf.keras.layers.BatchNormalization()(x, training=training)
        x = tf.nn.relu(x)
        x = tf.keras.layers.MaxPool2D(pool_size=(3, 3),
                                      strides=2,
                                      padding="same")(x)
        num_channels = num_init_features
        x = Densenet.densenet_denseblock(x, num_layers=block_layers[0], growth_rate=growth_rate, drop_rate=drop_rate)
        num_channels += growth_rate * block_layers[0]
        num_channels = compression_rate * num_channels
        x = Densenet.densenet_transitionlayer(x, out_channels=int(num_channels))
        x = Densenet.densenet_denseblock(x, num_layers=block_layers[1], growth_rate=growth_rate, drop_rate=drop_rate)
        num_channels += growth_rate * block_layers[1]
        num_channels = compression_rate * num_channels
        x = Densenet.densenet_transitionlayer(x, out_channels=int(num_channels))
        x = Densenet.densenet_denseblock(x, num_layers=block_layers[2], growth_rate=growth_rate, drop_rate=drop_rate)
        num_channels += growth_rate * block_layers[2]
        num_channels = compression_rate * num_channels
        x = Densenet.densenet_transitionlayer(x, out_channels=int(num_channels))
        x = Densenet.densenet_denseblock(x, num_layers=block_layers[3], growth_rate=growth_rate, drop_rate=drop_rate)
        x = tf.keras.layers.GlobalAveragePooling2D()(x)
        outputs = tf.keras.layers.Dense(units=NUM_CLASSES, activation=tf.keras.activations.softmax)(x)
        model = tf.keras.Model(inputs=inputs, outputs=outputs)
        return model


# efficientnet
class Efficientnet(object):

    @staticmethod
    def round_filters(filters, multiplier):
        depth_divisor = 8
        min_depth = None
        min_depth = min_depth or depth_divisor
        filters = filters * multiplier
        new_filters = max(min_depth, int(filters + depth_divisor / 2) // depth_divisor * depth_divisor)
        if new_filters < 0.9 * filters:
            new_filters += depth_divisor
        return int(new_filters)

    @staticmethod
    def round_repeats(repeats, multiplier):
        if not multiplier:
            return repeats
        return int(math.ceil(multiplier * repeats))

    @staticmethod
    def efficientnet_seblock(inputs, input_channels, ratio=0.25, **kwargs):
        num_reduced_filters = max(1, int(input_channels * ratio))
        x = tf.keras.layers.GlobalAveragePooling2D()(inputs)
        x = tf.expand_dims(input=x, axis=1)
        x = tf.expand_dims(input=x, axis=1)
        x = tf.keras.layers.Conv2D(filters=num_reduced_filters, kernel_size=(1, 1), strides=1, padding='same')(x)
        x = tf.nn.swish(x)
        x = tf.keras.layers.Conv2D(filters=input_channels, kernel_size=(1, 1), strides=1, padding='same')(x)
        x = tf.nn.sigmoid(x)
        x = inputs * x
        return x

    @staticmethod
    def efficientnet_mbconv(inputs, in_channels, out_channels, expansion_factor, stride, k, drop_connect_rate,
                            training=None, **kwargs):
        x = tf.keras.layers.Conv2D(filters=in_channels * expansion_factor,
                                   kernel_size=(1, 1),
                                   strides=1,
                                   padding="same",
                                   use_bias=False)(inputs)
        x = tf.keras.layers.BatchNormalization()(x, training=training)
        x = tf.nn.swish(x)
        x = tf.keras.layers.DepthwiseConv2D(kernel_size=(k, k),
                                            strides=stride,
                                            padding="same",
                                            use_bias=False)(x)
        x = tf.keras.layers.BatchNormalization()(x, training=training)
        x = Efficientnet.efficientnet_seblock(x, input_channels=in_channels * expansion_factor)
        x = tf.keras.layers.Conv2D(filters=in_channels * expansion_factor,
                                   kernel_size=(1, 1),
                                   strides=1,
                                   padding="same",
                                   use_bias=False)(x)
        x = tf.keras.layers.BatchNormalization()(x, training=training)
        if stride == 1 and in_channels == out_channels:
            if drop_connect_rate:
                x = tf.keras.layers.Dropout(rate=drop_connect_rate)(x)
            x = tf.keras.layers.concatenate([x, inputs])
        return x

    @staticmethod
    def efficientnet_build_mbconv_block(x, in_channels, out_channels, layers, stride, expansion_factor, k,
                                        drop_connect_rate):
        for i in range(layers):
            if i == 0:
                x = Efficientnet.efficientnet_mbconv(x, in_channels=in_channels,
                                                     out_channels=out_channels,
                                                     expansion_factor=expansion_factor,
                                                     stride=stride,
                                                     k=k,
                                                     drop_connect_rate=drop_connect_rate)
            else:
                x = Efficientnet.efficientnet_mbconv(x, in_channels=out_channels,
                                                     out_channels=out_channels,
                                                     expansion_factor=expansion_factor,
                                                     stride=1,
                                                     k=k,
                                                     drop_connect_rate=drop_connect_rate)
        return x

    @staticmethod
    def Efficientnet(width_coefficient, depth_coefficient, dropout_rate, drop_connect_rate=0.2, training=None,
                     mask=None):
        inputs = tf.keras.layers.Input(shape=inputs_shape)
        x = tf.keras.layers.Conv2D(filters=Efficientnet.round_filters(32, width_coefficient),
                                   kernel_size=(3, 3),
                                   strides=2,
                                   padding="same",
                                   use_bias=False)(inputs)
        x = tf.keras.layers.BatchNormalization()(x, training=training)
        x = tf.nn.swish(x)
        x = Efficientnet.efficientnet_build_mbconv_block(x,
                                                         in_channels=Efficientnet.round_filters(32, width_coefficient),
                                                         out_channels=Efficientnet.round_filters(16, width_coefficient),
                                                         layers=Efficientnet.round_repeats(1, depth_coefficient),
                                                         stride=1,
                                                         expansion_factor=1, k=3, drop_connect_rate=drop_connect_rate)
        x = Efficientnet.efficientnet_build_mbconv_block(x,
                                                         in_channels=Efficientnet.round_filters(16, width_coefficient),
                                                         out_channels=Efficientnet.round_filters(24, width_coefficient),
                                                         layers=Efficientnet.round_repeats(2, depth_coefficient),
                                                         stride=2,
                                                         expansion_factor=6, k=3, drop_connect_rate=drop_connect_rate)
        x = Efficientnet.efficientnet_build_mbconv_block(x,
                                                         in_channels=Efficientnet.round_filters(24, width_coefficient),
                                                         out_channels=Efficientnet.round_filters(40, width_coefficient),
                                                         layers=Efficientnet.round_repeats(2, depth_coefficient),
                                                         stride=2,
                                                         expansion_factor=6, k=5, drop_connect_rate=drop_connect_rate)
        x = Efficientnet.efficientnet_build_mbconv_block(x,
                                                         in_channels=Efficientnet.round_filters(40, width_coefficient),
                                                         out_channels=Efficientnet.round_filters(80, width_coefficient),
                                                         layers=Efficientnet.round_repeats(3, depth_coefficient),
                                                         stride=2,
                                                         expansion_factor=6, k=3, drop_connect_rate=drop_connect_rate)
        x = Efficientnet.efficientnet_build_mbconv_block(x,
                                                         in_channels=Efficientnet.round_filters(80, width_coefficient),
                                                         out_channels=Efficientnet.round_filters(112,
                                                                                                 width_coefficient),
                                                         layers=Efficientnet.round_repeats(3, depth_coefficient),
                                                         stride=1,
                                                         expansion_factor=6, k=5, drop_connect_rate=drop_connect_rate)
        x = Efficientnet.efficientnet_build_mbconv_block(x,
                                                         in_channels=Efficientnet.round_filters(112, width_coefficient),
                                                         out_channels=Efficientnet.round_filters(192,
                                                                                                 width_coefficient),
                                                         layers=Efficientnet.round_repeats(4, depth_coefficient),
                                                         stride=2,
                                                         expansion_factor=6, k=5, drop_connect_rate=drop_connect_rate)
        x = Efficientnet.efficientnet_build_mbconv_block(x,
                                                         in_channels=Efficientnet.round_filters(192, width_coefficient),
                                                         out_channels=Efficientnet.round_filters(320,
                                                                                                 width_coefficient),
                                                         layers=Efficientnet.round_repeats(1, depth_coefficient),
                                                         stride=1,
                                                         expansion_factor=6, k=3, drop_connect_rate=drop_connect_rate)
        x = tf.keras.layers.Conv2D(filters=Efficientnet.round_filters(1280, width_coefficient),
                                   kernel_size=(1, 1),
                                   strides=1,
                                   padding="same",
                                   use_bias=False)(x)
        x = tf.keras.layers.BatchNormalization()(x, training=training)
        x = tf.nn.swish(x)
        x = tf.keras.layers.GlobalAveragePooling2D()(x)
        x = tf.keras.layers.Dropout(rate=dropout_rate)(x)
        outputs = tf.keras.layers.Dense(units=NUM_CLASSES, activation=tf.keras.activations.softmax)(x)
        model = tf.keras.Model(inputs=inputs, outputs=outputs)
        return model


# mobilenet
class Mobilenet(object):
    @staticmethod
    def bottleneck(inputs, input_channels, output_channels, expansion_factor, stride, training=None, **kwargs):
        x = tf.keras.layers.Conv2D(filters=input_channels * expansion_factor,
                                   kernel_size=(1, 1),
                                   strides=1,
                                   padding="same")(inputs)
        x = tf.keras.layers.BatchNormalization()(x, training=training)
        x = tf.nn.relu6(x)
        x = tf.keras.layers.DepthwiseConv2D(kernel_size=(3, 3),
                                            strides=stride,
                                            padding="same")(x)
        x = tf.keras.layers.BatchNormalization()(x, training=training)
        x = tf.nn.relu6(x)
        x = tf.keras.layers.Conv2D(filters=output_channels,
                                   kernel_size=(1, 1),
                                   strides=1,
                                   padding="same")(x)
        x = tf.keras.layers.BatchNormalization()(x, training=training)
        x = tf.keras.layers.Activation(tf.keras.activations.linear)(x)
        if stride == 1 and input_channels == output_channels:
            x = tf.keras.layers.concatenate([x, inputs])
        return x

    @staticmethod
    def build_bottleneck(inputs, t, in_channel_num, out_channel_num, n, s):
        bottleneck = inputs
        for i in range(n):
            if i == 0:
                bottleneck = Mobilenet.bottleneck(inputs, input_channels=in_channel_num,
                                                  output_channels=out_channel_num,
                                                  expansion_factor=t,
                                                  stride=s)
            else:
                bottleneck = Mobilenet.bottleneck(inputs, input_channels=out_channel_num,
                                                  output_channels=out_channel_num,
                                                  expansion_factor=t,
                                                  stride=1)
        return bottleneck

    @staticmethod
    def h_sigmoid(x):
        return tf.nn.relu6(x + 3) / 6

    @staticmethod
    def seblock(inputs, input_channels, r=16, **kwargs):
        x = tf.keras.layers.GlobalAveragePooling2D()(inputs)
        x = tf.keras.layers.Dense(units=input_channels // r)(x)
        x = tf.nn.relu(x)
        x = tf.keras.layers.Dense(units=input_channels)(x)
        x = Mobilenet.h_sigmoid(x)
        x = tf.expand_dims(x, axis=1)
        x = tf.expand_dims(x, axis=1)
        output = inputs * x
        return output

    @staticmethod
    def BottleNeck(inputs, in_size, exp_size, out_size, s, is_se_existing, NL, k, training=None, **kwargs):
        x = tf.keras.layers.Conv2D(filters=exp_size,
                                   kernel_size=(1, 1),
                                   strides=1,
                                   padding="same")(inputs)
        x = tf.keras.layers.BatchNormalization()(x, training=training)
        if NL == 'HS':
            x = Mobilenet.h_swish(x)
        elif NL == 'RE':
            x = tf.nn.relu6(x)
        x = tf.keras.layers.DepthwiseConv2D(kernel_size=(k, k),
                                            strides=s,
                                            padding="same")(x)
        x = tf.keras.layers.BatchNormalization()(x, training=training)
        if NL == 'HS':
            x = Mobilenet.h_swish(x)
        elif NL == 'RE':
            x = tf.nn.relu6(x)
        if is_se_existing:
            x = Mobilenet.seblock(x, input_channels=exp_size)
        x = tf.keras.layers.Conv2D(filters=out_size,
                                   kernel_size=(1, 1),
                                   strides=1,
                                   padding="same")(x)
        x = tf.keras.layers.BatchNormalization()(x, training=training)
        x = tf.keras.layers.Activation(tf.keras.activations.linear)(x)
        if s == 1 and in_size == out_size:
            x = tf.keras.layers.add([x, inputs])
        return x

    @staticmethod
    def h_swish(x):
        return x * Mobilenet.h_sigmoid(x)

    @staticmethod
    def MobileNetV1(training=None, mask=None):
        inputs = tf.keras.layers.Input(shape=inputs_shape)
        x = tf.keras.layers.Conv2D(filters=32,
                                   kernel_size=(3, 3),
                                   strides=2,
                                   padding="same")(inputs)
        x = tf.keras.layers.SeparableConv2D(filters=64,
                                            kernel_size=(3, 3),
                                            strides=1,
                                            padding="same")(x)
        x = tf.keras.layers.SeparableConv2D(filters=128,
                                            kernel_size=(3, 3),
                                            strides=2,
                                            padding="same")(x)
        x = tf.keras.layers.SeparableConv2D(filters=128,
                                            kernel_size=(3, 3),
                                            strides=1,
                                            padding="same")(x)
        x = tf.keras.layers.SeparableConv2D(filters=256,
                                            kernel_size=(3, 3),
                                            strides=2,
                                            padding="same")(x)
        x = tf.keras.layers.SeparableConv2D(filters=256,
                                            kernel_size=(3, 3),
                                            strides=1,
                                            padding="same")(x)
        x = tf.keras.layers.SeparableConv2D(filters=512,
                                            kernel_size=(3, 3),
                                            strides=2,
                                            padding="same")(x)
        x = tf.keras.layers.SeparableConv2D(filters=512,
                                            kernel_size=(3, 3),
                                            strides=1,
                                            padding="same")(x)
        x = tf.keras.layers.SeparableConv2D(filters=512,
                                            kernel_size=(3, 3),
                                            strides=1,
                                            padding="same")(x)
        x = tf.keras.layers.SeparableConv2D(filters=512,
                                            kernel_size=(3, 3),
                                            strides=1,
                                            padding="same")(x)
        x = tf.keras.layers.SeparableConv2D(filters=512,
                                            kernel_size=(3, 3),
                                            strides=1,
                                            padding="same")(x)
        x = tf.keras.layers.SeparableConv2D(filters=512,
                                            kernel_size=(3, 3),
                                            strides=1,
                                            padding="same")(x)
        x = tf.keras.layers.SeparableConv2D(filters=1024,
                                            kernel_size=(3, 3),
                                            strides=2,
                                            padding="same")(x)
        x = tf.keras.layers.SeparableConv2D(filters=1024,
                                            kernel_size=(3, 3),
                                            strides=1,
                                            padding="same")(x)
        x = tf.keras.layers.AveragePooling2D(pool_size=(7, 7),
                                             strides=1)(x)
        outputs = tf.keras.layers.Dense(units=NUM_CLASSES,
                                        activation=tf.keras.activations.softmax)(x)
        model = tf.keras.Model(inputs=inputs, outputs=outputs)
        return model

    @staticmethod
    def MobileNetV2(training=None, mask=None):
        inputs = tf.keras.layers.Input(shape=inputs_shape)
        x = tf.keras.layers.Conv2D(filters=32,
                                   kernel_size=(3, 3),
                                   strides=2,
                                   padding="same")(inputs)
        x = Mobilenet.build_bottleneck(x, t=1,
                                       in_channel_num=32,
                                       out_channel_num=16,
                                       n=1,
                                       s=1)
        x = Mobilenet.build_bottleneck(x, t=6,
                                       in_channel_num=16,
                                       out_channel_num=24,
                                       n=2,
                                       s=2)
        x = Mobilenet.build_bottleneck(x, t=6,
                                       in_channel_num=24,
                                       out_channel_num=32,
                                       n=3,
                                       s=2)
        x = Mobilenet.build_bottleneck(x, t=6,
                                       in_channel_num=32,
                                       out_channel_num=64,
                                       n=4,
                                       s=2)
        x = Mobilenet.build_bottleneck(x, t=6,
                                       in_channel_num=64,
                                       out_channel_num=96,
                                       n=3,
                                       s=1)
        x = Mobilenet.build_bottleneck(x, t=6,
                                       in_channel_num=96,
                                       out_channel_num=160,
                                       n=3,
                                       s=2)
        x = Mobilenet.build_bottleneck(x, t=6,
                                       in_channel_num=160,
                                       out_channel_num=320,
                                       n=1,
                                       s=1)
        x = tf.keras.layers.Conv2D(filters=1280,
                                   kernel_size=(1, 1),
                                   strides=1,
                                   padding="same")(x)
        x = tf.keras.layers.AveragePooling2D(pool_size=(7, 7))(x)
        outputs = tf.keras.layers.Conv2D(filters=NUM_CLASSES,
                                         kernel_size=(1, 1),
                                         strides=1,
                                         padding="same",
                                         activation=tf.keras.activations.softmax)(x)
        model = tf.keras.Model(inputs=inputs, outputs=outputs)
        return model

    @staticmethod
    def MobileNetV3Large(training=None, mask=None):
        inputs = tf.keras.layers.Input(shape=inputs_shape)
        x = tf.keras.layers.Conv2D(filters=16,
                                   kernel_size=(3, 3),
                                   strides=2,
                                   padding="same")(inputs)
        x = tf.keras.layers.BatchNormalization()(x, training=training)
        x = Mobilenet.h_swish(x)
        x = Mobilenet.BottleNeck(x, in_size=16, exp_size=16, out_size=16, s=1, is_se_existing=False, NL="RE", k=3,
                                 training=training)
        x = Mobilenet.BottleNeck(x, in_size=16, exp_size=64, out_size=24, s=2, is_se_existing=False, NL="RE", k=3,
                                 training=training)
        x = Mobilenet.BottleNeck(x, in_size=24, exp_size=72, out_size=24, s=1, is_se_existing=False, NL="RE", k=3,
                                 training=training)
        x = Mobilenet.BottleNeck(x, in_size=24, exp_size=72, out_size=40, s=2, is_se_existing=True, NL="RE", k=5,
                                 training=training)
        x = Mobilenet.BottleNeck(x, in_size=40, exp_size=120, out_size=40, s=1, is_se_existing=True, NL="RE", k=5,
                                 training=training)
        x = Mobilenet.BottleNeck(x, in_size=40, exp_size=120, out_size=40, s=1, is_se_existing=True, NL="RE", k=5,
                                 training=training)
        x = Mobilenet.BottleNeck(x, in_size=40, exp_size=240, out_size=80, s=2, is_se_existing=False, NL="HS", k=3,
                                 training=training)
        x = Mobilenet.BottleNeck(x, in_size=80, exp_size=200, out_size=80, s=1, is_se_existing=False, NL="HS", k=3,
                                 training=training)
        x = Mobilenet.BottleNeck(x, in_size=80, exp_size=184, out_size=80, s=1, is_se_existing=False, NL="HS", k=3,
                                 training=training)
        x = Mobilenet.BottleNeck(x, in_size=80, exp_size=184, out_size=80, s=1, is_se_existing=False, NL="HS", k=3,
                                 training=training)
        x = Mobilenet.BottleNeck(x, in_size=80, exp_size=480, out_size=112, s=1, is_se_existing=True, NL="HS", k=3,
                                 training=training)
        x = Mobilenet.BottleNeck(x, in_size=112, exp_size=672, out_size=112, s=1, is_se_existing=True, NL="HS", k=3,
                                 training=training)
        x = Mobilenet.BottleNeck(x, in_size=112, exp_size=672, out_size=160, s=2, is_se_existing=True, NL="HS", k=5,
                                 training=training)
        x = Mobilenet.BottleNeck(x, in_size=160, exp_size=960, out_size=160, s=1, is_se_existing=True, NL="HS", k=5,
                                 training=training)
        x = Mobilenet.BottleNeck(x, in_size=160, exp_size=960, out_size=160, s=1, is_se_existing=True, NL="HS", k=5,
                                 training=training)
        x = tf.keras.layers.Conv2D(filters=960,
                                   kernel_size=(1, 1),
                                   strides=1,
                                   padding="same")(x)
        x = tf.keras.layers.BatchNormalization()(x, training=training)
        x = Mobilenet.h_swish(x)
        x = tf.keras.layers.AveragePooling2D(pool_size=(7, 7),
                                             strides=1)(x)
        x = tf.keras.layers.Conv2D(filters=1280,
                                   kernel_size=(1, 1),
                                   strides=1,
                                   padding="same")(x)
        x = Mobilenet.h_swish(x)
        outputs = tf.keras.layers.Conv2D(filters=NUM_CLASSES,
                                         kernel_size=(1, 1),
                                         strides=1,
                                         padding="same",
                                         activation=tf.keras.activations.softmax)(x)
        model = tf.keras.Model(inputs=inputs, outputs=outputs)
        return model

    @staticmethod
    def MobileNetV3Small(training=None, mask=None):
        inputs = tf.keras.layers.Input(shape=inputs_shape)
        x = tf.keras.layers.Conv2D(filters=16,
                                   kernel_size=(3, 3),
                                   strides=2,
                                   padding="same")(inputs)
        x = tf.keras.layers.BatchNormalization()(x, training=training)
        x = Mobilenet.h_swish(x)
        x = Mobilenet.BottleNeck(x, in_size=16, exp_size=16, out_size=16, s=2, is_se_existing=True, NL="RE", k=3)
        x = Mobilenet.BottleNeck(x, in_size=16, exp_size=72, out_size=24, s=2, is_se_existing=False, NL="RE", k=3)
        x = Mobilenet.BottleNeck(x, in_size=24, exp_size=88, out_size=24, s=1, is_se_existing=False, NL="RE", k=3)
        x = Mobilenet.BottleNeck(x, in_size=24, exp_size=96, out_size=40, s=2, is_se_existing=True, NL="HS", k=5)
        x = Mobilenet.BottleNeck(x, in_size=40, exp_size=240, out_size=40, s=1, is_se_existing=True, NL="HS", k=5)
        x = Mobilenet.BottleNeck(x, in_size=40, exp_size=240, out_size=40, s=1, is_se_existing=True, NL="HS", k=5)
        x = Mobilenet.BottleNeck(x, in_size=40, exp_size=120, out_size=48, s=1, is_se_existing=True, NL="HS", k=5)
        x = Mobilenet.BottleNeck(x, in_size=48, exp_size=144, out_size=48, s=1, is_se_existing=True, NL="HS", k=5)
        x = Mobilenet.BottleNeck(x, in_size=48, exp_size=288, out_size=96, s=2, is_se_existing=True, NL="HS", k=5)
        x = Mobilenet.BottleNeck(x, in_size=96, exp_size=576, out_size=96, s=1, is_se_existing=True, NL="HS", k=5)
        x = Mobilenet.BottleNeck(x, in_size=96, exp_size=576, out_size=96, s=1, is_se_existing=True, NL="HS", k=5)
        x = tf.keras.layers.Conv2D(filters=576,
                                   kernel_size=(1, 1),
                                   strides=1,
                                   padding="same")(x)
        x = tf.keras.layers.BatchNormalization()(x, training=training)
        x = Mobilenet.h_swish(x)
        x = tf.keras.layers.AveragePooling2D(pool_size=(7, 7),
                                             strides=1)(x)
        x = tf.keras.layers.Conv2D(filters=1280,
                                   kernel_size=(1, 1),
                                   strides=1,
                                   padding="same")(x)
        outputs = tf.keras.layers.Conv2D(filters=NUM_CLASSES,
                                         kernel_size=(1, 1),
                                         strides=1,
                                         padding="same",
                                         activation=tf.keras.activations.softmax)(x)
        model = tf.keras.Model(inputs=inputs, outputs=outputs)
        return model


# resnext
class ResNeXt(object):
    @staticmethod
    def BasicBlock(inputs, filter_num, stride=1, training=None, **kwargs):
        if stride != 1:
            residual = tf.keras.layers.Conv2D(filters=filter_num,
                                              kernel_size=(1, 1),
                                              strides=stride)(inputs)
            residual = tf.keras.layers.BatchNormalization()(residual, training=training)
        else:
            residual = inputs

        x = tf.keras.layers.Conv2D(filters=filter_num,
                                   kernel_size=(3, 3),
                                   strides=stride,
                                   padding="same")(inputs)
        x = tf.keras.layers.BatchNormalization()(x, training=training)
        x = tf.nn.relu(x)
        x = tf.keras.layers.Conv2D(filters=filter_num,
                                   kernel_size=(3, 3),
                                   strides=1,
                                   padding="same")(x)
        x = tf.keras.layers.BatchNormalization()(x, training=training)
        x = tf.nn.relu(x)
        output = tf.keras.layers.concatenate([residual, x])
        return output

    @staticmethod
    def BottleNeck(inputs, filter_num, stride=1, training=None, **kwargs):
        residual = tf.keras.layers.Conv2D(filters=filter_num * 4,
                                          kernel_size=(1, 1),
                                          strides=stride)(inputs)
        residual = tf.keras.layers.BatchNormalization()(residual, training=training)
        x = tf.keras.layers.Conv2D(filters=filter_num,
                                   kernel_size=(1, 1),
                                   strides=1,
                                   padding='same')(inputs)
        x = tf.keras.layers.BatchNormalization()(x, training=training)
        x = tf.nn.relu(x)
        x = tf.keras.layers.Conv2D(filters=filter_num,
                                   kernel_size=(3, 3),
                                   strides=stride,
                                   padding='same')(x)
        x = tf.keras.layers.BatchNormalization()(x, training=training)
        x = tf.nn.relu(x)
        x = tf.keras.layers.Conv2D(filters=filter_num * 4,
                                   kernel_size=(1, 1),
                                   strides=1,
                                   padding='same')(x)
        x = tf.keras.layers.BatchNormalization()(x, training=training)
        return tf.nn.relu(tf.keras.layers.add([residual, x]))

    @staticmethod
    def make_basic_block_layer(inputs, filter_num, blocks, stride=1, training=None, mask=None):
        res_block = ResNeXt.BasicBlock(inputs, filter_num, stride=stride)
        for _ in range(1, blocks):
            res_block = ResNeXt.BasicBlock(inputs, filter_num, stride=1)
        return res_block

    @staticmethod
    def make_bottleneck_layer(inputs, filter_num, blocks, stride=1, training=None, mask=None):
        res_block = ResNeXt.BottleNeck(inputs, filter_num, stride=stride)
        for _ in range(1, blocks):
            res_block = ResNeXt.BottleNeck(inputs, filter_num, stride=1)
        return res_block

    @staticmethod
    def ResNeXt_BottleNeck(inputs, filters, strides, groups, training=None, **kwargs):
        x = tf.keras.layers.Conv2D(filters=filters,
                                   kernel_size=(1, 1),
                                   strides=1,
                                   padding="same")(inputs)
        x = tf.keras.layers.BatchNormalization()(x, training=training)
        x = tf.keras.layers.Conv2D(filters=filters,
                                   kernel_size=(3, 3),
                                   strides=strides,
                                   padding="same", )(x)
        x = tf.keras.layers.BatchNormalization()(x, training=training)
        x = tf.nn.relu(x)
        x = tf.keras.layers.Conv2D(filters=2 * filters,
                                   kernel_size=(1, 1),
                                   strides=1,
                                   padding="same")(x)
        x = tf.keras.layers.BatchNormalization()(x, training=training)
        shortcut = tf.keras.layers.Conv2D(filters=2 * filters,
                                          kernel_size=(1, 1),
                                          strides=strides,
                                          padding="same")(inputs)
        shortcut = tf.keras.layers.BatchNormalization()(shortcut, training=training)
        output = tf.nn.relu(tf.keras.layers.add([x, shortcut]))
        return output

    @staticmethod
    def build_ResNeXt_block(inputs, filters, strides, groups, repeat_num):
        block = ResNeXt.ResNeXt_BottleNeck(inputs, filters=filters,
                                           strides=strides,
                                           groups=groups)
        for _ in range(1, repeat_num):
            block = ResNeXt.ResNeXt_BottleNeck(inputs, filters=filters,
                                               strides=1,
                                               groups=groups)
        return block

    @staticmethod
    def ResNetTypeI(layer_params, training=None, mask=None):
        inputs = tf.keras.layers.Input(shape=inputs_shape)
        x = tf.keras.layers.Conv2D(filters=64,
                                   kernel_size=(7, 7),
                                   strides=2,
                                   padding="same")(inputs)
        x = tf.keras.layers.BatchNormalization()(x, training=training)
        x = tf.nn.relu(x)
        x = tf.keras.layers.MaxPool2D(pool_size=(3, 3),
                                      strides=2,
                                      padding="same")(x)
        x = ResNeXt.make_basic_block_layer(x, filter_num=64,
                                           blocks=layer_params[0])
        x = ResNeXt.make_basic_block_layer(x, filter_num=128,
                                           blocks=layer_params[1],
                                           stride=2)
        x = ResNeXt.make_basic_block_layer(x, filter_num=256,
                                           blocks=layer_params[2],
                                           stride=2)
        x = ResNeXt.make_basic_block_layer(x, filter_num=512,
                                           blocks=layer_params[3],
                                           stride=2)
        x = tf.keras.layers.GlobalAveragePooling2D()(x)
        outputs = tf.keras.layers.Dense(units=NUM_CLASSES, activation=tf.keras.activations.softmax)(x)
        model = tf.keras.Model(inputs=inputs, outputs=outputs)
        return model

    @staticmethod
    def ResNetTypeII(layer_params, training=None, mask=None):
        inputs = tf.keras.layers.Input(shape=inputs_shape)
        x = tf.keras.layers.Conv2D(filters=64,
                                   kernel_size=(7, 7),
                                   strides=2,
                                   padding="same")(inputs)
        x = tf.keras.layers.BatchNormalization()(x, training)
        x = tf.nn.relu(x)
        x = tf.keras.layers.MaxPool2D(pool_size=(3, 3),
                                      strides=2,
                                      padding="same")(x)
        x = ResNeXt.make_bottleneck_layer(x, filter_num=64,
                                          blocks=layer_params[0], training=training)
        x = ResNeXt.make_bottleneck_layer(x, filter_num=128,
                                          blocks=layer_params[1],
                                          stride=2, training=training)
        x = ResNeXt.make_bottleneck_layer(x, filter_num=256,
                                          blocks=layer_params[2],
                                          stride=2, training=training)
        x = ResNeXt.make_bottleneck_layer(x, filter_num=512,
                                          blocks=layer_params[3],
                                          stride=2, training=training)
        x = tf.keras.layers.GlobalAveragePooling2D()(x)
        outputs = tf.keras.layers.Dense(units=NUM_CLASSES, activation=tf.keras.activations.softmax)(x)
        model = tf.keras.Model(inputs=inputs, outputs=outputs)
        return model

    @staticmethod
    def Resnext(repeat_num_list, cardinality, training=None, mask=None):
        inputs = tf.keras.layers.Input(shape=inputs_shape)
        x = tf.keras.layers.Conv2D(filters=64,
                                   kernel_size=(7, 7),
                                   strides=2,
                                   padding="same")(inputs)
        x = tf.keras.layers.BatchNormalization()(x, training=training)
        x = tf.nn.relu(x)
        x = tf.keras.layers.MaxPool2D(pool_size=(3, 3),
                                      strides=2,
                                      padding="same")(x)
        x = ResNeXt.build_ResNeXt_block(x, filters=128,
                                        strides=1,
                                        groups=cardinality,
                                        repeat_num=repeat_num_list[0])
        x = ResNeXt.build_ResNeXt_block(x, filters=256,
                                        strides=2,
                                        groups=cardinality,
                                        repeat_num=repeat_num_list[1])
        x = ResNeXt.build_ResNeXt_block(x, filters=512,
                                        strides=2,
                                        groups=cardinality,
                                        repeat_num=repeat_num_list[2])
        x = ResNeXt.build_ResNeXt_block(x, filters=1024,
                                        strides=2,
                                        groups=cardinality,
                                        repeat_num=repeat_num_list[3])
        x = tf.keras.layers.GlobalAveragePooling2D()(x)
        outputs = tf.keras.layers.Dense(units=NUM_CLASSES, activation=tf.keras.activations.softmax)(x)
        model = tf.keras.Model(inputs=inputs, outputs=outputs)
        return model


# SEResNet
class SEResNet(object):
    @staticmethod
    def seblock(inputs, input_channels, r=16, **kwargs):
        x = tf.keras.layers.GlobalAveragePooling2D()(inputs)
        x = tf.keras.layers.Dense(units=input_channels // r)(x)
        x = tf.nn.relu(x)
        x = tf.keras.layers.Dense(units=input_channels)(x)
        x = tf.nn.sigmoid(x)
        x = tf.expand_dims(x, axis=1)
        x = tf.expand_dims(x, axis=1)
        output = tf.keras.layers.multiply(inputs=[inputs, x])
        return output

    @staticmethod
    def bottleneck(inputs, filter_num, stride=1, training=None):
        identity = tf.keras.layers.Conv2D(filters=filter_num * 4,
                                          kernel_size=(1, 1),
                                          strides=stride)(inputs)
        identity = tf.keras.layers.BatchNormalization()(identity)
        x = tf.keras.layers.Conv2D(filters=filter_num,
                                   kernel_size=(1, 1),
                                   strides=1,
                                   padding='same')(inputs)
        x = tf.keras.layers.BatchNormalization()(x, training)
        x = tf.nn.relu(x)
        x = tf.keras.layers.Conv2D(filters=filter_num,
                                   kernel_size=(3, 3),
                                   strides=stride,
                                   padding='same')(x)
        x = tf.keras.layers.BatchNormalization()(x, training)
        x = tf.nn.relu(x)
        x = tf.keras.layers.Conv2D(filters=filter_num * 4,
                                   kernel_size=(1, 1),
                                   strides=1,
                                   padding='same')(x)
        x = tf.keras.layers.BatchNormalization()(x, training)
        x = SEResNet.seblock(x, input_channels=filter_num * 4)
        output = tf.nn.relu(tf.keras.layers.add([identity, x]))
        return output

    @staticmethod
    def _make_res_block(inputs, filter_num, blocks, stride=1):
        x = SEResNet.bottleneck(inputs, filter_num, stride=stride)
        for _ in range(1, blocks):
            x = SEResNet.bottleneck(x, filter_num, stride=1)
        return x

    @staticmethod
    def SEResNet(block_num, training=None, mask=None):
        inputs = tf.keras.layers.Input(shape=inputs_shape)
        x = tf.keras.layers.Conv2D(filters=64,
                                   kernel_size=(7, 7),
                                   strides=2,
                                   padding='same')(inputs)
        x = tf.keras.layers.BatchNormalization()(x, training)
        x = tf.keras.layers.Activation(tf.keras.activations.relu)(x)
        x = tf.keras.layers.MaxPool2D(pool_size=(3, 3),
                                      strides=2)(x)
        x = SEResNet._make_res_block(x, filter_num=64,
                                     blocks=block_num[0])
        x = SEResNet._make_res_block(x, filter_num=128,
                                     blocks=block_num[1],
                                     stride=2)
        x = SEResNet._make_res_block(x, filter_num=256,
                                     blocks=block_num[2],
                                     stride=2)
        x = SEResNet._make_res_block(x, filter_num=512,
                                     blocks=block_num[3],
                                     stride=2)
        x = tf.keras.layers.GlobalAveragePooling2D()(x)
        outputs = tf.keras.layers.Dense(units=NUM_CLASSES, activation=tf.keras.activations.softmax)(x)
        model = tf.keras.Model(inputs=inputs, outputs=outputs)
        return model


# ShuffleNetV2
class ShuffleNetV2(object):
    @staticmethod
    def channel_shuffle(feature, group):
        channel_num = feature.shape[-1]
        if channel_num % group != 0:
            raise ValueError("The group must be divisible by the shape of the last dimension of the feature.")
        x = tf.reshape(feature, shape=(-1, feature.shape[1], feature.shape[2], group, channel_num // group))
        x = tf.transpose(x, perm=[0, 1, 2, 4, 3])
        x = tf.reshape(x, shape=(-1, feature.shape[1], feature.shape[2], channel_num))
        return x

    @staticmethod
    def ShuffleBlockS1(inputs, in_channels, out_channels, training=None, **kwargs):
        branch, x = tf.split(inputs, num_or_size_splits=2, axis=-1)
        x = tf.keras.layers.Conv2D(filters=out_channels // 2,
                                   kernel_size=(1, 1),
                                   strides=1,
                                   padding="same")(x)
        x = tf.keras.layers.BatchNormalization()(x, training=training)
        x = tf.nn.relu(x)
        x = tf.keras.layers.DepthwiseConv2D(kernel_size=(3, 3), strides=1, padding="same")(x)
        x = tf.keras.layers.BatchNormalization()(x, training=training)
        x = tf.keras.layers.Conv2D(filters=out_channels // 2,
                                   kernel_size=(1, 1),
                                   strides=1,
                                   padding="same")(x)
        x = tf.keras.layers.BatchNormalization()(x, training=training)
        x = tf.nn.relu(x)
        outputs = tf.concat(values=[branch, x], axis=-1)
        outputs = ShuffleNetV2.channel_shuffle(feature=outputs, group=2)
        return outputs

    @staticmethod
    def ShuffleBlockS2(inputs, in_channels, out_channels, training=None, **kwargs):
        x = tf.keras.layers.Conv2D(filters=out_channels // 2,
                                   kernel_size=(1, 1),
                                   strides=1,
                                   padding="same")(inputs)
        x = tf.keras.layers.BatchNormalization()(x, training=training)
        x = tf.nn.relu(x)
        x = tf.keras.layers.DepthwiseConv2D(kernel_size=(3, 3), strides=2, padding="same")(x)
        x = tf.keras.layers.BatchNormalization()(x, training=training)
        x = tf.keras.layers.Conv2D(filters=out_channels - in_channels,
                                   kernel_size=(1, 1),
                                   strides=1,
                                   padding="same")(x)
        x = tf.keras.layers.BatchNormalization()(x, training=training)
        x = tf.nn.relu(x)
        branch = tf.keras.layers.DepthwiseConv2D(kernel_size=(3, 3), strides=2, padding="same")(inputs)
        branch = tf.keras.layers.BatchNormalization()(branch, training=training)
        branch = tf.keras.layers.Conv2D(filters=in_channels,
                                        kernel_size=(1, 1),
                                        strides=1,
                                        padding="same")(branch)
        branch = tf.keras.layers.BatchNormalization()(branch, training=training)
        branch = tf.nn.relu(branch)
        outputs = tf.concat(values=[x, branch], axis=-1)
        outputs = ShuffleNetV2.channel_shuffle(feature=outputs, group=2)
        return outputs

    @staticmethod
    def _make_layer(inputs, repeat_num, in_channels, out_channels):
        x = ShuffleNetV2.ShuffleBlockS2(inputs, in_channels=in_channels, out_channels=out_channels)
        for _ in range(1, repeat_num):
            x = ShuffleNetV2.ShuffleBlockS1(x, in_channels=out_channels, out_channels=out_channels)
        return x

    @staticmethod
    def ShuffleNetV2(channel_scale, training=None, mask=None):
        inputs = tf.keras.layers.Input(shape=inputs_shape)
        x = tf.keras.layers.Conv2D(filters=24, kernel_size=(3, 3), strides=2, padding="same")(inputs)
        x = tf.keras.layers.BatchNormalization()(x, training)
        x = tf.nn.relu(x)
        x = tf.keras.layers.MaxPool2D(pool_size=(3, 3), strides=2, padding="same")(x)
        x = ShuffleNetV2._make_layer(x, repeat_num=4, in_channels=24, out_channels=channel_scale[0])
        x = ShuffleNetV2._make_layer(x, repeat_num=8, in_channels=channel_scale[0], out_channels=channel_scale[1])
        x = ShuffleNetV2._make_layer(x, repeat_num=4, in_channels=channel_scale[1], out_channels=channel_scale[2])
        x = tf.keras.layers.Conv2D(filters=channel_scale[3], kernel_size=(1, 1), strides=1, padding="same")(x)
        x = tf.keras.layers.BatchNormalization()(x, training)
        x = tf.keras.layers.GlobalAveragePooling2D()(x)
        outputs = tf.keras.layers.Dense(units=NUM_CLASSES, activation=tf.keras.activations.softmax)(x)
        model = tf.keras.Model(inputs=inputs, outputs=outputs)
        return model


# SqueezeNet
class SqueezeNet(object):
    @staticmethod
    def FireModule(inputs, s1, e1, e3, **kwargs):
        x = tf.keras.layers.Conv2D(filters=s1,
                                   kernel_size=(1, 1),
                                   strides=1,
                                   padding="same")(inputs)
        x = tf.nn.relu(x)
        y1 = tf.keras.layers.Conv2D(filters=e1,
                                    kernel_size=(1, 1),
                                    strides=1,
                                    padding="same")(x)
        y1 = tf.nn.relu(y1)
        y2 = tf.keras.layers.Conv2D(filters=e3,
                                    kernel_size=(3, 3),
                                    strides=1,
                                    padding="same")(x)
        y2 = tf.nn.relu(y2)
        return tf.concat(values=[y1, y2], axis=-1)

    @staticmethod
    def SqueezeNet(training=None, mask=None):
        inputs = tf.keras.layers.Input(shape=inputs_shape)
        x = tf.keras.layers.Conv2D(filters=96,
                                   kernel_size=(7, 7),
                                   strides=2,
                                   padding="same")(inputs)
        x = tf.keras.layers.MaxPool2D(pool_size=(3, 3),
                                      strides=2)(x)
        x = SqueezeNet.FireModule(x, s1=16, e1=64, e3=64)
        x = SqueezeNet.FireModule(x, s1=16, e1=64, e3=64)
        x = SqueezeNet.FireModule(x, s1=32, e1=128, e3=128)
        x = tf.keras.layers.MaxPool2D(pool_size=(3, 3),
                                      strides=2)(x)
        x = SqueezeNet.FireModule(x, s1=32, e1=128, e3=128)
        x = SqueezeNet.FireModule(x, s1=48, e1=192, e3=192)
        x = SqueezeNet.FireModule(x, s1=48, e1=192, e3=192)
        x = SqueezeNet.FireModule(x, s1=64, e1=256, e3=256)
        x = tf.keras.layers.MaxPool2D(pool_size=(3, 3),
                                      strides=2)(x)
        x = SqueezeNet.FireModule(x, s1=64, e1=256, e3=256)
        x = tf.keras.layers.Dropout(rate=0.5)(x)
        x = tf.keras.layers.Conv2D(filters=NUM_CLASSES,
                                   kernel_size=(1, 1),
                                   strides=1,
                                   padding="same")(x)
        outputs = tf.keras.layers.GlobalAveragePooling2D()(x)
        model = tf.keras.Model(inputs=inputs, outputs=outputs)
        return model


if __name__ == '__main__':
    pass
    Densenet_121 = Densenet.Densenet(num_init_features=64, growth_rate=32, block_layers=[6, 12, 24, 16],
                                     compression_rate=0.5,
                                     drop_rate=0.5)
    Densenet_169 = Densenet.Densenet(num_init_features=64, growth_rate=32, block_layers=[6, 12, 32, 32],
                                     compression_rate=0.5,
                                     drop_rate=0.5)
    Densenet_201 = Densenet.Densenet(num_init_features=64, growth_rate=32, block_layers=[6, 12, 48, 32],
                                     compression_rate=0.5,
                                     drop_rate=0.5)
    Densenet_264 = Densenet.Densenet(num_init_features=64, growth_rate=32, block_layers=[6, 12, 64, 48],
                                     compression_rate=0.5,
                                     drop_rate=0.5)
    Efficient_net_b0 = Efficientnet.Efficientnet(width_coefficient=1.0, depth_coefficient=1.0, dropout_rate=0.2)
    Efficient_net_b1 = Efficientnet.Efficientnet(width_coefficient=1.0, depth_coefficient=1.1, dropout_rate=0.2)
    Efficient_net_b2 = Efficientnet.Efficientnet(width_coefficient=1.1, depth_coefficient=1.2, dropout_rate=0.3)
    Efficient_net_b3 = Efficientnet.Efficientnet(width_coefficient=1.2, depth_coefficient=1.4, dropout_rate=0.3)
    Efficient_net_b4 = Efficientnet.Efficientnet(width_coefficient=1.4, depth_coefficient=1.8, dropout_rate=0.4)
    Efficient_net_b5 = Efficientnet.Efficientnet(width_coefficient=1.6, depth_coefficient=2.2, dropout_rate=0.4)
    Efficient_net_b6 = Efficientnet.Efficientnet(width_coefficient=1.8, depth_coefficient=2.6, dropout_rate=0.5)
    Efficient_net_b7 = Efficientnet.Efficientnet(width_coefficient=2.0, depth_coefficient=3.1, dropout_rate=0.5)
    MobileNetV1 = Mobilenet.MobileNetV1()
    MobileNetV2 = Mobilenet.MobileNetV2()
    MobileNetV3Large = Mobilenet.MobileNetV3Large()
    MobileNetV3Small = Mobilenet.MobileNetV3Small()
    Resnet_18 = ResNeXt.ResNetTypeI(layer_params=(2, 2, 2, 2))
    Resnet_34 = ResNeXt.ResNetTypeI(layer_params=(3, 4, 6, 3))
    Resnet_50 = ResNeXt.ResNetTypeII(layer_params=(3, 4, 6, 3))
    Resnet_101 = ResNeXt.ResNetTypeII(layer_params=(3, 4, 23, 3))
    Resnet_152 = ResNeXt.ResNetTypeII(layer_params=(3, 8, 36, 3))
    ResNeXt50 = ResNeXt.Resnext(repeat_num_list=(3, 4, 6, 3), cardinality=32)
    ResNeXt101 = ResNeXt.Resnext(repeat_num_list=(3, 4, 23, 3), cardinality=32)
    SEResNet50 = SEResNet.SEResNet(block_num=[3, 4, 6, 3])
    SEResNet152 = SEResNet.SEResNet(block_num=[3, 8, 36, 3])
    ShuffleNet_0_5x = ShuffleNetV2.ShuffleNetV2(channel_scale=[48, 96, 192, 1024])
    ShuffleNet_1_0x = ShuffleNetV2.ShuffleNetV2(channel_scale=[116, 232, 464, 1024])
    ShuffleNet_1_5x = ShuffleNetV2.ShuffleNetV2(channel_scale=[176, 352, 704, 1024])
    ShuffleNet_2_0x = ShuffleNetV2.ShuffleNetV2(channel_scale=[244, 488, 976, 2048])
    SqueezeNet = SqueezeNet.SqueezeNet()
    SqueezeNet._layers = [layer for layer in SqueezeNet.layers if not isinstance(layer, dict)]
    tf.keras.utils.plot_model(SqueezeNet, to_file='SqueezeNet.png', show_shapes=True, dpi=48)
