import tensorflow as tf

################################################################################
# Layers
################################################################################
class ConvolutionBnActivation(tf.keras.layers.Layer):
    """
    """
    # def __init__(self, filters, kernel_size, strides=(1, 1), activation=tf.keras.activations.relu, **kwargs):
    def __init__(self, filters, kernel_size, strides=(1, 1), padding="same", data_format=None, dilation_rate=(1, 1),
                 groups=1, activation=None, kernel_initializer="glorot_uniform", bias_initializer="zeros", kernel_regularizer=None,
                 bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None, use_batchnorm=False, 
                 axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True, trainable=True,
                 post_activation="relu", block_name=None):
        super(ConvolutionBnActivation, self).__init__()


        # 2D Convolution Arguments
        self.filters = filters
        self.kernel_size = kernel_size
        self.strides = strides
        self.padding = padding
        self.data_format = data_format
        self.dilation_rate = dilation_rate
        self.activation = activation
        self.use_bias = not (use_batchnorm)
        self.kernel_initializer = kernel_initializer
        self.bias_initializer = bias_initializer
        self.kernel_regularizer = kernel_regularizer
        self.bias_regularizer = bias_regularizer
        self.activity_regularizer = activity_regularizer
        self.kernel_constraint = kernel_constraint
        self.bias_constraint = bias_constraint

        # Batch Normalization Arguments
        self.axis = axis
        self.momentum = momentum
        self.epsilon = epsilon
        self.center = center
        self.scale = scale
        self.trainable = trainable
        
        self.block_name = block_name
        
        self.conv = None
        self.bn = None
        #tf.keras.layers.BatchNormalization(scale=False, momentum=0.9)
        self.post_activation = tf.keras.layers.Activation(post_activation)

    def build(self, input_shape):
        self.conv = tf.keras.layers.Conv2D(
            filters=self.filters,
            kernel_size=self.kernel_size,
            strides=self.strides,
            padding=self.padding,
            data_format=self.data_format,
            dilation_rate=self.dilation_rate,
            activation=self.activation,
            use_bias=self.use_bias,
            kernel_initializer=self.kernel_initializer,
            bias_initializer=self.bias_initializer,
            kernel_regularizer=self.kernel_regularizer,
            bias_regularizer=self.bias_regularizer,
            activity_regularizer=self.activity_regularizer,
            kernel_constraint=self.kernel_constraint,
            bias_constraint=self.bias_constraint,
            name=self.block_name + "_conv" if self.block_name is not None else None

        )

        self.bn = tf.keras.layers.BatchNormalization(
            axis=self.axis,
            momentum=self.momentum,
            epsilon=self.epsilon,
            center=self.center,
            scale=self.scale,
            trainable=self.trainable,
            name=self.block_name + "_bn" if self.block_name is not None else None
        )

    def call(self, x, training=None):
        x = self.conv(x)
        x = self.bn(x, training=training)
        x = self.post_activation(x)

        return x

    def compute_output_shape(self, input_shape):
        print(input_shape)
        return [input_shape[0], input_shape[1], input_shape[2], self.filters]

class AtrousSeparableConvolutionBnReLU(tf.keras.layers.Layer):
    """
    """
    def __init__(self, filters, kernel_size, strides=[1, 1, 1, 1], padding="SAME", data_format=None,
                 dilation=None, channel_multiplier=1, axis=-1, momentum=0.99, epsilon=0.001,
                 center=True, scale=True, trainable=True, post_activation=None, block_name=None):
        super(AtrousSeparableConvolutionBnReLU, self).__init__()

        self.filters = filters
        self.kernel_size = kernel_size
        self.strides = strides
        self.padding = padding
        self.data_format = data_format
        self.dilation = dilation
        self.channel_multiplier = channel_multiplier

        # Batch Normalization Arguments
        self.axis = axis
        self.momentum = momentum
        self.epsilon = epsilon
        self.center = center
        self.scale = scale
        self.trainable = trainable
        
        self.block_name = block_name
        
        self.bn = None

        self.activation = tf.keras.layers.Activation(tf.keras.activations.relu)
        
        self.dw_filter = None
        self.pw_filter = None

    def build(self, input_shape):
        in_channels = input_shape[-1]
        self.dw_filter = self.add_weight(
            name="kernel",
            shape=[self.kernel_size, self.kernel_size, in_channels, self.channel_multiplier],
            initializer=tf.keras.initializers.GlorotNormal(),
            regularizer=tf.keras.regularizers.l2(l=1e-4)
        )
        self.pw_filter = self.add_weight(
            name="kernel",
            shape=[1, 1, in_channels * self.channel_multiplier, self.filters],
            initializer=tf.keras.initializers.GlorotNormal(),
            regularizer=tf.keras.regularizers.l2(l=1e-4)
        )

        self.bn = tf.keras.layers.BatchNormalization(
            axis=self.axis,
            momentum=self.momentum,
            epsilon=self.epsilon,
            center=self.center,
            scale=self.scale,
            trainable=self.trainable,
            name=self.block_name + "_bn" if self.block_name is not None else None
        )
        
    def call(self, x, training=None):
        x = tf.nn.separable_conv2d(
            x,
            self.dw_filter,
            self.pw_filter,
            strides=self.strides,
            dilations=[self.dilation, self.dilation],
            padding=self.padding,
            )
        x = self.bn(x, training=training)
        x = self.activation(x)

        return x

    def compute_output_shape(self, input_shape):
        print(input_shape)
        return [input_shape[0], input_shape[1], input_shape[2], self.filters]

class AtrousSpatialPyramidPooling(tf.keras.layers.Layer):
    """
    """
    def __init__(self, atrous_rates, filters):
        super(AtrousSpatialPyramidPooling, self).__init__()
        self.filters = filters

        # adapt scale and mometum for bn
        self.conv_bn_relu = ConvolutionBnActivation(filters=filters, kernel_size=1)

        self.atrous_sepconv_bn_relu_1 = AtrousSeparableConvolutionBnReLU(dilation=atrous_rates[0], filters=filters, kernel_size=3)
        self.atrous_sepconv_bn_relu_2 = AtrousSeparableConvolutionBnReLU(dilation=atrous_rates[1], filters=filters, kernel_size=3)
        self.atrous_sepconv_bn_relu_3 = AtrousSeparableConvolutionBnReLU(dilation=atrous_rates[2], filters=filters, kernel_size=3)

        # 1x1 reduction convolutions
        self.conv_reduction_1 = tf.keras.layers.Conv2D(
            filters=256,
            kernel_size=1,
            use_bias=False,
            kernel_regularizer=tf.keras.regularizers.l2(l=1e-4))


    def call(self, input_tensor, training=None):
        # global average pooling input_tensor
        glob_avg_pool = tf.reduce_mean(input_tensor, axis=[1, 2], keepdims=True)
        glob_avg_pool = self.conv_bn_relu(glob_avg_pool, training=training)
        glob_avg_pool = tf.image.resize(glob_avg_pool, [input_tensor.shape[1], input_tensor.shape[2]])

        # process with atrous
        w = self.conv_bn_relu(input_tensor, training=training)
        x = self.atrous_sepconv_bn_relu_1(input_tensor, training=training)
        y = self.atrous_sepconv_bn_relu_2(input_tensor, training=training)
        z = self.atrous_sepconv_bn_relu_3(input_tensor, training=training)

        # concatenation
        net = tf.concat([glob_avg_pool, w, x, y, z], axis=-1)
        net = self.conv_reduction_1(net, training=training)

        return net

    def compute_output_shape(self, input_shape):
        print(input_shape)
        return [input_shape[0], input_shape[1], input_shape[2], 256]

class Upsample_x2_Block(tf.keras.layers.Layer):
    """
    """
    def __init__(self, filters, trainable=None):
        super(Upsample_x2_Block, self).__init__()
        self.trainable = trainable

        self.upsample2d_size2 = tf.keras.layers.UpSampling2D(size=2, interpolation="bilinear")
        self.conv3x3_bn_relu1 = tf.keras.layers.Conv2D(filters, kernel_size=(2, 2), padding="same")

        self.concat = tf.keras.layers.Concatenate(axis=3)

        self.conv3x3_bn_relu2 = ConvolutionBnActivation(filters, kernel_size=(3, 3), post_activation="relu")
        self.conv3x3_bn_relu3 = ConvolutionBnActivation(filters, kernel_size=(3, 3), post_activation="relu")

    def call(self, x, skip, training=None):
        x = self.upsample2d_size2(x)
        x = self.conv3x3_bn_relu1(x, training=training)

        if skip is not None:
            x = self.concat([x, skip])

        x = self.conv3x3_bn_relu2(x, training=training)
        x = self.conv3x3_bn_relu3(x, training=training)

        return x

    def compute_output_shape(self, input_shape):
        print(input_shape)
        return [input_shape[0], input_shape[1] * 2, input_shape[2] * 2, input_shape[3]]

class SpatialContextBlock(tf.keras.layers.Layer):
    def __init__(self, level, filters=256, pooling_type="avg"):
        super(SpatialContextBlock, self).__init__()

        self.level = level
        self.filters = filters
        self.pooling_type = pooling_type

        self.pooling2d = None
        self.conv1x1_bn_relu = ConvolutionBnActivation(filters, kernel_size=(1, 1))
        self.upsample2d = None

    def build(self, input_shape):
        if self.pooling_type not in ("max", "avg"):
            raise ValueError("Unsupported pooling type - '{}'".format(pooling_type) + "Use 'avg' or 'max'")

        self.pooling2d = tf.keras.layers.MaxPool2D if self.pooling_type == "max" else tf.keras.layers.AveragePooling2D

        spatial_size = input_shape[1:3] if K.image_data_format() == "channels_last" else input_shape[2:]

        pool_size = up_size = [spatial_size[0] // self.level, spatial_size[1] // self.level]
        self.pooling2d = self.pooling2d(pool_size, strides=pool_size, padding="same")
        
        self.upsample2d = tf.keras.layers.UpSampling2D(up_size, interpolation="bilinear")

    def call(self, x, training=None):
        x = self.pooling2d(x, training=training)
        x = self.conv1x1_bn_relu(x, training=training)
        x = self.upsample2d(x)

        return x

    def compute_output_shape(self, input_shape):
        return input_shape

class FPNBlock(tf.keras.layers.Layer):
    def __init__(self, filters):
        super(FPNBlock, self).__init__()

        self.filters = filters
        self.input_filters = None

        self.conv1x1_1 = tf.keras.layers.Conv2D(filters, (1, 1), padding="same", kernel_initializer="he_uniform")
        self.conv1x1_2 = tf.keras.layers.Conv2D(filters, (1, 1), padding="same", kernel_initializer="he_uniform")

        self.upsample2d = tf.keras.layers.UpSampling2D((2, 2))
        self.add = tf.keras.layers.Add()

    def build(self, input_shape):
        if input_shape != self.filters:
            self.input_filters = True

    def call(self, x, skip, training=None):
        if self.input_filters:
            x = self.conv1x1_1(x)

        skip = self.conv1x1_2(skip)
        x = self.upsample2d(x)
        x = self.add([x, skip])

        return x