import tensorflow as tf
import tensorflow.keras.backend as K

################################################################################
# Layers
################################################################################
from tensorflow.keras import activations


class ConvolutionBnActivation(tf.keras.layers.Layer):
    """
    """
    # def __init__(self, filters, kernel_size, strides=(1, 1), activation=tf.keras.activations.relu, **kwargs):
    def __init__(self, filters, kernel_size, strides=(1, 1), padding="same", data_format=None, dilation_rate=(1, 1),
                 groups=1, activation=None, kernel_initializer="glorot_uniform", bias_initializer="zeros", kernel_regularizer=None,
                 bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None, use_batchnorm=False, 
                 axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True, trainable=True,
                 post_activation="relu", block_name=None, **kwargs):
        super(ConvolutionBnActivation, self).__init__(**kwargs)


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
        # self.post_activation = tf.keras.layers.Activation(post_activation)
        self.post_activation = activations.get(post_activation)

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

    def get_config(self):
        config = {
            "filters": self.filters,
            "kernel_size": self.kernel_size,
            "strides": self.strides,
            "padding": self.padding,
            "data_format": self.data_format,
            "dilation_rate": self.dilation_rate,
            "activation": activations.serialize(self.activation),
            "use_batchnorm": not self.use_bias,
            "kernel_initializer": self.kernel_initializer,
            "bias_initializer": self.bias_initializer,
            "kernel_regularizer": self.kernel_regularizer,
            "bias_regularizer": self.bias_regularizer,
            "activity_regularizer": self.activity_regularizer,
            "kernel_constraint": self.kernel_constraint,
            "bias_constraint": self.bias_constraint,
            # Batch Normalization Arguments
            "axis": self.axis,
            "momentum": self.momentum,
            "epsilon": self.epsilon,
            "center": self.center,
            "scale": self.scale,
            "trainable": self.trainable,
            "block_name": self.block_name,


        # self.use_bias = not (use_batchnorm)
        #
        #
        #
        #
        # self.conv = None
        # self.bn = None


        # tf.keras.layers.BatchNormalization(scale=False, momentum=0.9)
        # self.post_activation = tf.keras.layers.Activation(post_activation)
            "post_activation": activations.serialize(self.post_activation),
        }
        base_config = super(ConvolutionBnActivation, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

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
            name="dw_kernel",
            shape=[self.kernel_size, self.kernel_size, in_channels, self.channel_multiplier],
            initializer=tf.keras.initializers.GlorotNormal(),
            regularizer=tf.keras.regularizers.l2(l=1e-4),
            trainable=True
        )
        self.pw_filter = self.add_weight(
            name="pw_kernel",
            shape=[1, 1, in_channels * self.channel_multiplier, self.filters],
            initializer=tf.keras.initializers.GlorotNormal(),
            regularizer=tf.keras.regularizers.l2(l=1e-4),
            trainable=True
        )

        self.bn = tf.keras.layers.BatchNormalization(
            axis=self.axis,
            momentum=self.momentum,
            epsilon=self.epsilon,
            center=self.center,
            scale=self.scale,
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

class AtrousSpatialPyramidPoolingV3(tf.keras.layers.Layer):
    """
    """
    def __init__(self, atrous_rates, filters):
        super(AtrousSpatialPyramidPoolingV3, self).__init__()
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
        glob_avg_pool = tf.keras.layers.Lambda(lambda x: tf.reduce_mean(x, axis=[1, 2], keepdims=True))(input_tensor)
        glob_avg_pool = self.conv_bn_relu(glob_avg_pool, training=training)
        glob_avg_pool = tf.keras.layers.Lambda(lambda x: tf.image.resize(x, [input_tensor.shape[1], input_tensor.shape[2]]))(glob_avg_pool)

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
    def __init__(self, filters, trainable=None, **kwargs):
        super(Upsample_x2_Block, self).__init__(**kwargs)
        self.trainable = trainable
        self.filters = filters

        self.upsample2d_size2 = tf.keras.layers.UpSampling2D(size=2, interpolation="bilinear")
        self.conv2x2_bn_relu = tf.keras.layers.Conv2D(filters, kernel_size=(2, 2), padding="same")

        self.concat = tf.keras.layers.Concatenate(axis=3)

        self.conv3x3_bn_relu1 = ConvolutionBnActivation(filters, kernel_size=(3, 3), post_activation="relu")
        self.conv3x3_bn_relu2 = ConvolutionBnActivation(filters, kernel_size=(3, 3), post_activation="relu")

    def call(self, x, skip=None, training=None):
        x = self.upsample2d_size2(x)
        x = self.conv2x2_bn_relu(x, training=training)

        if skip is not None:
            x = self.concat([x, skip])

        x = self.conv3x3_bn_relu1(x, training=training)
        x = self.conv3x3_bn_relu2(x, training=training)

        return x

    def compute_output_shape(self, input_shape):
        print(input_shape)
        return [input_shape[0], input_shape[1] * 2, input_shape[2] * 2, input_shape[3]]

    def get_config(self):
        config = {
            "filters": self.filters,
            "trainable": self.trainable,
        }
        base_config = super(Upsample_x2_Block, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

class Upsample_x2_Add_Block(tf.keras.layers.Layer):
    """
    """
    def __init__(self, filters):
        super(Upsample_x2_Add_Block, self).__init__()

        self.upsample2d_size2 = tf.keras.layers.UpSampling2D(size=2, interpolation="bilinear")
        self.conv1x1_bn_relu = tf.keras.layers.Conv2D(filters, kernel_size=(1, 1), padding="same")
        self.add = tf.keras.layers.Add()

    def call(self, x, skip, training=None):
        x = self.upsample2d_size2(x)
        skip = self.conv1x1_bn_relu(x, training=training)
        x = self.add([x, skip])

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

class AtrousSpatialPyramidPoolingV1(tf.keras.layers.Layer):
    def __init__(self, filters):
        super(AtrousSpatialPyramidPoolingV1, self).__init__()

        self.filters = filters
        
        self.conv1x1_bn_relu = ConvolutionBnActivation(filters, (1, 1), post_activation="relu")
        self.atrous6_conv3x3_bn_relu = ConvolutionBnActivation(filters, (3, 3), dilation_rate=6, post_activation="relu")
        self.atrous12_conv3x3_bn_relu = ConvolutionBnActivation(filters, (3, 3), dilation_rate=12, post_activation="relu")
        self.atrous18_conv3x3_bn_relu = ConvolutionBnActivation(filters, (3, 3), dilation_rate=18, post_activation="relu")
        self.atrous24_conv3x3_bn_relu = ConvolutionBnActivation(filters, (3, 3), dilation_rate=24, post_activation="relu")

        axis = 3 if K.image_data_format() == "channels_last" else 1

        self.concat = tf.keras.layers.Concatenate(axis=axis)

    def call(self, x, training=None):
        
        x1 = self.conv1x1_bn_relu(x, training=training)
        x3_r6 = self.atrous6_conv3x3_bn_relu(x, training=training)
        x3_r12 = self.atrous12_conv3x3_bn_relu(x, training=training)
        x3_r18 = self.atrous18_conv3x3_bn_relu(x, training=training)
        x3_r24 = self.atrous24_conv3x3_bn_relu(x, training=training)
        
        x = self.concat([x1, x3_r6, x3_r12, x3_r18, x3_r24])

        return x

class Base_OC_Module(tf.keras.layers.Layer):
    def __init__(self, filters):
        super(Base_OC_Module, self).__init__()

        self.filters = filters

        axis = 3 if K.image_data_format() == "channels_last" else 1

        self.self_attention_block2d = SelfAttentionBlock2D(filters)
        self.concat = tf.keras.layers.Concatenate(axis=axis)
        self.conv1x1_bn_relu = ConvolutionBnActivation(filters, (1, 1))

    def call(self, x, training=None):
        
        attention = self.self_attention_block2d(x, training=training)
        x = self.concat([attention, x])
        x = self.conv1x1_bn_relu(x, training=training)

        return x

class Pyramid_OC_Module(tf.keras.layers.Layer):
    def __init__(self, levels, filters=256, pooling_type="avg"):
        super(Pyramid_OC_Module, self).__init__()

        self.levels = levels
        self.filters = filters
        self.pooling_type = pooling_type

        self.pyramid_block_1 = SelfAttentionBlock2D(filters)
        self.pyramid_block_2 = SelfAttentionBlock2D(filters)
        self.pyramid_block_3 = SelfAttentionBlock2D(filters)
        self.pyramid_block_6 = SelfAttentionBlock2D(filters)

        self.pooling2d_1 = None
        self.pooling2d_2 = None
        self.pooling2d_3 = None
        self.pooling2d_6 = None

        self.conv1x1_bn_relu_1 = ConvolutionBnActivation(filters, kernel_size=(1, 1))
        
        self.upsample2d_1 = None
        self.upsample2d_2 = None
        self.upsample2d_3 = None
        self.upsample2d_6 = None
        
        axis = 3 if K.image_data_format() == "channels_last" else 1

        self.concat = tf.keras.layers.Concatenate(axis=axis)
        self.conv1x1_bn_relu_2 = ConvolutionBnActivation(filters, kernel_size=(1, 1))

    def build(self, input_shape):
        if self.pooling_type not in ("max", "avg"):
            raise ValueError("Unsupported pooling type - '{}'".format(pooling_type) + "Use 'avg' or 'max'")

        self.pooling2d_1 = tf.keras.layers.MaxPool2D if self.pooling_type == "max" else tf.keras.layers.AveragePooling2D
        self.pooling2d_2 = tf.keras.layers.MaxPool2D if self.pooling_type == "max" else tf.keras.layers.AveragePooling2D
        self.pooling2d_3 = tf.keras.layers.MaxPool2D if self.pooling_type == "max" else tf.keras.layers.AveragePooling2D
        self.pooling2d_6 = tf.keras.layers.MaxPool2D if self.pooling_type == "max" else tf.keras.layers.AveragePooling2D

        spatial_size = input_shape[1:3] if K.image_data_format() == "channels_last" else input_shape[2:]
        pool_size_1 = up_size_1 = [spatial_size[0] // self.levels[0], spatial_size[1] // self.levels[0]]
        pool_size_2 = up_size_2 = [spatial_size[0] // self.levels[1], spatial_size[1] // self.levels[1]]
        pool_size_3 = up_size_3 = [spatial_size[0] // self.levels[2], spatial_size[1] // self.levels[2]]
        pool_size_6 = up_size_6 = [spatial_size[0] // self.levels[3], spatial_size[1] // self.levels[3]]

        self.pooling2d_1 = self.pooling2d_1(pool_size_1, strides=pool_size_1, padding="same")
        self.pooling2d_2 = self.pooling2d_2(pool_size_2, strides=pool_size_2, padding="same")
        self.pooling2d_3 = self.pooling2d_3(pool_size_3, strides=pool_size_3, padding="same")
        self.pooling2d_6 = self.pooling2d_6(pool_size_6, strides=pool_size_6, padding="same")
        
        self.upsample2d_1 = tf.keras.layers.UpSampling2D(up_size_1, interpolation="bilinear")
        self.upsample2d_2 = tf.keras.layers.UpSampling2D(up_size_2, interpolation="bilinear")
        self.upsample2d_3 = tf.keras.layers.UpSampling2D(up_size_3, interpolation="bilinear")
        self.upsample2d_6 = tf.keras.layers.UpSampling2D(up_size_6, interpolation="bilinear")

    def call(self, x, training=None):
        attention_1 = self.pooling2d_1(x, training=training)
        attention_1 = self.pyramid_block_1(attention_1, training=training)
        attention_1 = self.upsample2d_1(attention_1)
        attention_2 = self.pooling2d_2(x, training=training)
        attention_2 = self.pyramid_block_2(attention_2, training=training)
        attention_2 = self.upsample2d_2(attention_2)
        attention_3 = self.pooling2d_3(x, training=training)
        attention_3 = self.pyramid_block_3(attention_3, training=training)
        attention_3 = self.upsample2d_3(attention_3)
        attention_6 = self.pooling2d_6(x, training=training)
        attention_6 = self.pyramid_block_6(attention_6, training=training)
        attention_6 = self.upsample2d_6(attention_6)

        x = self.conv1x1_bn_relu_1(x, training=training)
        
        x = self.concat([attention_1, attention_2, attention_3, attention_6, x])
        x = self.conv1x1_bn_relu_2(x, training=training)

        return x

class ASP_OC_Module(tf.keras.layers.Layer):
    def __init__(self, filters, dilations):
        super(ASP_OC_Module, self).__init__()
        self.filters = filters
        self.dilations = dilations

        self.conv3x3_bn_relu = ConvolutionBnActivation(filters, (3, 3))
        self.context = Base_OC_Module(filters)

        self.conv1x1_bn_relu_1 = ConvolutionBnActivation(filters, (1, 1), post_activation="relu")
        self.atrous6_conv3x3_bn_relu = ConvolutionBnActivation(filters, (3, 3), dilation_rate=6, post_activation="relu")
        self.atrous12_conv3x3_bn_relu = ConvolutionBnActivation(filters, (3, 3), dilation_rate=12, post_activation="relu")
        self.atrous18_conv3x3_bn_relu = ConvolutionBnActivation(filters, (3, 3), dilation_rate=18, post_activation="relu")
        
        axis = 3 if K.image_data_format() == "channels_last" else 1

        self.concat = tf.keras.layers.Concatenate(axis=axis)
        self.conv1x1_bn_relu_2 = ConvolutionBnActivation(filters, (1, 1))

    def call(self, x, training=None):

        a = self.conv3x3_bn_relu(x, training=training)
        a = self.context(a, training=training)
        b = self.conv1x1_bn_relu_1(x, training=training)
        c = self.atrous6_conv3x3_bn_relu(x, training=training)
        d = self.atrous12_conv3x3_bn_relu(x, training=training)
        e = self.atrous18_conv3x3_bn_relu(x, training=training)

        x = self.concat([a, b, c, d, e])
        x = self.conv1x1_bn_relu_2(x, training=training)

        return x


class PAM_Module(tf.keras.layers.Layer):
    def __init__(self, filters):
        super(PAM_Module, self).__init__()

        self.filters = filters

        axis = 3 if K.image_data_format() == "channels_last" else 1
        self.concat = tf.keras.layers.Concatenate(axis=axis)

        self.conv1x1_bn_relu_1 = ConvolutionBnActivation(filters, (1, 1))
        self.conv1x1_bn_relu_2 = ConvolutionBnActivation(filters, (1, 1))
        self.conv1x1_bn_relu_3 = ConvolutionBnActivation(filters, (1, 1))

        self.gamma = None

        self.softmax = tf.keras.layers.Activation("softmax")
    
    def build(self, input_shape):
        self.gamma = self.add_weight(
            shape=(1,),
            initializer="random_normal",
            name="pam_gamma",
            trainable=True,
            )
    
    def call(self, x, training=None):
        BS, C, H, W = x.shape

        query = self.conv1x1_bn_relu_1(x, training=training)
        key = self.conv1x1_bn_relu_2(x, training=training)

        if K.image_data_format() == "channels_last":
            query = tf.keras.layers.Reshape((H * W, -1))(query) 
            key = tf.keras.layers.Reshape((H * W, -1))(key)

            energy = tf.linalg.matmul(query, key, transpose_b=True)
        else:
            query = tf.keras.layers.Reshape((-1, H * W))(query)
            key = tf.keras.layers.Reshape((-1, H * W))(key)
            
            energy = tf.linalg.matmul(query, key, transpose_a=True)
        
        attention = self.softmax(energy)

        value = self.conv1x1_bn_relu_3(x, training=training)

        if K.image_data_format() == "channels_last":
            value = tf.keras.layers.Reshape((H * W, -1))(value) 
            out = tf.linalg.matmul(value, attention, transpose_a=True)
        else:
            value = tf.keras.layers.Reshape((-1, H * W))(value)
            out = tf.linalg.matmul(value, attention)

        out = tf.keras.layers.Reshape(x.shape[1:])(out)
        out = self.gamma * out + x

        return out

class CAM_Module(tf.keras.layers.Layer):
    def __init__(self, filters):
        super(CAM_Module, self).__init__()

        self.filters = filters

        self.gamma = None

        self.softmax = tf.keras.layers.Activation("softmax")

    def build(self, input_shape):
        self.gamma = self.add_weight(
            shape=(1,),
            initializer="random_normal",
            name="cam_gamma",
            trainable=True,
            )
    
    def call(self, x, training=None):
        BS, C, H, W = x.shape

        if K.image_data_format() == "channels_last":
            query = tf.keras.layers.Reshape((-1, C))(x) 
            key = tf.keras.layers.Reshape((-1, C))(x)

            energy = tf.linalg.matmul(query, key, transpose_a=True)
            energy_2 = tf.math.reduce_max(energy, axis=1, keepdims=True)[0] - energy
        else:
            query = tf.keras.layers.Reshape((C, -1))(query)
            key = tf.keras.layers.Reshape((C, -1))(key)
        
            energy = tf.linalg.matmul(query, key, transpose_b=True)
            energy_2 = tf.math.reduce_max(energy, axis=-1, keepdims=True)[0] - energy
        
        attention = self.softmax(energy_2)

        if K.image_data_format() == "channels_last":
            value = tf.keras.layers.Reshape((-1, C))(x)
            out = tf.linalg.matmul(attention, value, transpose_b=True) 
        else:
            value = tf.keras.layers.Reshape((C, -1))(x)
            out = tf.linalg.matmul(attention, value)

        out = tf.keras.layers.Reshape(x.shape[1:])(out)
        out = self.gamma * out + x

        return out


class SelfAttentionBlock2D(tf.keras.layers.Layer):
    def __init__(self, filters):
        super(SelfAttentionBlock2D, self).__init__()

        self.filters = filters

        self.conv1x1_1 = tf.keras.layers.Conv2D(filters // 8, (1, 1), padding="same")
        self.conv1x1_2 = tf.keras.layers.Conv2D(filters // 8, (1, 1), padding="same")
        self.conv1x1_3 = tf.keras.layers.Conv2D(filters // 2, (1, 1), padding="same")
        self.conv1x1_4 = tf.keras.layers.Conv2D(filters, (1, 1), padding="same")

        self.gamma = None
        
        self.softmax_activation = tf.keras.layers.Activation("softmax")

    def build(self, input_shape):
        self.gamma = self.add_weight(
            shape=(1,),
            initializer="random_normal",
            name="gamma",
            trainable=True,
            )

    def call(self, x, training=None):
        f = self.conv1x1_1(x, training=training)

        g = self.conv1x1_2(x, training=training)

        h = self.conv1x1_3(x, training=training)

        g = tf.reshape(g, (g.shape[0], -1, g.shape[-1]))
        f = tf.reshape(f, (f.shape[0], -1, f.shape[-1]))

        # s = tf.matmul(tf.reshape(g, (x.shape[0], -1, x.shape[-1])), tf.reshape(f, (x.shape[0], -1, x.shape[-1])), transpose_b=True)
        s = tf.matmul(g, f, transpose_b=True)
        beta = self.softmax_activation(s)

        h = tf.reshape(h, (h.shape[0], -1, h.shape[-1]))

        o = tf.matmul(beta, h)

        o = tf.reshape(o, shape=(x.shape[0], x.shape[1], x.shape[2], x.shape[3] // 2))
        o = self.conv1x1_4(o, training=training)
        x = self.gamma * o + x

        return x

class GlobalPooling(tf.keras.layers.Layer):
    def __init__(self, filters):
        super(GlobalPooling, self).__init__()
        self.filters = filters

        self.conv1x1_bn_relu = ConvolutionBnActivation(filters, (1, 1))

    def call(self, x, training=None):
        if K.image_data_format() == "channels_last":
            BS, H, W, C = x.shape
            glob_avg_pool = tf.keras.layers.Lambda(lambda x: tf.reduce_mean(x, axis=[1, 2], keepdims=True))(x)
            glob_avg_pool = self.conv1x1_bn_relu(glob_avg_pool, training=training)
            glob_avg_pool = tf.keras.layers.Lambda(lambda x: tf.image.resize(x, [H, W]))(glob_avg_pool)
        else:
            BS, C, H, W = x.shape
            glob_avg_pool = tf.keras.layers.Lambda(lambda x: tf.reduce_mean(x, axis=[1, 2], keepdims=True))(x)
            glob_avg_pool = self.conv1x1_bn_relu(glob_avg_pool, training=training)
            glob_avg_pool = tf.keras.layers.Lambda(lambda x: tf.image.resize(x, [H, W]))(glob_avg_pool)

        return glob_avg_pool

class MixtureOfSoftMaxACF(tf.keras.layers.Layer):
    def __init__(self, d_k, att_dropout=0.1):
        super(MixtureOfSoftMaxACF, self).__init__()
        self.temperature = tf.math.pow(tf.cast(d_k, tf.float32), 0.5)
        self.att_dropout = att_dropout
        
        self.dropout = tf.keras.layers.Dropout(att_dropout)
        self.softmax_1 = tf.keras.layers.Activation("softmax")
        self.softmax_2 = tf.keras.layers.Activation("softmax")

        self.d_k = d_k

    def call(self, qt, kt, vt, training=None):
        if K.image_data_format() == "channels_last":
            BS, N, d_k = qt.shape # (BS, H * W, C)

            assert d_k == self.d_k
            d = d_k

            q = tf.keras.layers.Reshape((N, d))(qt)          # (BS, N, d)
            # q = tf.transpose(q, perm=[0, 2, 1])                 # (BS, d, N)
            N2 = kt.shape[1]
            kt = tf.keras.layers.Reshape((N2, d))(kt)       # (BS, N2, d)
            # v = tf.keras.layers.transpose(vt, perm=[0, 2, 1])   # (BS, d, N2)

            att = tf.linalg.matmul(q, kt, transpose_b=True)     # (BS, N, N2)
            att = att / self.temperature                        # (BS, N, N2)
            att = self.softmax_2(att)                           # (BS, N, N2)
            att = self.dropout(att, training=training)          # (BS, N, N2)

            out = tf.linalg.matmul(att, vt)                     # (BS, N, d)

        else:
            BS, d_k, N = qt.shape

            assert d_k == self.d_k
            d = d_k

            q = tf.keras.layers.Reshape((d, N))(qt)          # (BS, d, N)
            # q = tf.transpose(q, perm=[0, 2, 1])                 # (BS, N, d)
            N2 = kt.shape[2]
            kt = tf.keras.layers.Reshape((d, N2))(kt)       # (BS, d, N2)
            # v = tf.transpose(vt, perm=[0, 2, 1])                # (BS, N2, d)

            att = tf.linalg.matmul(q, kt, transpose_a=True)     # (BS, N, N2)
            att = att / self.temperature                        # (BS, N, N2)
            att = self.softmax_2(att)                           # (BS, N, N2)
            att = self.dropout(att, training=training)          # (BS, N, N2)

            out = tf.linalg.matmul(att, vt, transpose_b=True)   # (BS, N, d)

        return out

class AggCF_Module(tf.keras.layers.Layer):
    def __init__(self, filters, kq_transform="conv", value_transform="conv",
                 pooling=True, concat=False, dropout=0.1):
        super(AggCF_Module, self).__init__()
        self.filters = filters
        self.kq_transform = kq_transform
        self.value_transform = value_transform
        self.pooling = pooling
        self.concat = concat # if True concat else Add
        self.dropout = dropout

        self.avg_pool2d_1 = tf.keras.layers.AveragePooling2D(pool_size=(2, 2), padding="same", data_format=K.image_data_format())
        self.avg_pool2d_2 = tf.keras.layers.AveragePooling2D(pool_size=(2, 2), padding="same", data_format=K.image_data_format())
        
        self.conv_ks_1 = None
        self.conv_ks_2 = None
        self.conv_vs = None

        self.attention = MixtureOfSoftMaxACF(d_k=filters, att_dropout=0.1)

        self.conv1x1_bn_relu = tf.keras.layers.Conv2D(filters, (1, 1), padding="same")
        
        axis = 3 if K.image_data_format() == "channels_last" else 1
        self.bn = tf.keras.layers.BatchNormalization(axis=axis)
        self.concat = tf.keras.layers.Concatenate(axis=axis)
        self.add = tf.keras.layers.Add()

    def build(self, input_shape):
        if self.kq_transform == "conv":
            self.conv_ks_1 = tf.keras.layers.Conv2D(self.filters, (1, 1), padding="same")
            self.conv_ks_2 = tf.keras.layers.Conv2D(self.filters, (1, 1), padding="same")
        elif self.kq_transform == "ffn":
            self.conv_ks_1 = tf.keras.Sequential(
                [ConvolutionBnActivation(self.filters, (3, 3)),
                tf.keras.layers.Conv2D(self.filters, (1, 1), padding="same")]
            )
            self.conv_ks_2 = tf.keras.Sequential(
                [ConvolutionBnActivation(self.filters, (3, 3)),
                tf.keras.layers.Conv2D(self.filters, (1, 1), padding="same")]
            )
        elif self.kq_transform == "dffn":
            self.conv_ks_1 = tf.keras.Sequential(
                [ConvolutionBnActivation(self.filters, (3, 3), dilation_rate=(4, 4)),
                tf.keras.layers.Conv2D(self.filters, (1, 1), padding="same")]
            )
            self.conv_ks_2 = tf.keras.Sequential(
                [ConvolutionBnActivation(self.filters, (3, 3), dilation_rate=(4, 4)),
                tf.keras.layers.Conv2D(self.filters, (1, 1), padding="same")]
            )
        else:
            raise NotImplementedError("Allowed options for 'kq_transform' are only ('conv', 'ffn', 'dffn'), got {}".format(self.kq_transform))
        
        if self.value_transform == "conv":
            self.conv_vs = tf.keras.layers.Conv2D(self.filters, (1, 1), padding="same")
        else:
            raise NotImplementedError("Allowed options for 'value_transform' is only 'conv', got {}".format(self.kq_transform))

    def call(self, x, training=None):
        residual = x
        d_k = self.filters / 8
        if K.image_data_format() == "channels_last":
            BS, H, W, C = x.shape
            
            if self.pooling:
                qt = self.conv_ks_1(x, training=training)
                qt = tf.keras.layers.Reshape((H * W, -1))(qt)       # (BS, N, C)
                kt = self.avg_pool2d_1(x)
                kt = self.conv_ks_2(kt, training=training)
                kt = tf.keras.layers.Reshape((H * W // 4, -1))(kt)  # (BS, N / 4, C)
                vt = self.avg_pool2d_2(x)
                vt = self.conv_vs(vt, training=training)
                vt = tf.keras.layers.Reshape((H * W // 4, -1))(vt)  # (BS, N / 4, C)
            else:
                qt = self.conv_ks_1(x, training=training)
                qt = tf.keras.layers.Reshape((H * W, -1))(qt)       # (BS, N, C)
                kt = self.conv_ks_2(x, training=training)
                kt = tf.keras.layers.Reshape((H * W, -1))(kt)       # (BS, N, C)
                vt = self.conv_vs(x, training=training)
                vt = tf.keras.layers.Reshape((H * W, -1))(vt)       # (BS, N, C)

            out = self.attention(qt, kt, vt, training=training)     # (BS, N, C)

            # out = tf.transpose(out, perm=[0, 2, 1])                 
            out = tf.keras.layers.Reshape((H, W, -1))(out)          # (BS, H, W, C)
            
        else:
            BS, C, H, W = x.shape

            if self.pooling:
                qt = self.conv_ks_1(x, training=training)
                qt = tf.keras.layers.Reshape((-1, H * W))(qt)       # (BS, C, N)
                kt = self.avg_pool2d_1(x)
                kt = self.conv_ks_2(kt, training=training)
                kt = tf.keras.layers.Reshape((-1, H * W // 4))(kt)  # (BS, C, N / 4)
                vt = self.avg_pool2d_2(x)
                vt = self.conv_vs(vt, training=training)
                vt = tf.keras.layers.Reshape((-1, H * W // 4))(vt)  # (BS, C, N / 4)
            else:
                qt = self.conv_ks_1(x, training=training)
                qt = tf.keras.layers.Reshape((-1, H * W))(qt)       # (BS, C, N)
                kt = self.conv_ks_2(x, training=training)
                kt = tf.keras.layers.Reshape((-1, H * W))(kt)       # (BS, C, N)
                vt = self.conv_vs(x, training=training)
                vt = tf.keras.layers.Reshape((-1, H * W))(vt)       # (BS, C, N)

            out = self.attention(qt, kt, vt)                        # (BS, N, C)

            out = tf.transpose(out, perm=[0, 2, 1])                 # (BS, C, N)
            out = tf.keras.layers.Reshape((-1, H, W))(out)          # (BS, C, H, W)

        out = self.conv1x1_bn_relu(out, training=training)
        if self.concat:
            out = self.concat([out, residual])
        else:
            out = self.add([out, residual])
              
        return out

class SpatialGather_Module(tf.keras.layers.Layer):
    def __init__(self, scale=1):
        super(SpatialGather_Module, self).__init__()

        self.scale = scale

        self.softmax = tf.keras.layers.Activation("softmax")

    def call(self, features, probabilities, training=None):
        if K.image_data_format() == "channels_last":
            BS, H, W, C = probabilities.shape
            p = tf.keras.layers.Reshape((-1, C))(probabilities)             # (BS, N, C)
            f = tf.keras.layers.Reshape((-1, features.shape[-1]))(features) # (BS, N, C2)

            p = self.softmax(self.scale * p)                                # (BS, N, C)
            ocr_context = tf.linalg.matmul(p, f, transpose_a=True)          # (BS, C, C2)

        else:
            BS, C, H, W = probabilities.shape
            p = tf.keras.layers.Reshape((C, -1))(probabilities)             # (BS, C, N)
            f = tf.keras.layers.Reshape((features.shape[1], -1))(features)  # (BS, C2, N)
            
            p = self.softmax(self.scale * p)                                # (BS, C, N)
            ocr_context = tf.linalg.matmul(p, f, transpose_b=True)          # (BS, C, C2)

        return ocr_context

class ObjectAttentionBlock2D(tf.keras.layers.Layer):
    def __init__(self, filters, scale=1.0):
        super(ObjectAttentionBlock2D, self).__init__()
        self.filters = filters
        self.scale = scale

        self.max_pool2d = tf.keras.layers.MaxPooling2D(pool_size=(scale, scale))
        self.f_pixel = tf.keras.models.Sequential([
                                                    ConvolutionBnActivation(filters, (1, 1)),
                                                    ConvolutionBnActivation(filters, (1, 1))
        ])
        # self.f_object = tf.keras.models.Sequential([
        #                                             ConvolutionBnActivation(filters, (1, 1)),
        #                                             ConvolutionBnActivation(filters, (1, 1))
        # ])
        # self.f_down = ConvolutionBnActivation(filters, (1, 1))
        self.f_up = ConvolutionBnActivation(filters, (1, 1))

        self.softmax = tf.keras.layers.Activation("softmax")
        self.upsampling2d = tf.keras.layers.UpSampling2D(size=scale, interpolation="bilinear")

    def call(self, feats, ctx, training=None):
        if K.image_data_format() == "channels_last":
            # feats-dim: (BS, H, W, C) & ctx-dim: (BS, C, C2)
            ctx = tf.keras.layers.Permute((2, 1))(ctx)
            BS, H, W, C = feats.shape
            if self.scale > 1:
                feats = self.pool(feats, training=training)

            query = self.f_pixel(feats, training=training)              # (BS, H, W, C)
            query = tf.keras.layers.Reshape((-1, C))(query)             # (BS, N, C)
            # key = self.f_object(ctx, training=training)                 # (BS, C2, C)
            key = tf.keras.layers.Reshape((-1, C))(ctx)                 # (BS, C2, C)
            # value = self.f_down(ctx, training=training)                 # (BS, C2, C)
            value = tf.keras.layers.Reshape((-1, C))(ctx)               # (BS, C2, C)

            sim_map = tf.linalg.matmul(query, key, transpose_b=True)    # (BS, N, C2)
            sim_map = (self.filters ** -0.5) * sim_map                  # (BS, N, C2)
            sim_map = self.softmax(sim_map)                             # (BS, N, C2)

            context = tf.linalg.matmul(sim_map, value)                   # (BS, N, C)
            context = tf.keras.layers.Reshape((H, W, C))(context)       # (BS, H, W, C)
            context = self.f_up(context, training=training)             # (BS, H, W, C)
            if self.scale > 1:
                context = self.upsampling2d(context)

        else:
            # feats-dim: (BS, C, H, W) & ctx-dim: (BS, C, C2)
            BS, C, H, W = feats.shape
            if self.scale > 1:
                feats = self.pool(feats, training=training)

            query = self.f_pixel(feats, training=training)              # (BS, C, H, W)
            query = tf.keras.layers.Reshape((C, -1))(query)             # (BS, C, N)
            # key = self.f_object(ctx, training=training)                 # (BS, C, C2)
            key = tf.keras.layers.Reshape((C, -1))(ctx)                 # (BS, C, C2)
            # value = self.f_down(ctx, training=training)                 # (BS, C, C2)
            value = tf.keras.layers.Reshape((C, -1))(ctx)               # (BS, C, C2)

            sim_map = tf.linalg.matmul(query, key, transpose_a=True)    # (BS, N, C2)
            sim_map = (self.filters ** -0.5) * sim_map                  # (BS, N, C2)
            sim_map = self.softmax(sim_map)                             # (BS, N, C2)

            context = tf.linalg.matmul(sim_map, value, transpose_b=True) # (BS, N, C)
            context = tf.keras.layers.Permute(2, 1)(context)            # (BS, C, N)
            context = tf.keras.layers.Reshape((C, H, W))(context)       # (BS, C, H, W)
            context = self.f_up(context, training=training)             # (BS, C, H, W)
            if self.scale > 1:
                context = self.upsampling2d(context)

        return context
        

class SpatialOCR_Module(tf.keras.layers.Layer):
    def __init__(self, filters, scale=1.0, dropout=0.1):
        super(SpatialOCR_Module, self).__init__()
        self.filters = filters
        self.scale = scale
        self.dropout = dropout

        self.object_attention = ObjectAttentionBlock2D(filters, scale)

        axis = 3 if K.image_data_format() == "channels_last" else 1
        self.concat = tf.keras.layers.Concatenate(axis=axis)
        self.conv1x1_bn_relu = ConvolutionBnActivation(filters, (1, 1))
        self.dropout = tf.keras.layers.Dropout(dropout)

    def call(self, features, ocr_context, training=None):
        # features-dim: (BS, H, W, C) & ocr_context-dim: (BS, C, C2) (if K.image_data_format() == "channels_last")
        context = self.object_attention(features, ocr_context, training=training)   # (BS, H, W, C)
        
        output = self.concat([context, features])                                   # (BS, H, W, 2*C)
        output = self.conv1x1_bn_relu(output, training=training)                    # (BS, H, W, C)
        output = self.dropout(output, training=training)                            # (BS, H, W, C)

        return output
        

class SpatialOCR_ASP_Module(tf.keras.layers.Layer):
    def __init__(self, filters, scale=1, dropout=0.1, dilations=(12, 24, 36)):
        super(SpatialOCR_ASP_Module, self).__init__()
        self.filters = filters
        self.scale = scale
        self.dropout = dropout
        self.dilations = dilations

        self.conv3x3_bn_relu_1 = ConvolutionBnActivation(filters, (3, 3))
        self.context = ObjectAttentionBlock2D(filters, scale)
        self.conv1x1_bn_relu_1 = ConvolutionBnActivation(filters, (1, 1))
        self.atrous_sepconv_bn_relu_1 = AtrousSeparableConvolutionBnReLU(dilation=dilations[0], filters=filters, kernel_size=3)
        self.atrous_sepconv_bn_relu_2 = AtrousSeparableConvolutionBnReLU(dilation=dilations[1], filters=filters, kernel_size=3)
        self.atrous_sepconv_bn_relu_3 = AtrousSeparableConvolutionBnReLU(dilation=dilations[2], filters=filters, kernel_size=3)
        
        self.spatial_context = SpatialGather_Module(scale=scale)

        self.axis = 3 if K.image_data_format() == "channels_last" else 1
        self.concat = tf.keras.layers.Concatenate(axis=self.axis)

        self.conv1x1_bn_relu_2 = ConvolutionBnActivation(filters, (1, 1))
        self.dropout = tf.keras.layers.Dropout(dropout)
    
    def call(self, x, probabilities, training=None):
        feat1 = self.conv3x3_bn_relu_1(x, training=training)
        context = self.spatial_context(feat1, probabilities, training=training)
        feat1 = self.context(feat1, context, training=training)
        feat2 = self.conv1x1_bn_relu_1(x, training=training)
        feat3 = self.atrous_sepconv_bn_relu_1(x, training=training)
        feat4 = self.atrous_sepconv_bn_relu_2(x, training=training)
        feat5 = self.atrous_sepconv_bn_relu_3(x, training=training)

        output = self.concat([feat1, feat2, feat3, feat4, feat5])
        output = self.conv1x1_bn_relu_2(output, training=training)
        output = self.dropout(output, training=training)

        return output


class AttCF_Module(tf.keras.layers.Layer):
    def __init__(self, filters):
        super(AttCF_Module, self).__init__()

        self.filters = filters

        self.softmax = tf.keras.layers.Activation("softmax")
        self.conv1x1 = tf.keras.layers.Conv2D(filters, (1, 1), padding="same")

    def call(self, aspp, coarse_x, training=None):
        if K.image_data_format() == "channels_last":
            BS, H, W, C = aspp.shape
            _, h, w, c = coarse_x.shape

            # CCB
            q = tf.keras.layers.Reshape((-1, c))(coarse_x)          # (BS, N, c)
            k = tf.keras.layers.Reshape((-1, C))(aspp)              # (BS, N, C)

            e = tf.linalg.matmul(q, k, transpose_a=True)            # (BS, c, C)
            e = tf.math.reduce_max(e, -1, keepdims=True)[0] - e     # (BS, c, C)
            att = self.softmax(e)                                   # (BS, c, C)

            # CAB
            v = tf.keras.layers.Reshape((-1, c))(coarse_x)          # (BS, N, c)
            output = tf.linalg.matmul(att, v, transpose_a=True, transpose_b=True) # (BS, C, N)
            output = tf.keras.layers.Permute((2, 1))(output)        # (BS, N, C)
            output = tf.keras.layers.Reshape((H, W, C))(output)     # (BS, H, W, C)

            output = self.conv1x1(output)                           # (BS, H, W, C)

        else:
            BS, C, H, W = aspp.shape
            bs, c, h, w = coarse_x.shape

            # CCB
            q = tf.keras.layers.Reshape((c, -1))(coarse_x)
            k = tf.keras.layers.Reshape((C, -1))(aspp)

            e = tf.linalg.matmul(q, k, transpose_b=True)            # (BS, c, C)
            e = tf.math.reduce_max(e, -1, keepdims=True)[0] - e     # (BS, c, C)
            att = self.softmax(e)                                   # (BS, c, C)

            # CAB
            v = tf.keras.layers.Reshape((c, -1))(coarse_x)          # (BS, c, N)
            output = tf.linalg.matmul(att, v, transpose_a=True)     # (BS, C, N)
            output = tf.keras.layers.Reshape((C, H, W))(output)

            output = self.conv1x1(output)

        return output


class BasicBlock(tf.keras.layers.Layer):
    def __init__(self, filters):
        super(BasicBlock, self).__init__()

        self.conv3x3_bn_relu = ConvolutionBnActivation(filters, (3, 3), momentum=0.1)
        self.conv3x3_bn = ConvolutionBnActivation(filters, (3, 3), momentum=0.1, post_activation="linear")

        self.relu = tf.keras.layers.Activation("relu")

    def call(self, input, training=None):
        residual = input

        out = self.conv3x3_bn_relu(input, training=training)
        out = self.conv3x3_bn(out, training=training)

        out = out + residual
        out = self.relu(out)

        return out


class BottleneckBlock(tf.keras.layers.Layer):
    def __init__(self, filters, downsample=False, expansion=4):
        super(BottleneckBlock, self).__init__()

        self.ds = downsample

        self.conv3x3_bn_relu_1 = ConvolutionBnActivation(filters, (1, 1), momentum=0.1)
        self.conv3x3_bn_relu_2 = ConvolutionBnActivation(filters, (3, 3), momentum=0.1)
        self.conv3x3_bn = ConvolutionBnActivation(filters * expansion, (1, 1), momentum=0.1, post_activation="linear")

        if downsample:
            self.downsample = ConvolutionBnActivation(filters * expansion, (1, 1), momentum=0.1)

        self.relu = tf.keras.layers.Activation("relu")

    def call(self, input, training=None):
        residual = input

        out = self.conv3x3_bn_relu_1(input, training=training)
        out = self.conv3x3_bn_relu_2(out, training=training)
        out = self.conv3x3_bn(out, training=training)

        if self.ds:
            residual = self.downsample(input)

        out = out + residual
        out = self.relu(out)

        return out


class HighResolutionModule(tf.keras.layers.Layer):
    def __init__(self, num_branches, blocks, filters):
        # filters_in unnecessary since it equals filters
        super(HighResolutionModule, self).__init__()

        self.num_branches = num_branches
        self.filters = filters
        self.num_in_channels = filters[0]

        self._check_branches(num_branches, blocks, filters)
        
        # Make Branches
        self.branch_1 = tf.keras.Sequential([BasicBlock(filters[0]), BasicBlock(filters[0]), BasicBlock(filters[0]), BasicBlock(filters[0])])
        self.branch_2 = tf.keras.Sequential([BasicBlock(filters[1]), BasicBlock(filters[1]), BasicBlock(filters[1]), BasicBlock(filters[1])])
        self.branch_3 = tf.keras.Sequential([BasicBlock(filters[2]), BasicBlock(filters[2]), BasicBlock(filters[2]), BasicBlock(filters[2])]) if num_branches >= 3 else None
        self.branch_4 = tf.keras.Sequential([BasicBlock(filters[3]), BasicBlock(filters[3]), BasicBlock(filters[3]), BasicBlock(filters[3])]) if num_branches >= 4 else None

        self.fuse_layers = self._make_fuse_layers()
        self.relu = tf.keras.layers.Activation("relu")

    def _check_branches(self, num_branches, blocks, filters):
        if num_branches != len(blocks):
            raise ValueError("'num_branches' = {} is not equal to length of 'blocks' = {}".format(num_branches, len(blocks)))
        
        if num_branches != len(filters):
            raise ValueError("'num_branches' = {} is not equal to length of 'filters' = {}".format(num_branches, len(filters)))

    def _make_fuse_layers(self):
        fuse_layers = []
        for i in range(self.num_branches):
            fuse_layer = []
            for j in range(self.num_branches):
                if j > i:
                    fuse_layer.append(ConvolutionBnActivation(self.filters[i], (1, 1), momentum=0.1, post_activation="linear"))
                elif j == i:
                    fuse_layer.append(None)
                else:
                    conv3x3s = []
                    for k in range(i - j):
                        if k == i - j - 1:
                            conv3x3s.append(ConvolutionBnActivation(self.filters[i], (3, 3), strides=(2, 2), momentum=0.1, post_activation="linear"))
                        else:
                            conv3x3s.append(ConvolutionBnActivation(self.filters[j], (3, 3), strides=(2, 2), momentum=0.1))
                    fuse_layer.append(tf.keras.Sequential(conv3x3s))

            fuse_layers.append(fuse_layer)

        return fuse_layers

    def call(self, input1, input2, input3, input4, training=None):
        x_1 = self.branch_1(input1, training=training)
        x_2 = self.branch_2(input2, training=training)
        x_3 = self.branch_3(input3, training=training) if self.num_branches >= 3 else None
        x_4 = self.branch_4(input4, training=training) if self.num_branches >= 4 else None
        
        x = [x_1, x_2]
        if x_3 is not None:
            x = [x_1, x_2, x_3]
        if x_4 is not None:
            x = [x_1, x_2, x_3, x_4]

        x_fuse = []
        for i in range(len(self.fuse_layers)):
            y = x[0] if i == 0 else self.fuse_layers[i][0](x[0])
            for j in range(1, self.num_branches):
                if i == j:
                    y += x[j]
                elif j > i:
                    f = self.fuse_layers[i][j](x[j])
                    scale_factor = int(x[i].shape[-2] / f.shape[-2])
                    if scale_factor > 1:
                        y += tf.keras.layers.UpSampling2D(size=scale_factor, interpolation="bilinear")(f)
                    else:
                        y += f
                else:
                    y += self.fuse_layers[i][j](x[j])
                
            x_fuse.append(self.relu(y))

        return x_fuse


custom_objects = {
    'ConvolutionBnActivation': ConvolutionBnActivation,
    'AtrousSeparableConvolutionBnReLU': AtrousSeparableConvolutionBnReLU,
    'AtrousSpatialPyramidPoolingV3': AtrousSpatialPyramidPoolingV3,
    'Upsample_x2_Block': Upsample_x2_Block,
    'Upsample_x2_Add_Block': Upsample_x2_Add_Block,
    'SpatialContextBlock': SpatialContextBlock,
    'FPNBlock': FPNBlock,
    'AtrousSpatialPyramidPoolingV1': AtrousSpatialPyramidPoolingV1,
    'Base_OC_Module': Base_OC_Module,
    'Pyramid_OC_Module': Pyramid_OC_Module,
    'ASP_OC_Module': ASP_OC_Module,
    'PAM_Module': PAM_Module,
    'CAM_Module': CAM_Module,
    'SelfAttentionBlock2D': SelfAttentionBlock2D,
}

tf.keras.utils.get_custom_objects().update(custom_objects)