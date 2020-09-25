import tensorflow as tf
import tensorflow.keras.backend as K

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
    def __init__(self, filters, trainable=None):
        super(Upsample_x2_Block, self).__init__()
        self.trainable = trainable

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

class SelfAttentionBlock2D(tf.keras.layers.Layer):
    def __init__(self, filters):
        super(SelfAttentionBlock2D, self).__init__()

        self.filters = filters

        self.conv1x1_1 = tf.keras.layers.Conv2D(filters // 8, (1, 1), padding="same")
        self.conv1x1_2 = tf.keras.layers.Conv2D(filters // 8, (1, 1), padding="same")
        self.conv1x1_3 = tf.keras.layers.Conv2D(filters // 2, (1, 1), padding="same")
        self.conv1x1_4 = tf.keras.layers.Conv2D(filters, (1, 1), padding="same")
        
        self.softmax_activation = tf.keras.layers.Activation("softmax")

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
        gamma = tf.Variable(0.0, trainable=training, name="gamma")

        o = tf.reshape(o, shape=(x.shape[0], x.shape[1], x.shape[2], x.shape[3] // 2))
        o = self.conv1x1_4(o, training=training)
        x = gamma * o + x

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

        self.softmax = tf.keras.layers.Activation("softmax")

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

        gamma = tf.Variable(0.0, trainable=training, name="pam_gamma")
        out = gamma * out + x

        return out

class CAM_Module(tf.keras.layers.Layer):
    def __init__(self, filters):
        super(CAM_Module, self).__init__()

        self.filters = filters

        self.softmax = tf.keras.layers.Activation("softmax")

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

        gamma = tf.Variable(0.0, trainable=training, name="cam_gamma")
        out = gamma * out + x

        return out
