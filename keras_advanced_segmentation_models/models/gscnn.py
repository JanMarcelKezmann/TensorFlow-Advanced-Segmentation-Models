from keras_applications import get_submodules_from_kwargs

from ._common_blocks import Conv2dBn
from ._utils import freeze_model
from ..backbones.backbones_factory import Backbones

import collections
import tensorflow as tf

backend = None
layers = None
models = None
keras_utils = None


# ---------------------------------------------------------------------
#  Utility functions
# ---------------------------------------------------------------------

def get_submodules():
    return {
        'backend': backend,
        'models': models,
        'layers': layers,
        'utils': keras_utils,
    }


# ---------------------------------------------------------------------
#  Blocks
# ---------------------------------------------------------------------
def Conv1x1BnActivation(filters, use_batchnorm, strides=(1, 1), activation="relu", name=None):
    kwargs = get_submodules()

    def wrapper(input_tensor):
        return Conv2dBn(
            filters,
            kernel_size=1,
            activation=activation,
            kernel_initializer='he_uniform',
            padding='same',
            strides=strides,
            use_batchnorm=use_batchnorm,
            name=name,
            **kwargs
        )(input_tensor)

    return wrapper

def ConvBnActivation(filters, use_batchnorm, kernel_size=1, strides=(1, 1), activation="relu", name=None):
    kwargs = get_submodules()

    def wrapper(input_tensor):
        return Conv2dBn(
            filters,
            kernel_size=kernel_size,
            activation=activation,
            kernel_initializer='he_uniform',
            padding='same',
            strides=strides,
            use_batchnorm=use_batchnorm,
            name=name,
            **kwargs
        )(input_tensor)

    return wrapper

def CannyBlock():
    lambda_name1 = "canny_block_lambda1"
    lambda_name2 = "canny_block_lambda2"
    lambda_name3 = "canny_block_lambda3"
    lambda_name4 = "canny_block_lambda4"
    lambda_name5 = "canny_block_lambda5"
    lambda_name6 = "canny_block_lambda6"
    lambda_name7 = "canny_block_lambda7"
    lambda_name8 = "canny_block_lambda8"
    lambda_name9 = "canny_block_lambda9"
    lambda_name10 = "canny_block_lambda10"

    def wrapper(input_tensor):
        # grad_components = tf.image.sobel_edges(img)
        # grad_mag_components = tf.math.square(grad_components)
        # grad_mag_square = tf.math.reduce_sum(grad_mag_components,axis=-1) # sum all magnitude components
        # grad_mag_img = tf.sqrt(grad_mag_square) 
        # sobel = tf.squeeze(grad_mag_img)
        # sobel0 = np.round(clip_0_1(sobel[...,0]/4+0.5) * 255.0, 0)
        # sobel1 = np.round(clip_0_1(sobel[...,1]/4+0.5) * 255.0, 0)
        # sobel0 = 1.0 * (sobel0 > 160)
        # sobel1 = 1.0 * (sobel1 > 160)
        # sobel_fin = sobel0 + sobel1
        # sobel_fin = 1.0 * (sobel_fin > 0.9)
        x = layers.Lambda(lambda x: tf.image.sobel_edges(x), name=lambda_name1)(input_tensor)
        x = layers.Lambda(lambda x: tf.math.square(x), name=lambda_name2)(x)
        x = layers.Lambda(lambda x: tf.math.reduce_sum(x, axis=-1), name=lambda_name3)(x)
        sobel = layers.Lambda(lambda x: tf.math.sqrt(x), name=lambda_name4)(x)
        sobel0 = layers.Lambda(lambda x: tf.clip_by_value(x[...,0]/4+0.5, clip_value_min=0.0, clip_value_max=1.0), name=lambda_name5)(sobel)
        sobel1 = layers.Lambda(lambda x: tf.clip_by_value(x[...,1]/4+0.5, clip_value_min=0.0, clip_value_max=1.0), name=lambda_name6)(sobel)
        sobel0 = layers.Lambda(lambda x: (tf.math.sign(x - 0.63) + 1.0) / 2.0, name=lambda_name7)(sobel0)
        sobel1 = layers.Lambda(lambda x: (tf.math.sign(x - 0.63) + 1.0) / 2.0, name=lambda_name8)(sobel1)
        sobel_fin = layers.Lambda(lambda x: x[0] + x[1], name=lambda_name9)([sobel0, sobel1])
        sobel_fin = layers.Lambda(lambda x: (tf.math.sign(x - 1.1) + 1.0) / 2.0, name=lambda_name10)(sobel_fin)

        return sobel_fin

    return wrapper

def ResNetBlock(filters, use_batchnorm, stage, strides=(1, 1), activation="relu", downsample=False, name=None):
    conv0_name = "ss_block{}_conv0".format(stage)
    conv1_name = "ss_block{}_conv1".format(stage)
    add_name = "ss_block{}_add".format(stage)
    activation_name = "ss_block{}_{}".format(stage, activation)

    def wrapper(input_tensor):
        res = input_tensor

        x = ConvBnActivation(filters, use_batchnorm, kernel_size=3, strides=strides, activation="relu", name=conv0_name)(input_tensor)
        x = ConvBnActivation(filters, use_batchnorm, kernel_size=3, strides=strides, activation=None, name=conv1_name)(x)

        if downsample is not None:
            residual = layers.MaxPooling2D(name="block{}_downsamplex2".format(stage))(x)

        x = layers.Add(name=add_name)([x, res])
        x = layers.Activation("relu", name=activation_name)(x)

        return x

    return wrapper

def _pair(x):

    def _ntuple(n):
        def parse(x):
            if isinstance(x, collections.abc.Iterable):
                return x
            return tuple(repeat(x, n))
        return parse

    return _ntuple(2)

def GCL(filters, kernel_size, stage, strides=(1, 1), name=None):
    kwargs = get_submodules()
    
    concat_name = "ss_block{}_GCL_concat".format(stage)
    bn_name = "ss_block{}_GCL_bn".format(stage)
    conv0_name = "ss_block{}_GCL_conv0".format(stage)
    conv1_name = "ss_block{}_GCL_conv1".format(stage)
    conv2_name = "ss_block{}_GCL_conv2".format(stage)
    add_name = "ss_block{}_GCL_add".format(stage)

    bn_axis = 3 if backend.image_data_format() == 'channels_last' else 1

    def wrapper(input_tensor, gate_features):
        alpha = layers.Concatenate(axis=bn_axis, name=concat_name)([input_tensor, gate_features])
        alpha = layers.BatchNormalization(axis=bn_axis, name=bn_name)(alpha)

        alpha = ConvBnActivation(filters, kernel_size=1, use_batchnorm=False, strides=strides, activation="relu", name=conv0_name)(alpha)
        alpha = ConvBnActivation(filters, kernel_size=1, use_batchnorm=True, strides=strides, activation="sigmoid", name=conv1_name)(alpha)

        x = layers.Multiply()([input_tensor, alpha])
        x = layers.Add(name=add_name)([input_tensor, x])

        x = ConvBnActivation(filters, kernel_size=1, use_batchnorm=False, strides=strides, activation=None, name=conv2_name)(x)

        return x

    return wrapper

def AtrousConv3x3BnReLU(atrous_filter, use_batchnorm, rate, name=None):
    kwargs = get_submodules()

    def wrapper(input_tensor):
        x = Conv2dBn(
            atrous_filter,
            kernel_size=3,
            activation='relu',
            kernel_initializer='he_uniform',
            padding='same',
            dilation_rate=rate,
            use_batchnorm=use_batchnorm,
            name=name,
            **kwargs
        )(input_tensor)
        return x

    return wrapper

def aspp(atrous_filter, use_batchnorm, atrous_rates, upsample_size, name=None):
    kwargs = get_submodules()

    up_name = "aspp_upsampling{}x".format(upsample_size)

    conv1x1_name1 = "aspp1_conv1x1"
    conv3x3_rate1_name1 = "atrous1_conv3x3_rate{}".format(atrous_rates[0])
    conv3x3_rate2_name1 = "atrous1_conv3x3_rate{}".format(atrous_rates[1])
    conv3x3_rate3_name1 = "atrous1_conv3x3_rate{}".format(atrous_rates[2])

    glob_avg_pooling_name1 = "global_average_pooling1"
    pool_conv1x1_name1 = "pooling1_conv1x1"
    pool_resize1 = "pooling1_resize"

    conv1x1_name2 = "aspp2_conv1x1"
    conv3x3_rate1_name2 = "atrous2_conv3x3_rate{}".format(atrous_rates[0])
    conv3x3_rate2_name2 = "atrous2_conv3x3_rate{}".format(atrous_rates[1])
    conv3x3_rate3_name2 = "atrous2_conv3x3_rate{}".format(atrous_rates[2])

    glob_avg_pooling_name2 = "global_average_pooling2"
    pool_conv1x1_name2 = "pooling2_conv1x1"
    pool_resize2 = "pooling2_resize"

    def wrapper(input_tensor, ss):
        print(input_tensor.shape[1:3])
        input_tensor = layers.UpSampling2D(size=upsample_size, name=up_name)(input_tensor)
        ### add missing features see line 138 to 151 in gscnn.py
        x1_1 = Conv1x1BnActivation(atrous_filter, use_batchnorm, activation="relu", name=conv1x1_name1)(input_tensor)
        x3_r1_1 = AtrousConv3x3BnReLU(atrous_filter, use_batchnorm, rate=atrous_rates[0], name=conv3x3_rate1_name1)(input_tensor)
        x3_r2_1 = AtrousConv3x3BnReLU(atrous_filter, use_batchnorm, rate=atrous_rates[1], name=conv3x3_rate2_name1)(input_tensor)
        x3_r3_1 = AtrousConv3x3BnReLU(atrous_filter, use_batchnorm, rate=atrous_rates[2], name=conv3x3_rate3_name1)(input_tensor)

        glob_avg_pool1 = layers.Lambda(lambda x: tf.reduce_mean(x, axis=[1, 2], keepdims=True, name=glob_avg_pooling_name1))(input_tensor)
        glob_avg_pool1 = Conv1x1BnActivation(atrous_filter, use_batchnorm, activation="relu", name=pool_conv1x1_name1)(glob_avg_pool1)
        glob_avg_pool1 = layers.Lambda(lambda x: tf.image.resize_bilinear(x, input_tensor.shape[1:3], name=pool_resize1))(glob_avg_pool1)

        x1_2 = Conv1x1BnActivation(atrous_filter, use_batchnorm, activation="relu", name=conv1x1_name2)(ss)
        x3_r1_2 = AtrousConv3x3BnReLU(atrous_filter, use_batchnorm, rate=atrous_rates[0], name=conv3x3_rate1_name2)(ss)
        x3_r2_2 = AtrousConv3x3BnReLU(atrous_filter, use_batchnorm, rate=atrous_rates[1], name=conv3x3_rate2_name2)(ss)
        x3_r3_2 = AtrousConv3x3BnReLU(atrous_filter, use_batchnorm, rate=atrous_rates[2], name=conv3x3_rate3_name2)(ss)

        glob_avg_pool2 = layers.Lambda(lambda x: tf.reduce_mean(x, axis=[1, 2], keepdims=True, name=glob_avg_pooling_name2))(ss)
        glob_avg_pool2 = Conv1x1BnActivation(atrous_filter, use_batchnorm, activation="relu", name=pool_conv1x1_name2)(glob_avg_pool2)
        glob_avg_pool2 = layers.Lambda(lambda x: tf.image.resize_bilinear(x, input_tensor.shape[1:3], name=pool_resize2))(glob_avg_pool2)

        concat = layers.Concatenate(axis=3, name="concat")([x1_1, x3_r1_1, x3_r2_1, x3_r3_1, glob_avg_pool1, x1_2, x3_r1_2, x3_r2_2, x3_r3_2, glob_avg_pool2])

        return concat

    return wrapper


# ---------------------------------------------------------------------
#  Unet Decoder
# ---------------------------------------------------------------------

def build_gscnn(
        backbone,
        backbone_name,
        skip_connection_layers,
        atrous_filter=256,
        classes=1,
        activation='sigmoid',
        use_batchnorm=True,
        output_stride=8,
        atrous_rates=[6, 12, 18]
    ):
    input_ = backbone.input
    h, w, c = int(input_.shape[1]), int(input_.shape[2]), int(input_.shape[3])
    x = backbone.output
    
    # extract skip connections
    skips = ([backbone.get_layer(name=i).output if isinstance(i, str)
              else backbone.get_layer(index=i).output for i in skip_connection_layers])    

    s0 = layers.UpSampling2D(size=2, interpolation="bilinear", name="block0_upsample2x")(skips[3])
    s1 = layers.UpSampling2D(size=8, interpolation="bilinear", name="block1_upsample8x")(Conv1x1BnActivation(filters=256, use_batchnorm=False, strides=(1, 1), activation=None, name="block1_conv1x1")(skips[1]))
    s2 = layers.UpSampling2D(size=16, interpolation="bilinear", name="block2_upsample16x")(Conv1x1BnActivation(filters=512, use_batchnorm=False, strides=(1, 1), activation=None, name="block2_conv1x1")(skips[0]))
    s3 = layers.UpSampling2D(size=32, interpolation="bilinear", name="block3_upsample32x")(Conv1x1BnActivation(filters=2048, use_batchnorm=False, strides=(1, 1), activation=None, name="block3_conv1x1")(x))

    canny = CannyBlock()(input_)
    canny = layers.Reshape(target_shape=(h, w, 1))(canny)

    ss = ResNetBlock(64, use_batchnorm, stage=0, strides=1, downsample=None)(s0)

    # ss = layers.UpSampling2D(size=2, interpolation="bilinear", name="ss_block0_upsample2x")(ss)

    ss = layers.Conv2D(filters=32, kernel_size=1, name="ss_block1_conv1x1")(ss)
    ss = GCL(filters=32, kernel_size=1, stage=1)(ss, s1)
    ss = ResNetBlock(32, use_batchnorm, stage=1, strides=1, downsample=None)(ss)
    # ss = layers.UpSampling2D(size=2, interpolation="bilinear", name="ss_block1_upsample2x")(ss)

    ss = layers.Conv2D(filters=16, kernel_size=1, name="ss_block2_conv1x1")(ss)
    ss = GCL(filters=16, kernel_size=1, stage=2)(ss, s2)
    ss = ResNetBlock(16, use_batchnorm, stage=2, strides=1, downsample=None)(ss)
    # ss = layers.UpSampling2D(size=2, interpolation="bilinear", name="ss_block2_upsample2x")(ss)

    ss = layers.Conv2D(filters=8, kernel_size=1, name="ss_block3_conv1x1")(ss)
    ss = GCL(filters=8, kernel_size=1, stage=3)(ss, s3)
    ss = layers.Conv2D(filters=1, kernel_size=1, name="ss_block3_final_conv")(ss)
    # ss = layers.UpSampling2D(size=2, interpolation="bilinear", name="ss_block3_upsample2x")(ss)

    edge_out = layers.Activation("sigmoid", name="ss_sigmoid")(ss)
    ss = layers.Concatenate(axis=3, name="ss_concat")([edge_out, canny])
    ss = ConvBnActivation(filters=1, kernel_size=1, use_batchnorm=False, strides=(1, 1), activation="sigmoid", name="ss_conv1x1")(ss)

    # ASPP Module
    if output_stride == 8:
        if "vgg" in backbone_name:
            x = skips[2]
        else:
            x = skips[1]
        atrous_rates = [2 * rate for rate in atrous_rates]
    elif output_stride == 16:
        if "vgg" in backbone_name:
            x = skips[1]
        else:
            x = skips[0]
    else:
        raise ValueError("output_stride must be either 8 or 16.")

    aspp_features = aspp(atrous_filter, use_batchnorm, atrous_rates, upsample_size=output_stride)(x, ss)
    aspp_features = ConvBnActivation(filters=256, kernel_size=1, use_batchnorm=False, activation=None, name="aspp_features_conv")(aspp_features)

    refine = ConvBnActivation(filters=48, kernel_size=1, use_batchnorm=False, activation="relu", name="refine_conv")(skips[3])
    refine = layers.UpSampling2D(size=2, name="refine_upsample2x")(refine)
    refine = layers.Concatenate(axis=3, name="refine_concat")([aspp_features, refine])

    refine = ConvBnActivation(filters=256, kernel_size=3, use_batchnorm=True, activation="relu", name="refine_conv2")(refine)
    refine = ConvBnActivation(filters=256, kernel_size=3, use_batchnorm=True, activation="relu", name="refine_conv3")(refine)
    refine = ConvBnActivation(filters=classes, kernel_size=1, use_batchnorm=False, activation=None, name="refine_conv4")(refine)
    
    # create keras model instance
    model = models.Model(input_, [edge_out, refine])

    return model

# ---------------------------------------------------------------------
#  Unet Model
# ---------------------------------------------------------------------

def GSCNN(
        backbone_name='vgg16',
        input_shape=(None, None, 3),
        classes=1,
        activation='sigmoid',
        weights=None,
        encoder_weights='imagenet',
        encoder_freeze=False,
        encoder_features='default',
        atrous_filter=256,
        atrous_use_batchnorm=True,
        atrous_rates=[6, 12, 18],
        output_stride=8,
        **kwargs
):
    """ Gated Shape CNN is a fully convolution neural network for image semantic segmentation
    Args:
        backbone_name: name of classification model (without last dense layers) used as feature
            extractor to build segmentation model.
        input_shape: shape of input data/image ``(H, W, C)``, in general
            case you do not need to set ``H`` and ``W`` shapes, just pass ``(None, None, C)`` to make your model be
            able to process images af any size, but ``H`` and ``W`` of input images should be divisible by factor ``32``.
        classes: a number of classes for output (output shape - ``(h, w, classes)``).
        activation: name of one of ``keras.activations`` for last model layer
            (e.g. ``sigmoid``, ``softmax``, ``linear``).
        weights: optional, path to model weights.
        encoder_weights: one of ``None`` (random initialization), ``imagenet`` (pre-training on ImageNet).
        encoder_freeze: if ``True`` set all layers of encoder (backbone model) as non-trainable.
        encoder_features: a list of layer numbers or names starting from top of the model.
            Each of these layers will be concatenated with corresponding decoder block. If ``default`` is used
            layer names are taken from ``DEFAULT_SKIP_CONNECTIONS``.
        decoder_block_type: one of blocks with following layers structure:
            - `upsampling`:  ``UpSampling2D`` -> ``Conv2D`` -> ``Conv2D``
            - `transpose`:   ``Transpose2D`` -> ``Conv2D``
        decoder_filters: list of numbers of ``Conv2D`` layer filters in decoder blocks
        decoder_use_batchnorm: if ``True``, ``BatchNormalisation`` layer between ``Conv2D`` and ``Activation`` layers
            is used.
    Returns:
        ``keras.models.Model``: **Unet**
    .. _Unet:
        https://arxiv.org/pdf/1505.04597
    """

    global backend, layers, models, keras_utils
    backend, layers, models, keras_utils = get_submodules_from_kwargs(kwargs)


    backbone = Backbones.get_backbone(
        backbone_name,
        input_shape=input_shape,
        weights=encoder_weights,
        include_top=False,
        **kwargs,
    )

    if encoder_features == 'default':
        encoder_features = Backbones.get_feature_layers(backbone_name, n=4)

    if "vgg" in backbone_name:
        encoder_features = ['block5_pool', 'block4_pool', 'block3_pool', 'block2_pool']


    model = build_gscnn(
        backbone=backbone,
        backbone_name=backbone_name,
        skip_connection_layers=encoder_features,
        atrous_filter=atrous_filter,
        classes=classes,
        activation=activation,
        use_batchnorm=atrous_use_batchnorm,
        output_stride=output_stride,
        atrous_rates=atrous_rates
    )

    # lock encoder weights for fine-tuning
    if encoder_freeze:
        freeze_model(backbone, **kwargs)

    # loading model weights
    if weights is not None:
        model.load_weights(weights)

    return model