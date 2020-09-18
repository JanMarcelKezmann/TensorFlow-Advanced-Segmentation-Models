from keras_applications import get_submodules_from_kwargs
import tensorflow as tf

from ._common_blocks import Conv2dBn
from ._utils import freeze_model
from ..backbones.backbones_factory import Backbones

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

def Conv1x1BnReLU(filters, use_batchnorm, name=None):
    kwargs = get_submodules()

    def wrapper(input_tensor):
        return Conv2dBn(
            filters,
            kernel_size=1,
            activation='relu',
            kernel_initializer='he_uniform',
            padding='same',
            use_batchnorm=use_batchnorm,
            name=name,
            **kwargs
        )(input_tensor)

    return wrapper


def Conv1x1BnActivation(filters, use_batchnorm, activation="relu", name=None):
    kwargs = get_submodules()

    def wrapper(input_tensor):
        return Conv2dBn(
            filters,
            kernel_size=1,
            activation=activation,
            kernel_initializer='he_uniform',
            padding='same',
            use_batchnorm=use_batchnorm,
            name=name,
            **kwargs
        )(input_tensor)

    return wrapper


def Conv3x3BnReLU(filters, use_batchnorm, name=None):
    kwargs = get_submodules()

    def wrapper(input_tensor):
        return Conv2dBn(
            filters,
            kernel_size=3,
            activation='relu',
            kernel_initializer='he_uniform',
            padding='same',
            use_batchnorm=use_batchnorm,
            name=name,
            **kwargs
        )(input_tensor)

    return wrapper

def AtrousConv3x3BnReLU(atrous_filter, use_batchnorm, rate, name=None):
	kwargs = get_submodules()

	conv_name, act_name, bn_name = None, None, None
	block_name = kwargs.pop('name', None)
	backend, layers, models, keras_utils = get_submodules_from_kwargs(kwargs)

	if block_name is not None:
		conv_name = block_name + '_conv'

	if block_name is not None and activation is not None:
		if callable(activation):
			act_str = activation.__name__
		else:
			act_str = str(activation)
		act_name = block_name + '_' + act_str

	if block_name is not None and use_batchnorm:
		bn_name = block_name + '_bn'

	if backend.image_data_format() == 'channels_last':
		bn_axis = 3 
	else:
		bn_axis = 1

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

def aspp(atrous_filter, use_batchnorm, atrous_rates, name=None):
    kwargs = get_submodules()

    conv1x1_name = "conv1x1"
    conv3x3_rate1_name = "atrous_conv3x3_rate{}".format(atrous_rates[0])
    conv3x3_rate2_name = "atrous_conv3x3_rate{}".format(atrous_rates[1])
    conv3x3_rate3_name = "atrous_conv3x3_rate{}".format(atrous_rates[2])

    glob_avg_pooling_name = "global_average_pooling"
    pool_conv1x1_name = "pooling_conv1x1"
    pool_resize = "pooling_resize"

    def wrapper(input_tensor):
        print(input_tensor.shape[1:3])
        x1 = Conv1x1BnReLU(atrous_filter, use_batchnorm, name=conv1x1_name)(input_tensor)
        x3_r1 = AtrousConv3x3BnReLU(atrous_filter, use_batchnorm, rate=atrous_rates[0], name=conv3x3_rate1_name)(input_tensor)
        x3_r2 = AtrousConv3x3BnReLU(atrous_filter, use_batchnorm, rate=atrous_rates[1], name=conv3x3_rate2_name)(input_tensor)
        x3_r3 = AtrousConv3x3BnReLU(atrous_filter, use_batchnorm, rate=atrous_rates[2], name=conv3x3_rate3_name)(input_tensor)

        glob_avg_pool = layers.Lambda(lambda x: tf.reduce_mean(x, axis=[1, 2], keepdims=True, name=glob_avg_pooling_name))(input_tensor)
        glob_avg_pool = Conv1x1BnReLU(atrous_filter, use_batchnorm, name=pool_conv1x1_name)(glob_avg_pool)
        glob_avg_pool = layers.Lambda(lambda x: tf.image.resize_bilinear(x, input_tensor.shape[1:3], name=pool_resize))(glob_avg_pool)

        concat = layers.Concatenate(axis=3, name="concat")([x1, x3_r1, x3_r2, x3_r3, glob_avg_pool])

        return concat

    return wrapper


# ---------------------------------------------------------------------
#  Unet Decoder
# ---------------------------------------------------------------------

def build_deeplabv3plus(
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
    
    # extract skip connections
    skips = ([backbone.get_layer(name=i).output if isinstance(i, str)
              else backbone.get_layer(index=i).output for i in skip_connection_layers])    

    if output_stride == 8:
        if "vgg" in backbone_name:
            x = skips[2]
            low_level_features = skips[3]
        else:
            x = skips[1]
            low_level_features = skips[2]
        atrous_rates = [2 * rate for rate in atrous_rates]
    elif output_stride == 16:
        if "vgg" in backbone_name:
            x = skips[1]
            low_level_features = skips[3]
        else:
            x = skips[0]
            low_level_features = skips[2]
    else:
        raise ValueError("output_stride must be either 8 or 16.")

    # Encoder Module
    encoder = AtrousConv3x3BnReLU(atrous_filter, use_batchnorm, rate=2, name="atrous_block0_conv3x3_rate2")(x)
    encoder = aspp(atrous_filter, use_batchnorm, atrous_rates)(encoder)
    encoder = Conv1x1BnReLU(256, use_batchnorm, name="encoder_final_conv1x1")(encoder)

    if output_stride == 8:
        encoder_resize = layers.UpSampling2D(size=2, interpolation="bilinear", name="resize_encoder_outputx2")(encoder)
    else:
        encoder_resize = layers.UpSampling2D(size=4, interpolation="bilinear", name="resize_encoder_outputx4")(encoder)
    

    # model = models.Model(input_, encoder_resize)
    # print(model.summary())

    # Decoder Module
    decoder_low_level_features = AtrousConv3x3BnReLU(atrous_filter, use_batchnorm, rate=2, name="decoder_atrous_block_conv3x3_rate2")(low_level_features)
    decoder_low_level_features = Conv1x1BnReLU(64, use_batchnorm, name="decoder_conv1x1")(decoder_low_level_features)
    
    decoder = layers.Concatenate(axis=3, name="encoder_decoder_concat")([decoder_low_level_features, encoder_resize])
    # model = models.Model(input_, decoder)
    # print(model.summary())
    decoder = Conv3x3BnReLU(256, use_batchnorm, name="decoder_conv3x3_1")(decoder)
    decoder = Conv3x3BnReLU(256, use_batchnorm, name="decoder_conv3x3_2")(decoder)
    decoder = Conv1x1BnActivation(classes, use_batchnorm, activation=activation, name=activation)(decoder)
    
    decoder = layers.UpSampling2D(size=4, interpolation="bilinear", name="final_x4_upsample")(decoder)
    
    # create keras model instance
    model = models.Model(input_, decoder)

    return model

# ---------------------------------------------------------------------
#  Unet Model
# ---------------------------------------------------------------------

def DeepLabV3plus(
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
    """ DeeplabV3plus is a fully convolution neural network for image semantic segmentation
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


    model = build_deeplabv3plus(
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