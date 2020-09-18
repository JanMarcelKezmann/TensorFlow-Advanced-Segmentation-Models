from keras_applications import get_submodules_from_kwargs

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

def DecoderUpsamplingX2AddBlock(filters, stage, use_batchnorm=False):
    up_name = 'decoder_stage{}_upsample'.format(stage)
    conv1_name = "decoder_stage{}_conv".format(stage)
    add_name = "decoder_stage{}_add".format(stage)

    def wrapper(input_tensor, skip):
        x  = layers.UpSampling2D(size=2, name=up_name)(input_tensor)
        skip = Conv1x1BnReLU(filters, use_batchnorm, name=conv1_name)(skip)
        x = layers.Add(name=add_name)([x, skip])

        return x

    return wrapper

def DecoderTransposeX2AddBlock(filters, stage, use_batchnorm=False):
    transp_name = 'decoder_stage{}_transpose'.format(stage)
    conv1_name = "decoder_stage{}_conv".format(stage)
    add_name = "decoder_stage{}_add".format(stage)

    def wrapper(input_tensor, skip):
        x = layers.Conv2DTranspose(
            filters,
            kernel_size=(4, 4),
            strides=(2, 2),
            padding='same',
            name=transp_name,
            use_bias=not use_batchnorm,
        )(input_tensor)
        skip = Conv1x1BnReLU(filters, use_batchnorm, name=conv1_name)(skip)
        x = layers.Add(name=add_name)([x, skip])

        return x

    return wrapper

# ---------------------------------------------------------------------
#  Unet Decoder
# ---------------------------------------------------------------------

def build_fcn(
        backbone,
        backbone_name,
        decoder_block,
        skip_connection_layers,
        decoder_filters=(256, 128, 64, 32, 16),
        n_upsample_blocks=3,
        classes=1,
        activation='sigmoid',
        use_batchnorm=True,
):
    input_ = backbone.input
    x = backbone.output

    # extract skip connections
    skips = ([backbone.get_layer(name=i).output if isinstance(i, str)
              else backbone.get_layer(index=i).output for i in skip_connection_layers])


    # add center block if previous operation was maxpooling (for vgg models)
    if isinstance(backbone.layers[-1], layers.MaxPooling2D):
        x = Conv3x3BnReLU(512, use_batchnorm, name='center_block1')(x)
        x = Conv3x3BnReLU(512, use_batchnorm, name='center_block2')(x)

    x = Conv1x1BnReLU(classes, use_batchnorm, name="center_block3")(x)
    
    
    for i in range(n_upsample_blocks):
        if i > 0 and i <= 2 and "vgg" in backbone_name:
            if i < len(skips):
                skip = skips[i]
            else:
                skip = None
            x = decoder_block(decoder_filters[i], stage=i, use_batchnorm=use_batchnorm)(x, skip)
        elif "vgg" not in backbone_name and i <= 1:
            if i < len(skips):
                skip = skips[i]
            else:
                skip = None
            x = decoder_block(decoder_filters[i], stage=i, use_batchnorm=use_batchnorm)(x, skip)

    # model head (define number of output classes)
    x = layers.UpSampling2D(size=8, name="final_x8_upsample")(x)
    x = Conv1x1BnActivation(classes, use_batchnorm, activation=activation, name=activation)(x)

    # create keras model instance
    model = models.Model(input_, x)

    return model

# ---------------------------------------------------------------------
#  Unet Model
# ---------------------------------------------------------------------

def FCN(
        backbone_name='vgg16',
        input_shape=(None, None, 3),
        classes=1,
        activation='sigmoid',
        weights=None,
        encoder_weights='imagenet',
        encoder_freeze=False,
        encoder_features='default',
        decoder_block_type='transpose',
        decoder_filters=(256, 128, 64, 32, 16),
        decoder_use_batchnorm=True,
        **kwargs
):
    """ FCN is a fully convolution neural network for image semantic segmentation
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

    if decoder_block_type == 'transpose':
        decoder_block = DecoderTransposeX2AddBlock
    elif decoder_block_type == "upsample":
        decode_block = DecoderUpsamplingX2AddBlock
    else:
        raise ValueError('Decoder block type should be in ("transpose"). '
                         'Got: {}'.format(decoder_block_type))

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

    model = build_fcn(
        backbone=backbone,
        backbone_name=backbone_name,
        decoder_block=decoder_block,
        skip_connection_layers=encoder_features,
        decoder_filters=decoder_filters,
        classes=classes,
        activation=activation,
        n_upsample_blocks=len(decoder_filters),
        use_batchnorm=decoder_use_batchnorm,
    )

    # lock encoder weights for fine-tuning
    if encoder_freeze:
        freeze_model(backbone, **kwargs)

    # loading model weights
    if weights is not None:
        model.load_weights(weights)

    return model