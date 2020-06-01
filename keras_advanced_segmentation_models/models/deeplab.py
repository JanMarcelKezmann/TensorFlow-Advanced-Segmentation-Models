from keras_applications import get_submodules_from_kwargs
import tensorflow as tf
# from keras_contrib.layers.crf import CRF
# from tensorflow.contrib.crf.python.ops import crf


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

def AtrousConv3x3BnReLU(filters, use_batchnorm, rate, name=None):
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
		# x = tf.nn.atrous_conv2d(
		# 	value=input_tensor,
		#     filters=filters,
		#     rate=rate,
		#     padding="same",
		#     name=name
		# )(input_tensor)
		x = Conv2dBn(
            filters,
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

def aspp(filters, use_batchnorm, name=None):
	kwargs = get_submodules()
	stage = 1
	conv1x1_name = 'decoder_stage{}_conv1x1'.format(stage)
	conv3x3_rate6_name = "decoder_stage{}_conv3x3_rate6".format(stage)
	conv3x3_rate12_name = "decoder_stage{}_conv3x3_rate12".format(stage)
	conv3x3_rate18_name = "decoder_stage{}_conv3x3_rate18".format(stage)
	conv3x3_rate24_name = "decoder_stage{}_conv3x3_rate24".format(stage)

	def wrapper(input_tensor):
		x1 = Conv1x1BnReLU(filters, use_batchnorm, name=conv1x1_name)(input_tensor)
		x3_r6 = AtrousConv3x3BnReLU(filters, use_batchnorm, rate=6, name=conv3x3_rate6_name)(input_tensor)
		x3_r12 = AtrousConv3x3BnReLU(filters, use_batchnorm, rate=12, name=conv3x3_rate12_name)(input_tensor)
		x3_r18 = AtrousConv3x3BnReLU(filters, use_batchnorm, rate=18, name=conv3x3_rate18_name)(input_tensor)
		x3_r24 = AtrousConv3x3BnReLU(filters, use_batchnorm, rate=24, name=conv3x3_rate24_name)(input_tensor)

		concat = layers.Concatenate()([x1, x3_r6, x3_r12, x3_r18, x3_r24])

		return concat

	return wrapper


from keras.layers import Layer
import keras.backend as K


class CRF(Layer):
    """纯Keras实现CRF层
    CRF层本质上是一个带训练参数的loss计算层，因此CRF层只用来训练模型，
    而预测则需要另外建立模型。
    """
    def __init__(self, ignore_last_label=False, **kwargs):
        """ignore_last_label：定义要不要忽略最后一个标签，起到mask的效果
        """
        self.ignore_last_label = 1 if ignore_last_label else 0
        super(CRF, self).__init__(**kwargs)

    def build(self, input_shape):
        self.num_labels = input_shape[-1] - self.ignore_last_label
        self.trans = self.add_weight(name='crf_trans',
                                     shape=(self.num_labels, self.num_labels),
                                     initializer='glorot_uniform',
                                     trainable=True)

    def log_norm_step(self, inputs, states):
        """递归计算归一化因子
        要点：1、递归计算；2、用logsumexp避免溢出。
        技巧：通过expand_dims来对齐张量。
        """
        inputs, mask = inputs[:, :-1], inputs[:, -1:]
        states = K.expand_dims(states[0], 2)  # (batch_size, output_dim, 1)
        trans = K.expand_dims(self.trans, 0)  # (1, output_dim, output_dim)
        outputs = K.logsumexp(states + trans, 1)  # (batch_size, output_dim)
        outputs = outputs + inputs
        outputs = mask * outputs + (1 - mask) * states[:, :, 0]
        return outputs, [outputs]

    def path_score(self, inputs, labels):
        """计算目标路径的相对概率（还没有归一化）
        要点：逐标签得分，加上转移概率得分。
        技巧：用“预测”点乘“目标”的方法抽取出目标路径的得分。
        """
        point_score = K.sum(K.sum(inputs * labels, 2), 1, keepdims=True)  # 逐标签得分
        labels1 = K.expand_dims(labels[:, :-1], 3)
        labels2 = K.expand_dims(labels[:, 1:], 2)
        labels = labels1 * labels2  # 两个错位labels，负责从转移矩阵中抽取目标转移得分
        trans = K.expand_dims(K.expand_dims(self.trans, 0), 0)
        trans_score = K.sum(K.sum(trans * labels, [2, 3]), 1, keepdims=True)
        return point_score + trans_score  # 两部分得分之和

    def call(self, inputs):  # CRF本身不改变输出，它只是一个loss
        return inputs

    def loss(self, y_true, y_pred):  # 目标y_pred需要是one hot形式
        if self.ignore_last_label:
            mask = 1 - y_true[:, :, -1:]
        else:
            mask = K.ones_like(y_pred[:, :, :1])
        y_true, y_pred = y_true[:, :, :self.num_labels], y_pred[:, :, :self.num_labels]
        path_score = self.path_score(y_pred, y_true)  # 计算分子（对数）
        init_states = [y_pred[:, 0]]  # 初始状态
        y_pred = K.concatenate([y_pred, mask])
        log_norm, _, _ = K.rnn(self.log_norm_step, y_pred[:, 1:], init_states)  # 计算Z向量（对数）
        log_norm = K.logsumexp(log_norm, 1, keepdims=True)  # 计算Z（对数）
        return log_norm - path_score  # 即log(分子/分母)

    def accuracy(self, y_true, y_pred):  # 训练过程中显示逐帧准确率的函数，排除了mask的影响
        mask = 1 - y_true[:, :, -1] if self.ignore_last_label else None
        y_true, y_pred = y_true[:, :, :self.num_labels], y_pred[:, :, :self.num_labels]
        isequal = K.equal(K.argmax(y_true, 2), K.argmax(y_pred, 2))
        isequal = K.cast(isequal, 'float32')
        if mask == None:
            return K.mean(isequal)
        else:
            return K.sum(isequal * mask) / K.sum(mask)


# ---------------------------------------------------------------------
#  Unet Decoder
# ---------------------------------------------------------------------

def build_deeplab(
        backbone,
        backbone_name,
        skip_connection_layers,
        decoder_filters=(256, 128, 64, 32, 16),
        n_upsample_blocks=3,
        classes=1,
        activation='sigmoid',
        use_batchnorm=True,
):
    input_ = backbone.input
    _, height, width, dims = backbone.input_shape
    print(height, width, dims)
    
    # x = backbone.output

    # extract skip connections
    skips = ([backbone.get_layer(name=i).output if isinstance(i, str)
              else backbone.get_layer(index=i).output for i in skip_connection_layers])    

    x = skips[2]

    model = models.Model(input_, x)
    print(model.summary())
    
    x = aspp(decoder_filters[3], use_batchnorm)(x)

    # model head (define number of output classes)
    x = layers.UpSampling2D(size=8, name="final_x8_upsample")(x)
    x = Conv1x1BnActivation(classes, use_batchnorm, activation=activation, name=activation)(x)
    # x = CRF()(x)
    # x = CrfRnnLayer(
    #     image_dims=(height, width),
    #     num_classes=classes,
    #     theta_alpha=160,
    #     theta_beta=3,
    #     theta_gamma=3,
    #     num_iteration=10,
    #     name="crfrnn")([x, input_])

    # create keras model instance
    model = models.Model(input_, x)

    return model

# ---------------------------------------------------------------------
#  Unet Model
# ---------------------------------------------------------------------

def DeepLab(
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
    """ Deeplab is a fully convolution neural network for image semantic segmentation
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


    model = build_deeplab(
        backbone=backbone,
        backbone_name=backbone_name,
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