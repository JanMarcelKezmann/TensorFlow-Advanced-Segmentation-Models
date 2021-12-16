import tensorflow as tf

from ._custom_layers_and_blocks import ConvolutionBnActivation, AtrousSeparableConvolutionBnReLU, AtrousSpatialPyramidPoolingV3
from ..backbones.tf_backbones import create_base_model

################################################################################
# DeepLabV3+
################################################################################
class DeepLabV3plus(tf.keras.Model):
    def __init__(self, n_classes, base_model, output_layers, height=None, width=None, filters=256,
                 final_activation="softmax", backbone_trainable=False,
                 output_stride=8, dilations=[6, 12, 18], **kwargs):
        super(DeepLabV3plus, self).__init__(**kwargs)

        self.n_classes = n_classes
        self.backbone = None
        self.filters = filters
        self.final_activation = final_activation
        self.output_stride = output_stride
        self.dilations = dilations
        self.height = height
        self.width = width


        if self.output_stride == 8:
            self.upsampling2d_1 = tf.keras.layers.UpSampling2D(size=2, interpolation="bilinear")
            output_layers = output_layers[:3]
            self.dilations = [2 * rate for rate in dilations]
        elif self.output_stride == 16:
            self.upsampling2d_1 = tf.keras.layers.UpSampling2D(size=4, interpolation="bilinear")
            output_layers = output_layers[:4]
            self.dilations = dilations
        else:
            raise ValueError("'output_stride' must be one of (8, 16), got {}".format(self.output_stride))

        base_model.trainable = backbone_trainable
        self.backbone = tf.keras.Model(inputs=base_model.input, outputs=output_layers)

        # Define Layers
        self.atrous_sepconv_bn_relu_1 = AtrousSeparableConvolutionBnReLU(dilation=2, filters=filters, kernel_size=3)
        self.atrous_sepconv_bn_relu_2 = AtrousSeparableConvolutionBnReLU(dilation=2, filters=filters, kernel_size=3)
        self.aspp = AtrousSpatialPyramidPoolingV3(self.dilations, filters)
        
        self.conv1x1_bn_relu_1 = ConvolutionBnActivation(filters, 1)
        self.conv1x1_bn_relu_2 = ConvolutionBnActivation(64, 1)

        self.upsample2d_1 = tf.keras.layers.UpSampling2D(size=2, interpolation="bilinear")
        self.upsample2d_2 = tf.keras.layers.UpSampling2D(size=4, interpolation="bilinear")

        self.concat = tf.keras.layers.Concatenate(axis=3)
        
        self.conv3x3_bn_relu_1 = ConvolutionBnActivation(filters, 3)
        self.conv3x3_bn_relu_2 = ConvolutionBnActivation(filters, 3)
        self.conv1x1_bn_sigmoid = ConvolutionBnActivation(self.n_classes, 1, post_activation="linear")

        self.final_activation = tf.keras.layers.Activation(final_activation)

    def call(self, inputs, training=None, mask=None):
        if training is None:
            training = True

        x = self.backbone(inputs)[-1]
        low_level_features = self.backbone(inputs)[1]
        
        # Encoder Module
        encoder = self.atrous_sepconv_bn_relu_1(x, training)
        encoder = self.aspp(encoder, training)
        encoder = self.conv1x1_bn_relu_1(encoder, training)
        encoder = self.upsample2d_1(encoder)

        # Decoder Module
        decoder_low_level_features = self.atrous_sepconv_bn_relu_2(low_level_features, training)
        decoder_low_level_features = self.conv1x1_bn_relu_2(decoder_low_level_features, training)

        decoder = self.concat([decoder_low_level_features, encoder])
        
        decoder = self.conv3x3_bn_relu_1(decoder, training)
        decoder = self.conv3x3_bn_relu_2(decoder, training)
        decoder = self.conv1x1_bn_sigmoid(decoder, training)

        decoder = self.upsample2d_2(decoder)
        decoder = self.final_activation(decoder)

        return decoder

    def model(self):
        x = tf.keras.layers.Input(shape=(self.height, self.width, 3))
        return tf.keras.Model(inputs=[x], outputs=self.call(x))