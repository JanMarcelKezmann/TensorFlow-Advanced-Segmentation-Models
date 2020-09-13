import tensorflow as tf

from ._custom_layers_and_blocks import ConvolutionBnActivation, AtrousSeparableConvolutionBnReLU, AtrousSpatialPyramidPooling
from ..backbones.tf_backbones import create_backbone

class DeepLabV3plus(tf.keras.Model):
    def __init__(self, n_classes, backbone, filters, input_shape=[320, 320, 3],
                 final_activation="softmax", output_stride=8, dilations=[6, 12, 18], **kwargs):
        super(DeepLabV3plus, self).__init__(**kwargs)

        self.n_classes = n_classes
        self.backbone = backbone
        # self.skip_connection_layers = skip_connection_layers,
        self.filters = filters
        self.final_activation = final_activation
        self.output_stride = output_stride
        self.dilations = dilations
        self.inp_shape = input_shape

        # Define Layers
        self.atrous_sepconv_bn_relu_1 = AtrousSeparableConvolutionBnReLU(dilation=2, filters=filters, kernel_size=3)
        self.atrous_sepconv_bn_relu_2 = AtrousSeparableConvolutionBnReLU(dilation=2, filters=filters, kernel_size=3)
        self.aspp = AtrousSpatialPyramidPooling(dilations, filters)
        
        self.conv1x1_bn_relu_1 = ConvolutionBnActivation(filters, 1)
        self.conv1x1_bn_relu_2 = ConvolutionBnActivation(64, 1)

        self.upsample2d_size2_1 = tf.keras.layers.UpSampling2D(size=2, interpolation="bilinear")
        self.upsample2d_size4_1 = tf.keras.layers.UpSampling2D(size=4, interpolation="bilinear")
        self.upsample2d_size4_2 = tf.keras.layers.UpSampling2D(size=4, interpolation="bilinear")

        self.concat = tf.keras.layers.Concatenate(axis=3)
        
        self.conv3x3_bn_relu_1 = ConvolutionBnActivation(256, 3)
        self.conv3x3_bn_relu_2 = ConvolutionBnActivation(256, 3)
        self.conv1x1_bn_sigmoid = ConvolutionBnActivation(self.n_classes, 1, post_activation="linear")

        self.final_activation = tf.keras.layers.Activation(final_activation)

    def call(self, inputs, training=None, mask=None):
        # inputs = tf.keras.layers.Input(shape=input_shape)
        # input = self.backbone.input
        # print(input.shape)
        # x = input
        input = inputs

        if training is None:
            training = True
            
        # print(self.backbone.get_layer(name=layer_names[1])(inputs))
        if self.output_stride == 8:
            x = self.backbone(input)[2]
            low_level_features = self.backbone(input)[1]
            dilations = [2 * rate for rate in self.dilations]
        elif self.output_stride == 16:
            x = self.backbone(input)[3]
            low_level_features = self.backbone(input)[1]
        else:
            raise ValueError("'output_stride' must be either 8 or 16.")
            

        # Encoder Module
        encoder = self.atrous_sepconv_bn_relu_1(x, training)
        encoder = self.aspp(encoder, training)
        encoder = self.conv1x1_bn_relu_1(encoder, training)

        if self.output_stride == 8:
            encoder_resize = self.upsample2d_size2_1(encoder)
        else:
            encoder_resize = self.upsample2d_size4_1(encoder)


        # Decoder Module
        decoder_low_level_features = self.atrous_sepconv_bn_relu_2(low_level_features, training)
        decoder_low_level_features = self.conv1x1_bn_relu_2(decoder_low_level_features, training)

        decoder = self.concat([decoder_low_level_features, encoder_resize])
        
        decoder = self.conv3x3_bn_relu_1(decoder, training)
        decoder = self.conv3x3_bn_relu_2(decoder, training)
        decoder = self.conv1x1_bn_sigmoid(decoder, training)

        decoder = self.upsample2d_size4_2(decoder)
        decoder = self.final_activation(decoder)

        return decoder