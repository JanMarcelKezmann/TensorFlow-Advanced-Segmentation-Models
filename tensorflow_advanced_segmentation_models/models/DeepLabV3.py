import tensorflow as tf
import tensorflow.keras.backend as K

from ._custom_layers_and_blocks import ConvolutionBnActivation, AtrousSeparableConvolutionBnReLU, AtrousSpatialPyramidPoolingV3
from ..backbones.tf_backbones import create_base_model

################################################################################
# DeepLabV3
################################################################################
class DeepLabV3(tf.keras.Model):
    def __init__(self, n_classes, base_model, output_layers, height=None, width=None, filters=256,
                 final_activation="softmax", backbone_trainable=False,
                 output_stride=8, dilations=[6, 12, 18], **kwargs):
        super(DeepLabV3, self).__init__(**kwargs)

        self.n_classes = n_classes
        self.backbone = None
        self.filters = filters
        self.final_activation = final_activation
        self.output_stride = output_stride
        self.height = height
        self.width = width


        if self.output_stride == 8:
            self.final_upsampling2d = tf.keras.layers.UpSampling2D(size=8, interpolation="bilinear")
            output_layers = output_layers[:3]
            self.dilations = [2 * rate for rate in dilations]
        elif self.output_stride == 16:
            self.final_upsampling2d = tf.keras.layers.UpSampling2D(size=16, interpolation="bilinear")
            output_layers = output_layers[:4]
            self.dilations = dilations
        else:
            raise ValueError("'output_stride' must be one of (8, 16), got {}".format(self.output_stride))

        base_model.trainable = backbone_trainable
        self.backbone = tf.keras.Model(inputs=base_model.input, outputs=output_layers)

        # Define Layers
        self.atrous_sepconv_bn_relu = AtrousSeparableConvolutionBnReLU(dilation=2, filters=filters, kernel_size=3)
        self.aspp = AtrousSpatialPyramidPoolingV3(dilations, filters)

        self.conv1x1_bn_relu = ConvolutionBnActivation(filters, (1, 1))
        self.conv1x1_bn_activation = ConvolutionBnActivation(n_classes, (1, 1), post_activation=final_activation)

        self.final_activation = tf.keras.layers.Activation(final_activation)

    def call(self, inputs, training=None, mask=None):
        if training is None:
            training = True
        
        x = self.backbone(inputs, training=training)[-1]
            
        x = self.atrous_sepconv_bn_relu(x, training=training)
        x = self.aspp(x, training=training)

        x = self.conv1x1_bn_relu(x, training=training)
        x = self.conv1x1_bn_activation(x, training=training)

        x = self.final_upsampling2d(x)
        x = self.final_activation(x)

        return x

    def model(self):
        x = tf.keras.layers.Input(shape=(self.height, self.width, 3))
        return tf.keras.Model(inputs=[x], outputs=self.call(x))