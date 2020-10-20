import tensorflow as tf
import tensorflow.keras.backend as K

from ._custom_layers_and_blocks import ConvolutionBnActivation, AtrousSpatialPyramidPoolingV1
from ..backbones.tf_backbones import create_base_model

################################################################################
# DeepLab
################################################################################
class DeepLab(tf.keras.Model):
    def __init__(self, n_classes, base_model, output_layers, height=None, width=None, filters=256,
                 final_activation="softmax", backbone_trainable=False,
                 final_upsample_factor=8, **kwargs):
        super(DeepLab, self).__init__(**kwargs)

        self.n_classes = n_classes
        self.backbone = None
        self.final_activation = final_activation
        self.filters = filters
        self.final_upsample_factor = final_upsample_factor
        self.height = height
        self.width = width


        if self.final_upsample_factor == 16:
            output_layers = output_layers[:4]
            self.final_upsample2d = tf.keras.layers.UpSampling2D(size=16)
        elif self.final_upsample_factor == 8:
            output_layers = output_layers[:3]
            self.final_upsample2d = tf.keras.layers.UpSampling2D(size=8)
        elif self.final_upsample_factor == 4:
            output_layers = output_layers[:2]
            self.final_upsample2d = tf.keras.layers.UpSampling2D(size=4)
        else:
            raise ValueError("'final_upsample_factor' must be one of (4, 8, 16), got {}".format(self.final_upsample_factor))

        base_model.trainable = backbone_trainable
        self.backbone = tf.keras.Model(inputs=base_model.input, outputs=output_layers)

        # Define Layers
        self.aspp = AtrousSpatialPyramidPoolingV1(filters)

        self.final_conv1x1_bn_activation = ConvolutionBnActivation(n_classes, kernel_size=(1, 1), post_activation=final_activation)

    def call(self, inputs, training=None, mask=None):
        if training is None:
            training = True

        x = self.backbone(inputs)[-1]

        aspp = self.aspp(x, training=training)

        upsample = self.final_upsample2d(aspp)
        x = self.final_conv1x1_bn_activation(upsample, training=training)

        return x

    def model(self):
        x = tf.keras.layers.Input(shape=(self.height, self.width, 3))
        return tf.keras.Model(inputs=[x], outputs=self.call(x))