import tensorflow as tf
import tensorflow.keras.backend as K

from ._custom_layers_and_blocks import ConvolutionBnActivation, Upsample_x2_Add_Block
from ..backbones.tf_backbones import create_base_model

################################################################################
# Fully Convolutional Network
################################################################################
class FCN(tf.keras.Model):
    def __init__(self, n_classes, base_model, output_layers, height=None, width=None, filters=256,
                 final_activation="softmax", backbone_trainable=False,
                 backbone_output_factor=32, **kwargs):
        super(FCN, self).__init__(**kwargs)

        self.n_classes = n_classes
        self.backbone = None
        self.final_activation = final_activation
        self.filters = filters
        self.backbone_output_factor = backbone_output_factor
        self.height = height
        self.width = width

        
        if self.backbone_output_factor == 32:
            output_layers = output_layers[:5]
            self.final_upsample2d = tf.keras.layers.UpSampling2D(size=8)
        elif self.backbone_output_factor == 16:
            output_layers = output_layers[:4]
            self.final_upsample2d = tf.keras.layers.UpSampling2D(size=4)
        elif self.backbone_output_factor == 8:
            output_layers = output_layers[:3]
            self.final_upsample2d = tf.keras.layers.UpSampling2D(size=2)
        else:
            raise ValueError("'backbone_output_factor' must be one of (8, 16, 32), got {}".format(self.backbone_output_factor))

        base_model.trainable = backbone_trainable
        self.backbone = tf.keras.Model(inputs=base_model.input, outputs=output_layers)

        # Define Layers
        self.conv1x1_bn_relu = ConvolutionBnActivation(n_classes, kernel_size=(1, 1), post_activation="relu")
        
        self.upsample2d_x2_add_block1 = Upsample_x2_Add_Block(n_classes)
        self.upsample2d_x2_add_block2 = Upsample_x2_Add_Block(n_classes)

        self.final_conv1x1_bn_activation = ConvolutionBnActivation(n_classes, kernel_size=(1, 1), post_activation=final_activation)

    def call(self, inputs, training=None, mask=None):
        if training is None:
            training = True

        x = self.backbone(inputs)[-1]

        x = self.conv1x1_bn_relu(x)

        upsample = self.upsample2d_x2_add_block1(x, self.backbone(inputs)[-2], training)
        upsample = self.upsample2d_x2_add_block2(upsample, self.backbone(inputs)[-3], training)

        upsample = self.final_upsample2d(upsample)
        x = self.final_conv1x1_bn_activation(upsample, training=training)

        return x

    def model(self):
        x = tf.keras.layers.Input(shape=(self.height, self.width, 3))
        return tf.keras.Model(inputs=[x], outputs=self.call(x))