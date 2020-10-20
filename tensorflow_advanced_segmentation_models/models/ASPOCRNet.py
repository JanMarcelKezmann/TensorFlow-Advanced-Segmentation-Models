import tensorflow as tf
import tensorflow.keras.backend as K

from ._custom_layers_and_blocks import ConvolutionBnActivation, SpatialOCR_ASP_Module
from ..backbones.tf_backbones import create_base_model

################################################################################
# ASP Object-Contextual Representations Network
################################################################################
class ASPOCRNet(tf.keras.Model):
    def __init__(self, n_classes, base_model, output_layers, height=None, width=None, filters=256,
                 final_activation="softmax", backbone_trainable=False,
                 spatial_context_scale=1, **kwargs):
        super(ASPOCRNet, self).__init__(**kwargs)

        self.n_classes = n_classes
        self.backbone = None
        self.filters = filters
        self.final_activation = final_activation
        self.spatial_context_scale = spatial_context_scale
        self.height = height
        self.width = width


        output_layers = output_layers[:4]

        base_model.trainable = backbone_trainable
        self.backbone = tf.keras.Model(inputs=base_model.input, outputs=output_layers)

        # Layers
        self.conv3x3_bn_relu = ConvolutionBnActivation(filters, (3, 3))
        self.dropout = tf.keras.layers.Dropout(0.05)
        self.conv1x1_bn_activation = ConvolutionBnActivation(filters, (1, 1), post_activation=final_activation)
        self.upsampling2d_x2 = tf.keras.layers.UpSampling2D(size=2, interpolation="bilinear")

        self.asp_ocr = SpatialOCR_ASP_Module(filters, scale=spatial_context_scale)
        
        self.final_conv1x1_bn_activation = ConvolutionBnActivation(self.n_classes, (1, 1), post_activation=final_activation)
        self.final_upsampling2d = tf.keras.layers.UpSampling2D(size=8, interpolation="bilinear")

    def call(self, inputs, training=None, mask=None):
        if training is None:
            training = True

        x0, x1, x2, x3 = self.backbone(inputs, training=training)

        x_dsn = self.conv3x3_bn_relu(x3, training=training)
        x_dsn = self.dropout(x_dsn, training=training)
        x_dsn = self.conv1x1_bn_activation(x_dsn, training=training)
        x_dsn = self.upsampling2d_x2(x_dsn)

        x = self.asp_ocr(x2, x_dsn, training=training)
        
        x = self.final_conv1x1_bn_activation(x, training=training)
        x = self.final_upsampling2d(x)

        return x

    def model(self):
        x = tf.keras.layers.Input(shape=(self.height, self.width, 3))
        return tf.keras.Model(inputs=[x], outputs=self.call(x))