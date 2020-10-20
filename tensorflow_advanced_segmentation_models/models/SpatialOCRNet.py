import tensorflow as tf
import tensorflow.keras.backend as K

from ._custom_layers_and_blocks import ConvolutionBnActivation, SpatialGather_Module, SpatialOCR_Module
from ..backbones.tf_backbones import create_base_model

################################################################################
# Spatial Object-Contextual Representations Network
################################################################################
class SpatialOCRNet(tf.keras.Model):
    def __init__(self, n_classes, base_model, output_layers, height=None, width=None, filters=256,
                 final_activation="softmax", backbone_trainable=False,
                 spatial_ocr_scale=1, spatial_context_scale=1, **kwargs):
        super(SpatialOCRNet, self).__init__(**kwargs)

        self.n_classes = n_classes
        self.backbone = None
        self.filters = filters
        self.final_activation = final_activation
        self.spatial_ocr_scale = spatial_ocr_scale
        self.spatial_context_scale = spatial_context_scale
        self.height = height
        self.width = width


        output_layers = output_layers[:4]

        base_model.trainable = backbone_trainable
        self.backbone = tf.keras.Model(inputs=base_model.input, outputs=output_layers)

        # Layers
        self.conv3x3_bn_relu_1 = ConvolutionBnActivation(filters, (3, 3))
        self.upsampling2d_x2 = tf.keras.layers.UpSampling2D(size=2, interpolation="bilinear")
        self.conv3x3_bn_relu_2 = ConvolutionBnActivation(filters, (3, 3))
        self.dropout_1 = tf.keras.layers.Dropout(0.05)
        self.conv1x1_bn_activation = ConvolutionBnActivation(filters, (1, 1), post_activation=final_activation)

        self.spatial_context = SpatialGather_Module(scale=spatial_context_scale)
        self.spatial_ocr = SpatialOCR_Module(filters, scale=spatial_ocr_scale, dropout=0.05)
        
        self.conv3x3_bn_relu_3 = ConvolutionBnActivation(filters, (3, 3))
        self.dropout = tf.keras.layers.Dropout(0.05)
        self.final_conv1x1_bn_activation = ConvolutionBnActivation(self.n_classes, (1, 1), post_activation=final_activation)
        self.final_upsampling2d = tf.keras.layers.UpSampling2D(size=8, interpolation="bilinear")

    def call(self, inputs, training=None, mask=None):
        if training is None:
            training = True

        x0, x1, x2, x3 = self.backbone(inputs, training=training)

        x = self.conv3x3_bn_relu_1(x3, training=training)
        x = self.upsampling2d_x2(x)

        x_dsn = self.conv3x3_bn_relu_2(x2, training=training)
        x_dsn = self.dropout(x_dsn, training=training)
        x_dsn = self.conv1x1_bn_activation(x_dsn, training=training)

        context = self.spatial_context(x, x_dsn, training=training)

        x = self.spatial_ocr(x, context, training=training)
        
        x = self.conv3x3_bn_relu_3(x, training=training)
        x = self.dropout(x, training=training)
        x = self.final_conv1x1_bn_activation(x, training=training)
        x = self.final_upsampling2d(x)

        return x

    def model(self):
        x = tf.keras.layers.Input(shape=(self.height, self.width, 3))
        return tf.keras.Model(inputs=[x], outputs=self.call(x))