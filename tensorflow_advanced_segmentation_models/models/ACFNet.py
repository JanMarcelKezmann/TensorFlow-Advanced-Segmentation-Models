import tensorflow as tf
import tensorflow.keras.backend as K

from ._custom_layers_and_blocks import ConvolutionBnActivation, AtrousSpatialPyramidPoolingV3, AttCF_Module
from ..backbones.tf_backbones import create_base_model

################################################################################
# Attentional Class Feature Network
################################################################################
class ACFNet(tf.keras.Model):
    def __init__(self, n_classes, base_model, output_layers, height=None, width=None, filters=256,
                 final_activation="softmax", backbone_trainable=False,
                 dilations=[6, 12, 18], **kwargs):
        super(ACFNet, self).__init__(**kwargs)

        self.n_classes = n_classes
        self.backbone = None
        self.filters = filters
        self.final_activation = final_activation
        self.height = height
        self.width = width

        output_layers = output_layers[:3]

        base_model.trainable = backbone_trainable
        self.backbone = tf.keras.Model(inputs=base_model.input, outputs=output_layers)

        # Layers
        self.aspp = AtrousSpatialPyramidPoolingV3(dilations, filters)

        self.dropout_2 = tf.keras.layers.Dropout(0.25)
        self.conv3x3_bn_relu_2 = ConvolutionBnActivation(filters, (3, 3))
        self.dropout_3 = tf.keras.layers.Dropout(0.1)
        self.conv1x1_bn_activation = ConvolutionBnActivation(n_classes, (1, 1), post_activation=final_activation)

        self.acf = AttCF_Module(filters)

        axis = 3 if K.image_data_format() == "channels_last" else 1
        self.concat = tf.keras.layers.Concatenate(axis=axis)

        self.conv3x3_bn_relu_3 = ConvolutionBnActivation(filters, (3, 3))
        self.dropout_4 = tf.keras.layers.Dropout(0.1)
        self.final_conv1x1_bn_activation = ConvolutionBnActivation(n_classes, (1, 1), post_activation=final_activation)
        self.final_upsampling2d = tf.keras.layers.UpSampling2D(size=8, interpolation="bilinear")

    def call(self, inputs, training=None, mask=None):
        if training is None:
            training = True

        x = self.backbone(inputs, training=training)[-1]
        aspp = self.aspp(x, training=training)

        x = self.dropout_2(aspp, training=training)
        x = self.conv3x3_bn_relu_2(x, training=training)
        x = self.dropout_3(x, training=training)
        x = self.conv1x1_bn_activation(x, training=training) # coarse segmentation map
        
        x = self.acf(aspp, x, training=training)

        x = self.concat([x, aspp])

        x = self.conv3x3_bn_relu_3(x, training=training)
        x = self.dropout_4(x, training=training)
        x = self.final_conv1x1_bn_activation(x, training=training)
        x = self.final_upsampling2d(x)
        
        return x

    def model(self):
        x = tf.keras.layers.Input(shape=(self.height, self.width, 3))
        return tf.keras.Model(inputs=[x], outputs=self.call(x))
