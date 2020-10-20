import tensorflow as tf
import tensorflow.keras.backend as K

from ._custom_layers_and_blocks import ConvolutionBnActivation, FPNBlock
from ..backbones.tf_backbones import create_base_model

################################################################################
# Feature Pyramid Network
################################################################################
class FPNet(tf.keras.models.Model):
    def __init__(self, n_classes, base_model, output_layers, height=None, width=None, filters=128,
                 final_activation="softmax", backbone_trainable=False,
                 pyramid_filters=256, aggregation="sum", dropout=None, **kwargs):
        super(FPNet, self).__init__()
        
        self.n_classes = n_classes
        self.backbone = None
        self.final_activation = final_activation
        self.filters = filters
        self.pyramid_filters = pyramid_filters
        self.aggregation = aggregation
        self.dropout = dropout
        self.height = height
        self.width = width


        self.axis = 3 if K.image_data_format() == "channels_last" else 1

        base_model.trainable = backbone_trainable
        self.backbone = tf.keras.Model(inputs=base_model.input, outputs=output_layers)

        # Define Layers
        self.fpn_block_p5 = FPNBlock(pyramid_filters)
        self.fpn_block_p4 = FPNBlock(pyramid_filters)
        self.fpn_block_p3 = FPNBlock(pyramid_filters)
        self.fpn_block_p2 = FPNBlock(pyramid_filters)

        self.conv3x3_bn_relu_1 = ConvolutionBnActivation(filters, (3, 3))
        self.conv3x3_bn_relu_2 = ConvolutionBnActivation(filters, (3, 3))
        self.conv3x3_bn_relu_3 = ConvolutionBnActivation(filters, (3, 3))
        self.conv3x3_bn_relu_4 = ConvolutionBnActivation(filters, (3, 3))
        self.conv3x3_bn_relu_5 = ConvolutionBnActivation(filters, (3, 3))
        self.conv3x3_bn_relu_6 = ConvolutionBnActivation(filters, (3, 3))
        self.conv3x3_bn_relu_7 = ConvolutionBnActivation(filters, (3, 3))
        self.conv3x3_bn_relu_8 = ConvolutionBnActivation(filters, (3, 3))

        self.upsample2d_s5 = tf.keras.layers.UpSampling2D((8, 8), interpolation="nearest")
        self.upsample2d_s4 = tf.keras.layers.UpSampling2D((4, 4), interpolation="nearest")
        self.upsample2d_s3 = tf.keras.layers.UpSampling2D((2, 2), interpolation="nearest")


        self.add = tf.keras.layers.Add()
        self.concat = tf.keras.layers.Concatenate(axis=self.axis)

        self.spatial_dropout = tf.keras.layers.SpatialDropout2D(dropout)
        self.pre_final_conv3x3_bn_relu = ConvolutionBnActivation(filters, (3, 3))
        self.final_upsample2d = tf.keras.layers.UpSampling2D(size=2, interpolation="bilinear")

        self.final_conv3x3 = tf.keras.layers.Conv2D(self.n_classes, (3, 3), strides=(1, 1), padding='same')
        self.final_activation = tf.keras.layers.Activation(final_activation)
    
    
    def call(self, inputs, training=None, mask=None):
        if self.axis == 3:
            if inputs.shape[1] % 160 != 0 or inputs.shape[2] % 160 != 0:
                raise ValueError("Input height and width must be a multiple of 160, got height = " + str(inputs.shape[1]) + " and width " + str(inputs.shape[0]) + ".")

        if training is None:
            training = True

        x = self.backbone(inputs)[4]

        p5 = self.fpn_block_p5(x, self.backbone(inputs)[3], training=training)
        p4 = self.fpn_block_p4(p5, self.backbone(inputs)[2], training=training)
        p3 = self.fpn_block_p3(p4, self.backbone(inputs)[1], training=training)
        p2 = self.fpn_block_p2(p3, self.backbone(inputs)[0], training=training)
         
        s5 = self.conv3x3_bn_relu_1(p5, training=training)
        s5 = self.conv3x3_bn_relu_2(s5, training=training)
        s4 = self.conv3x3_bn_relu_3(p4, training=training)
        s4 = self.conv3x3_bn_relu_4(s4, training=training)
        s3 = self.conv3x3_bn_relu_5(p3, training=training)
        s3 = self.conv3x3_bn_relu_6(s3, training=training)
        s2 = self.conv3x3_bn_relu_7(p2, training=training)
        s2 = self.conv3x3_bn_relu_8(s2, training=training)

        s5 = self.upsample2d_s5(s5)
        s4 = self.upsample2d_s4(s4)
        s3 = self.upsample2d_s3(s3)

        if self.aggregation == "sum":
            x = self.add([s2, s3, s4, s5])
        elif self.aggregation == "concat":
            x = self.concat([s2, s3, s4, s5])
        else:
            raise ValueError("Aggregation parameter should be one of ['sum', 'concat'], got {}".format(aggregation))

        if self.dropout is not None:
            if self.dropout >= 1 or self.dropout < 0:
                raise ValueError("'dropout' must be between 0 and 1, got {}".format(dropout))
            else:
                x = self.spatial_dropout(x, training=training)

        # Final Stage
        x = self.pre_final_conv3x3_bn_relu(x, training=training)
        x = self.final_upsample2d(x)

        x = self.final_conv3x3(x)
        x = self.final_activation(x)

        return x

    def model(self):
        x = tf.keras.layers.Input(shape=(self.height, self.width, 3))
        return tf.keras.Model(inputs=[x], outputs=self.call(x))