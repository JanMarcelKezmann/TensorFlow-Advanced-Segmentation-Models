import tensorflow as tf
import tensorflow.keras.backend as K

from ._custom_layers_and_blocks import ConvolutionBnActivation, SpatialContextBlock
from ..backbones.tf_backbones import create_base_model

################################################################################
# Pyramid Scene Parsing Network
################################################################################
class PSPNet(tf.keras.models.Model):
    def __init__(self, n_classes, base_model, output_layers, height=None, width=None, filters=256,
                 final_activation="softmax", backbone_trainable=False,
                 dropout=None, pooling_type="avg", final_upsample_factor=2, **kwargs):
        super(PSPNet, self).__init__()
        
        self.n_classes = n_classes
        self.backbone = None
        self.final_activation = final_activation
        self.filters = filters
        self.dropout = dropout
        self.pooling_type = pooling_type
        self.final_upsample_factor = final_upsample_factor
        self.height = height
        self.width = width


        axis = 3 if K.image_data_format() == "channels_last" else 1

        if self.final_upsample_factor == 8:
            output_layers = output_layers[:3]
            self.final_upsample2d = tf.keras.layers.UpSampling2D(size=final_upsample_factor, interpolation="bilinear")
        elif self.final_upsample_factor == 4:
            output_layers = output_layers[:2]
            self.final_upsample2d = tf.keras.layers.UpSampling2D(size=final_upsample_factor, interpolation="bilinear")
        elif self.final_upsample_factor == 2:
            output_layers = output_layers[:1]
            self.final_upsample2d = tf.keras.layers.UpSampling2D(size=final_upsample_factor, interpolation="bilinear")
        else:
            raise ValueError("'final_upsample_factor' must be one of (2, 4, 8), got {}".format(self.final_upsample_factor))

        base_model.trainable = backbone_trainable
        self.backbone = tf.keras.Model(inputs=base_model.input, outputs=output_layers)

        # Define Layers
        self.spatial_context_block_1 = SpatialContextBlock(1, filters, pooling_type)
        self.spatial_context_block_2 = SpatialContextBlock(2, filters, pooling_type)
        self.spatial_context_block_3 = SpatialContextBlock(3, filters, pooling_type)
        self.spatial_context_block_4 = SpatialContextBlock(6, filters, pooling_type)

        self.concat = tf.keras.layers.Concatenate(axis=axis)
        self.conv1x1_bn_relu = ConvolutionBnActivation(filters, (1, 1))

        if dropout is not None:
            self.spatial_dropout = tf.keras.layers.SpatialDropout2D(dropout)

        self.final_conv3x3 = tf.keras.layers.Conv2D(self.n_classes, (3, 3), strides=(1, 1), padding='same')
        self.final_activation = tf.keras.layers.Activation(final_activation)
    
    
    def call(self, inputs, training=None, mask=None):
        if training is None:
            training = True

        if self.final_upsample_factor == 2:
            x = self.backbone(inputs)
        else:
            x = self.backbone(inputs)[-1]

        if K.image_data_format() == "channels_last":
            if x.shape[1] % 6 != 0 or x.shape[2] % 6 != 0:
                raise ValueError("Height and Width of the backbone output must be divisible by 6, i.e. \
                input_height or input_width / final_upsample_factor must be divisble by 6.")
        else:
            if x.shape[2] % 6 != 0 or x.shape[2] % 6 != 0:
                raise ValueError("Height and Width of the backbone output must be divisible by 6, i.e. \
                input_height or input_width / final_upsample_factor must be divisble by 6.")


        x1 = self.spatial_context_block_1(x, training=training)
        x2 = self.spatial_context_block_2(x, training=training)
        x3 = self.spatial_context_block_3(x, training=training)
        x6 = self.spatial_context_block_4(x, training=training)

        x = self.concat([x1, x2, x3, x6])
        x = self.conv1x1_bn_relu(x, training=training)

        if self.dropout is not None:
            x = self.spatial_dropout(x, training=training)

        x = self.final_conv3x3(x)
        x = self.final_upsample2d(x)
        x = self.final_activation(x)

        return x

    def model(self):
        x = tf.keras.layers.Input(shape=(self.height, self.width, 3))
        return tf.keras.Model(inputs=[x], outputs=self.call(x))