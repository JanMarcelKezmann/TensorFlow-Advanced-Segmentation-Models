import tensorflow as tf

from ._custom_layers_and_blocks import ConvolutionBnActivation, SpatialContextBlock
from ..backbones.tf_backbones import create_backbone

class PSPNet(tf.keras.models.Model):
    def __init__(self, n_classes, backbone, filters=256, activation="softmax",
                 dropout=None, pooling_type="avg", backbone_output_layer_idx=0,
                 final_upsample_factor=2, **kwargs):
        super(PSPNet, self).__init__()
        
        self.n_classes = n_classes
        self.backbone = backbone
        self.activation = activation
        self.filters = filters
        self.dropout = dropout
        self.pooling_type = pooling_type
        self.backbone_output_layer_idx = backbone_output_layer_idx
        self.final_upsample_factor = final_upsample_factor

        axis = 3 if K.image_data_format() == "channels_last" else 1


        # Define Layers
        self.spatial_context_block_1 = SpatialContextBlock(1, filters, pooling_type)
        self.spatial_context_block_2 = SpatialContextBlock(2, filters, pooling_type)
        self.spatial_context_block_3 = SpatialContextBlock(3, filters, pooling_type)
        self.spatial_context_block_4 = SpatialContextBlock(6, filters, pooling_type)

        self.concat = tf.keras.layers.Concatenate(axis=axis)
        self.conv1x1_bn_relu = ConvolutionBnActivation(filters, (1, 1))

        self.spatial_dropout = tf.keras.layers.SpatialDropout2D(dropout)

        self.final_conv3x3 = tf.keras.layers.Conv2D(self.n_classes, (3, 3), strides=(1, 1), padding='same')
        self.final_upsample2d = tf.keras.layers.UpSampling2D(size=final_upsample_factor, interpolation="bilinear")
        self.final_activation = tf.keras.layers.Activation(activation)
    
    
    def call(self, inputs, training=None, mask=None):
        if training is None:
            training = True

        x = self.backbone(inputs)[self.backbone_output_layer_idx]

        if K.image_data_format() == "channels_last":
            if x.shape[1] % 6 != 0 or x.shape[2] % 6 != 0:
                raise ValueError("Height and Width of the backbone output must be divisible by 6, i.e. \
                input_height or input_width / (backbone_output_layer_idx * 2 + 2) must be divisble by 6.")
        else:
            if x.shape[2] % 6 != 0 or x.shape[2] % 6 != 0:
                raise ValueError("Height and Width of the backbone output must be divisible by 6, i.e. \
                input_height or input_width / (backbone_output_layer_idx * 2 + 2) must be divisble by 6.")


        x1 = self.spatial_context_block_1(x, training=training)
        x2 = self.spatial_context_block_2(x, training=training)
        x3 = self.spatial_context_block_3(x, training=training)
        x6 = self.spatial_context_block_4(x, training=training)

        # print("Shape x1: " + str(x1.shape))
        # print("Shape x2: " + str(x2.shape))
        # print("Shape x3: " + str(x3.shape))
        # print("Shape x6: " + str(x6.shape))

        x = self.concat([x1, x2, x3, x6])
        x = self.conv1x1_bn_relu(x, training=training)

        if self.dropout is not None:
            x = self.spatial_dropout(x, training=training)

        x = self.final_conv3x3(x)
        x = self.final_upsample2d(x)
        x = self.final_activation(x)

        return x
