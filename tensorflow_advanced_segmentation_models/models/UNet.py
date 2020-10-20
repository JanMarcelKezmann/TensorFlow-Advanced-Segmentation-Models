import tensorflow as tf

from ._custom_layers_and_blocks import ConvolutionBnActivation, Upsample_x2_Block
from ..backbones.tf_backbones import create_base_model

################################################################################
# UNet
################################################################################
class UNet(tf.keras.Model):
    def __init__(self, n_classes, base_model, output_layers, height=None, width=None, filters=128,
                 final_activation="softmax", backbone_trainable=False,
                 up_filters=[32, 64, 128, 256, 512], include_top_conv=True, **kwargs):
        super(UNet, self).__init__(**kwargs)

        self.n_classes = n_classes
        self.backbone = None
        self.final_activation = final_activation
        self.filters = filters
        self.up_filters = up_filters
        self.include_top_conv = include_top_conv
        self.height = height
        self.width = width


        base_model.trainable = backbone_trainable
        self.backbone = tf.keras.Model(inputs=base_model.input, outputs=output_layers)

        # Define Layers
        self.conv3x3_bn_relu1 = ConvolutionBnActivation(filters, kernel_size=(3, 3), post_activation="relu")
        self.conv3x3_bn_relu2 = ConvolutionBnActivation(filters, kernel_size=(3, 3), post_activation="relu")
        
        self.upsample2d_x2_block1 = Upsample_x2_Block(up_filters[4])
        self.upsample2d_x2_block2 = Upsample_x2_Block(up_filters[3])
        self.upsample2d_x2_block3 = Upsample_x2_Block(up_filters[2])
        self.upsample2d_x2_block4 = Upsample_x2_Block(up_filters[1])
        self.upsample2d_x2_block5 = Upsample_x2_Block(up_filters[0])

        self.final_conv3x3 = tf.keras.layers.Conv2D(self.n_classes, (3, 3), strides=(1, 1), padding='same')

        self.final_activation = tf.keras.layers.Activation(final_activation)

    def call(self, inputs, training=None, mask=None):
        if training is None:
            training = True

        if self.include_top_conv:
            conv1 = self.conv3x3_bn_relu1(inputs, training=training)
            conv1 = self.conv3x3_bn_relu2(conv1, training=training)
        else:
            conv1 = None

        x = self.backbone(inputs)[4]

        upsample = self.upsample2d_x2_block1(x, self.backbone(inputs)[3], training)
        upsample = self.upsample2d_x2_block2(upsample, self.backbone(inputs)[2], training)
        upsample = self.upsample2d_x2_block3(upsample, self.backbone(inputs)[1], training)
        upsample = self.upsample2d_x2_block4(upsample, self.backbone(inputs)[0], training)
        upsample = self.upsample2d_x2_block5(upsample, conv1, training)

        x = self.final_conv3x3(upsample, training=training)
        x = self.final_activation(x)

        return x

    def model(self):
        x = tf.keras.layers.Input(shape=(self.height, self.width, 3))
        return tf.keras.Model(inputs=[x], outputs=self.call(x))