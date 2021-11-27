import tensorflow as tf
import tensorflow.keras.backend as K

from ._custom_layers_and_blocks import ConvolutionBnActivation, BottleneckBlock, HighResolutionModule,SpatialGather_Module, SpatialOCR_Module
from ..backbones.tf_backbones import create_base_model

################################################################################
# High Resolution Network + Object-Contextual Representations
################################################################################
class HRNetOCR(tf.keras.Model):
    def __init__(self, n_classes, filters=64, height=None, width=None, final_activation="softmax",
                 spatial_ocr_scale=1, spatial_context_scale=1, **kwargs):
        super(HRNetOCR, self).__init__(**kwargs)

        self.n_classes = n_classes
        self.filters = filters
        self.height = height
        self.width = width
        self.final_activation = final_activation
        self.spatial_ocr_scale = spatial_ocr_scale
        self.spatial_context_scale = spatial_context_scale

        axis = 3 if K.image_data_format() == "channels_last" else 1

        # Stem Net
        ### Probably set strides to default, i.e. (1, 1)
        self.conv3x3_bn_relu_1 = ConvolutionBnActivation(filters, (3, 3))
        self.conv3x3_bn_relu_2 = ConvolutionBnActivation(filters, (3, 3))

        # stage 1
        self.bottleneck_downsample = BottleneckBlock(64, downsample=True)
        self.bottleneck_1 = BottleneckBlock(64)
        self.bottleneck_2 = BottleneckBlock(64)
        self.bottleneck_3 = BottleneckBlock(64)

        # Stage 2
        # Transition
        self.conv3x3_bn_relu_stage2_1 = ConvolutionBnActivation(48, (3, 3), momentum=0.1)
        self.conv3x3_bn_relu_stage2_2 = ConvolutionBnActivation(96, (3, 3), strides=(2, 2), momentum=0.1)

        # Stage
        # num_modules=1, num_branches=2, blocks=[4, 4], channels=[48, 96]
        self.hrn_stage2_module_1 = HighResolutionModule(num_branches=2, blocks=[4, 4], filters=[48, 96])
        self.hrn_stage2_module_2 = HighResolutionModule(num_branches=2, blocks=[4, 4], filters=[48, 96])

        # Stage 3
        # Transition
        self.conv3x3_bn_relu_stage3 = ConvolutionBnActivation(192, (3, 3), strides=(2, 2), momentum=0.1)

        # Stage
        # num_modules=4, num_branches=3, blocks=[4, 4, 4], channels=[48, 96, 192]
        self.hrn_stage3_module_1 = HighResolutionModule(num_branches=3, blocks=[4, 4, 4], filters=[48, 96, 192])
        self.hrn_stage3_module_2 = HighResolutionModule(num_branches=3, blocks=[4, 4, 4], filters=[48, 96, 192])
        self.hrn_stage3_module_3 = HighResolutionModule(num_branches=3, blocks=[4, 4, 4], filters=[48, 96, 192])
        self.hrn_stage3_module_4 = HighResolutionModule(num_branches=3, blocks=[4, 4, 4], filters=[48, 96, 192])

        # Stage 4
        # Transition
        self.conv3x3_bn_relu_stage4 = ConvolutionBnActivation(384, (3, 3), strides=(2, 2), momentum=0.1)

        # Stage
        # num_modules=3, num_branches=4, num_blocks=[4, 4, 4, 4], num_channels=[48, 96, 192, 384]
        self.hrn_stage4_module_1 = HighResolutionModule(num_branches=4, blocks=[4, 4, 4, 4], filters=[48, 96, 192, 384])
        self.hrn_stage4_module_2 = HighResolutionModule(num_branches=4, blocks=[4, 4, 4, 4], filters=[48, 96, 192, 384])
        self.hrn_stage4_module_3 = HighResolutionModule(num_branches=4, blocks=[4, 4, 4, 4], filters=[48, 96, 192, 384])

        # Upsampling and Concatentation of stages
        self.upsample_x2 = tf.keras.layers.UpSampling2D(size=2, interpolation="bilinear")
        self.upsample_x4 = tf.keras.layers.UpSampling2D(size=4, interpolation="bilinear")
        self.upsample_x8 = tf.keras.layers.UpSampling2D(size=8, interpolation="bilinear")

        self.concat = tf.keras.layers.Concatenate(axis=axis)

        # OCR
        self.aux_head = tf.keras.Sequential([
            ConvolutionBnActivation(720, (1, 1)),
            tf.keras.layers.Conv2D(filters=self.n_classes, kernel_size=(1, 1), use_bias=True),
            tf.keras.layers.Activation(final_activation)
            ])
        self.conv3x3_bn_relu_ocr = ConvolutionBnActivation(512, (3, 3))

        self.spatial_context = SpatialGather_Module(scale=spatial_context_scale)
        self.spatial_ocr = SpatialOCR_Module(512, scale=spatial_ocr_scale, dropout=0.05)

        self.final_conv3x3 = tf.keras.layers.Conv2D(filters=self.n_classes, kernel_size=(1, 1), use_bias=True)
        self.final_activation = tf.keras.layers.Activation(final_activation)

    def call(self, inputs, training=None, mask=None):
        if training is None:
            training = True

        x = self.conv3x3_bn_relu_1(inputs, training=training)
        x = self.conv3x3_bn_relu_2(x, training=training)

        # Stage 1
        x = self.bottleneck_downsample(x, training=training)
        x = self.bottleneck_1(x, training=training)
        x = self.bottleneck_2(x, training=training)
        x = self.bottleneck_3(x, training=training)

        # Stage 2
        x_1 = self.conv3x3_bn_relu_stage2_1(x, training=training)
        x_2 = self.conv3x3_bn_relu_stage2_2(x, training=training) # includes strided convolution

        y_list = self.hrn_stage2_module_1(x_1, x_2, None, None, training=training)
        y_list = self.hrn_stage2_module_2(y_list[0], y_list[1], None, None, training=training)

        # Stage 3
        x_3 = self.conv3x3_bn_relu_stage3(y_list[1], training=training) # includes strided convolution

        y_list = self.hrn_stage3_module_1(y_list[0], y_list[1], x_3, None, training=training)
        y_list = self.hrn_stage3_module_2(y_list[0], y_list[1], y_list[2], None, training=training)
        y_list = self.hrn_stage3_module_3(y_list[0], y_list[1], y_list[2], None, training=training)
        y_list = self.hrn_stage3_module_4(y_list[0], y_list[1], y_list[2], None, training=training)

        # Stage 4
        x_4 = self.conv3x3_bn_relu_stage4(y_list[2], training=training)

        y_list = self.hrn_stage4_module_1(y_list[0], y_list[1], y_list[2], x_4, training=training)
        y_list = self.hrn_stage4_module_2(y_list[0], y_list[1], y_list[2], y_list[3], training=training)
        y_list = self.hrn_stage4_module_3(y_list[0], y_list[1], y_list[2], y_list[3], training=training)

        # Upsampling + Concatentation
        x_2 = self.upsample_x2(y_list[1])
        x_3 = self.upsample_x4(y_list[2])
        x_4 = self.upsample_x8(y_list[3])

        feats = self.concat([y_list[0], x_2, x_3, x_4])

        # OCR
        aux = self.aux_head(feats)
        
        feats = self.conv3x3_bn_relu_ocr(feats)

        context = self.spatial_context(feats, aux)
        feats = self.spatial_ocr(feats, context)

        out = self.final_conv3x3(feats)
        out = self.final_activation(out)

        return out

    def model(self):
        x = tf.keras.layers.Input(shape=(self.height, self.width, 3))
        return tf.keras.Model(inputs=[x], outputs=self.call(x))
