import tensorflow as tf
import tensorflow.keras.backend as K

from ._custom_layers_and_blocks import ConvolutionBnActivation, Base_OC_Module, Pyramid_OC_Module, ASP_OC_Module
from ..backbones.tf_backbones import create_base_model

################################################################################
# Object Context Network
################################################################################
class OCNet(tf.keras.Model):
    def __init__(self, n_classes, base_model, output_layers, height=None, width=None, filters=256,
                 final_activation="softmax", backbone_trainable=False,
                 output_stride=8, dilations=[6, 12, 18], oc_module="base_oc", **kwargs):
        super(OCNet, self).__init__(**kwargs)

        self.n_classes = n_classes
        self.backbone = None
        self.filters = filters
        self.final_activation = final_activation
        self.output_stride = output_stride
        self.height = height
        self.width = width


        if self.output_stride == 8:
            self.final_upsampling2d = tf.keras.layers.UpSampling2D(size=8, interpolation="bilinear")
            output_layers = output_layers[:3]
            self.dilations = [2 * rate for rate in dilations]
        elif self.output_stride == 16:
            self.final_upsampling2d = tf.keras.layers.UpSampling2D(size=16, interpolation="bilinear")
            self.output_layers = self.output_layers[:4]
            self.dilations = dilations
        else:
            raise ValueError("'output_stride' must be one of (8, 16), got {}".format(self.output_stride))

        base_model.trainable = backbone_trainable
        self.backbone = tf.keras.Model(inputs=base_model.input, outputs=output_layers)

        # Define Layers
        self.conv3x3_bn_relu = ConvolutionBnActivation(filters, (3, 3))
        if oc_module == "base_oc":
            self.oc = Base_OC_Module(filters)
        elif oc_module == "pyramid_oc":
            self.oc = Pyramid_OC_Module(filters=filters, levels=[1, 2, 3, 6])
        elif oc_module == "asp_oc":
            self.oc = ASP_OC_Module(filters, self.dilations)
        else:
            raise ValueError("'oc_module' must be one of ('base_oc', 'pyramid_oc', 'asp_oc'), got {}".format(oc_module))

        self.final_conv1x1_bn_activation = ConvolutionBnActivation(n_classes, (1, 1), post_activation=final_activation)

    def call(self, inputs, training=None, mask=None):
        if training is None:
            training = True
        
        x = self.backbone(inputs, training=training)[-1]
        
        x = self.conv3x3_bn_relu(x, training=training)

        if K.image_data_format() == "channels_last":
            if x.shape[1] % 6 != 0 or x.shape[2] % 6 != 0:
                raise ValueError("Height and Width of the backbone output must be divisible by 6, i.e. \
                input_height or input_width / final_upsample_factor must be divisble by 6.")
        else:
            if x.shape[2] % 6 != 0 or x.shape[2] % 6 != 0:
                raise ValueError("Height and Width of the backbone output must be divisible by 6, i.e. \
                input_height or input_width / final_upsample_factor must be divisble by 6.")

        x = self.oc(x, training=training)        

        x = self.final_conv1x1_bn_activation(x, training=training)

        x = self.final_upsampling2d(x)

        return 

    def model(self):
        x = tf.keras.layers.Input(shape=(self.height, self.width, 3))
        return tf.keras.Model(inputs=[x], outputs=self.call(x))
