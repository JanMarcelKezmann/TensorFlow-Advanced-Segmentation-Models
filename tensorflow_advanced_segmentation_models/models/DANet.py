import tensorflow as tf
import tensorflow.keras.backend as K

from ._custom_layers_and_blocks import ConvolutionBnActivation, PAM_Module, CAM_Module
from ..backbones.tf_backbones import create_base_model

################################################################################
# Dual Attention Network
################################################################################
class DANet(tf.keras.Model):
    def __init__(self, n_classes, base_model, output_layers, height=None, width=None, filters=256,
                 final_activation="softmax", backbone_trainable=False,
                 output_stride=8, **kwargs):
        super(DANet, self).__init__(**kwargs)

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
        elif self.output_stride == 16:
            self.final_upsampling2d = tf.keras.layers.UpSampling2D(size=16, interpolation="bilinear")
            self.output_layers = self.output_layers[:4]
        else:
            raise ValueError("'output_stride' must be one of (8, 16), got {}".format(self.output_stride))

        base_model.trainable = backbone_trainable
        self.backbone = tf.keras.Model(inputs=base_model.input, outputs=output_layers)

        # Layers
        self.conv3x3_bn_relu_1 = ConvolutionBnActivation(filters, (3, 3))
        self.conv3x3_bn_relu_2 = ConvolutionBnActivation(filters, (3, 3))
        
        self.pam = PAM_Module(filters)
        self.cam = CAM_Module(filters)

        self.conv3x3_bn_relu_3 = ConvolutionBnActivation(filters, (3, 3))
        self.conv3x3_bn_relu_4 = ConvolutionBnActivation(filters, (3, 3))
        
        self.dropout_1 = tf.keras.layers.Dropout(0.1)
        self.dropout_2 = tf.keras.layers.Dropout(0.1)
        self.dropout_3 = tf.keras.layers.Dropout(0.1)

        self.conv1x1_bn_relu_1 = ConvolutionBnActivation(n_classes, (1, 1), post_activation="relu")
        self.conv1x1_bn_relu_2 = ConvolutionBnActivation(n_classes, (1, 1), post_activation="relu")
        self.conv1x1_bn_relu_3 = ConvolutionBnActivation(n_classes, (1, 1), post_activation="relu")
        
        axis = 3 if K.image_data_format() == "channels_last" else 1
        self.concat_1 = tf.keras.layers.Concatenate(axis=axis)
        self.concat_2 = tf.keras.layers.Concatenate(axis=axis)
        
        self.final_conv1x1_bn_activation = ConvolutionBnActivation(n_classes, (1, 1), post_activation=final_activation)
    
    def call(self, inputs, training=None, mask=None):
        if training is None:
            training = True
        
        x = self.backbone(inputs, training=training)[-1]

        x_pam = self.conv3x3_bn_relu_1(x, training=training)
        x_pam_out = self.pam(x_pam, training=training)
        x_pam = self.conv3x3_bn_relu_3(x_pam_out, training=training)
        x_pam = self.dropout_1(x_pam, training=training)
        x_pam = self.conv1x1_bn_relu_1(x_pam, training=training)

        x_cam = self.conv3x3_bn_relu_2(x, training=training)
        x_cam_out = self.cam(x_cam, training=training)
        x_cam = self.conv3x3_bn_relu_4(x_cam_out, training=training)
        x_cam = self.dropout_2(x_cam, training=training)
        x_cam = self.conv1x1_bn_relu_2(x_cam, training=training)

        # x_pam_cam = x_pam_out + x_cam_out # maybe add or concat layer
        x_pam_cam = self.concat_1([x_pam_out, x_cam_out])
        x = self.dropout_3(x_pam_cam, training=training)
        x = self.conv1x1_bn_relu_3(x, training=training)

        x = self.concat_2([x_pam, x_cam, x])
        x = self.final_conv1x1_bn_activation(x, training=training)
        x = self.final_upsampling2d(x)

        return x

    def model(self):
        x = tf.keras.layers.Input(shape=(self.height, self.width, 3))
        return tf.keras.Model(inputs=[x], outputs=self.call(x))