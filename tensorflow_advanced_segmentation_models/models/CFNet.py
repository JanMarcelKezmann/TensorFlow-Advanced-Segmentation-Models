import tensorflow as tf
import tensorflow.keras.backend as K

from ._custom_layers_and_blocks import ConvolutionBnActivation, GlobalPooling, AggCF_Module
from ..backbones.tf_backbones import create_base_model

################################################################################
# Co-Occurent Feature Network
################################################################################
class CFNet(tf.keras.Model):
    # Co-occurent Feature Network
    def __init__(self, n_classes, base_model, output_layers, height=None, width=None, filters=256,
                 final_activation="softmax", backbone_trainable=False,
                 lateral=True, global_pool=False, acf_pool=True,
                 acf_kq_transform="conv", acf_concat=False, **kwargs):
        super(CFNet, self).__init__(**kwargs)

        self.n_classes = n_classes
        self.backbone = None
        self.filters = filters
        self.final_activation = final_activation
        self.lateral = lateral
        self.global_pool = global_pool
        self.acf_pool = acf_pool
        self.acf_kq_transform = acf_kq_transform
        self.acf_concat = acf_concat
        self.height = height
        self.width = width


        output_layers = output_layers[:4]

        base_model.trainable = backbone_trainable
        self.backbone = tf.keras.Model(inputs=base_model.input, outputs=output_layers)

        # Layers
        self.conv3x3_bn_relu_1 = ConvolutionBnActivation(filters, (3, 3))
        self.conv3x3_bn_relu_2 = ConvolutionBnActivation(filters, (3, 3))
        self.conv3x3_bn_relu_3 = ConvolutionBnActivation(filters, (3, 3))
        self.conv3x3_bn_relu_4 = ConvolutionBnActivation(filters, (3, 3))

        self.upsample2d_2x = tf.keras.layers.UpSampling2D(size=2, interpolation="bilinear")
        self.pool2d = tf.keras.layers.MaxPooling2D((2, 2), padding="same")

        axis = 3 if K.image_data_format() == "channels_last" else 1
        self.concat_1 = tf.keras.layers.Concatenate(axis=axis)
        self.concat_2 = tf.keras.layers.Concatenate(axis=axis)

        self.glob_pool = GlobalPooling(filters)

        self.acf = AggCF_Module(filters, kq_transform=self.acf_kq_transform, value_transform="conv",
                              pooling=self.acf_pool, concat=self.acf_concat, dropout=0.1)
        
        self.final_conv3x3_bn_activation = ConvolutionBnActivation(n_classes, (3, 3), post_activation=final_activation)
        self.final_upsampling2d = tf.keras.layers.UpSampling2D(size=8, interpolation="bilinear")

    def call(self, inputs, training=None, mask=None):
        if training is None:
            training = True

        x0, x1, x2, x3 = self.backbone(inputs, training=training)

        feat = self.conv3x3_bn_relu_1(x3, training=training)
        if self.lateral:
            feat = self.upsample2d_2x(feat)
            c2 = self.conv3x3_bn_relu_2(x1, training=training)
            c2 = self.pool2d(c2)
            c3 = self.conv3x3_bn_relu_3(x2, training=training)
            feat = self.concat_1([feat, c2, c3])
            feat = self.conv3x3_bn_relu_4(feat, training=training)

        if self.global_pool:
            pool = self.glob_pool(feat, training=training)
            feat = self.acf(feat, training=training)
            feat = self.concat_2([pool, feat])
        else:
            feat = self.acf(feat, training=training)

        x = self.final_conv3x3_bn_activation(feat, training=training)
        x = self.final_upsampling2d(x)

        return x

    def model(self):
        x = tf.keras.layers.Input(shape=(self.height, self.width, 3))
        return tf.keras.Model(inputs=[x], outputs=self.call(x))