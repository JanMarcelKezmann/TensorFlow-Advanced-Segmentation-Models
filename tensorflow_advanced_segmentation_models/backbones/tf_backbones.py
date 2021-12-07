import tensorflow as tf

################################################################################
# Backbone
################################################################################
def create_base_model(name="ResNet50", weights="imagenet", height=None, width=None,
                      channels=3, include_top=False, pooling=None, alpha=1.0,
                      depth_multiplier=1, dropout=0.001):
    if not isinstance(height, int) or not isinstance(width, int) or not isinstance(channels, int):
        raise TypeError("'height', 'width' and 'channels' need to be of type 'int'")
        
    if channels <= 0:
        raise ValueError(f"'channels' must be greater of equal to 1 but given was {channels}")
    
    input_shape = [height, width, channels]

    if name.lower() == "densenet121":
        if height <= 31 or width <= 31:
            raise ValueError("Parameters 'height' and 'width' should not be smaller than 32.")
        base_model = tf.keras.applications.DenseNet121(include_top=include_top, weights=weights, input_shape=input_shape, pooling=pooling)
        layer_names = ["conv1/relu", "pool2_relu", "pool3_relu", "pool4_relu", "relu"]
    elif name.lower() == "densenet169":
        if height <= 31 or width <= 31:
            raise ValueError("Parameters 'height' and 'width' should not be smaller than 32.")
        base_model = tf.keras.applications.DenseNet169(include_top=include_top, weights=weights, input_shape=input_shape, pooling=pooling)
        layer_names = ["conv1/relu", "pool2_relu", "pool3_relu", "pool4_relu", "relu"]
    elif name.lower() == "densenet201":
        if height <= 31 or width <= 31:
            raise ValueError("Parameters 'height' and 'width' should not be smaller than 32.")
        base_model = tf.keras.applications.DenseNet201(include_top=include_top, weights=weights, input_shape=input_shape, pooling=pooling)
        layer_names = ["conv1/relu", "pool2_relu", "pool3_relu", "pool4_relu", "relu"]
    elif name.lower() == "efficientnetb0":
        if height <= 31 or width <= 31:
            raise ValueError("Parameters 'height' and 'width' should not be smaller than 32.")
        base_model = tf.keras.applications.EfficientNetB0(include_top=include_top, weights=weights, input_shape=input_shape, pooling=pooling)
        layer_names = ["block2a_expand_activation", "block3a_expand_activation", "block4a_expand_activation", "block6a_expand_activation", "top_activation"]
    elif name.lower() == "efficientnetb1":
        if height <= 31 or width <= 31:
            raise ValueError("Parameters 'height' and 'width' should not be smaller than 32.")
        base_model = tf.keras.applications.EfficientNetB1(include_top=include_top, weights=weights, input_shape=input_shape, pooling=pooling)
        layer_names = ["block2a_expand_activation", "block3a_expand_activation", "block4a_expand_activation", "block6a_expand_activation", "top_activation"]
    elif name.lower() == "efficientnetb2":
        if height <= 31 or width <= 31:
            raise ValueError("Parameters 'height' and 'width' should not be smaller than 32.")
        base_model = tf.keras.applications.EfficientNetB2(include_top=include_top, weights=weights, input_shape=input_shape, pooling=pooling)
        layer_names = ["block2a_expand_activation", "block3a_expand_activation", "block4a_expand_activation", "block6a_expand_activation", "top_activation"]
    elif name.lower() == "efficientnetb3":
        if height <= 31 or width <= 31:
            raise ValueError("Parameters 'height' and 'width' should not be smaller than 32.")
        base_model = tf.keras.applications.EfficientNetB3(include_top=include_top, weights=weights, input_shape=input_shape, pooling=pooling)
        layer_names = ["block2a_expand_activation", "block3a_expand_activation", "block4a_expand_activation", "block6a_expand_activation", "top_activation"]
    elif name.lower() == "efficientnetb4":
        if height <= 31 or width <= 31:
            raise ValueError("Parameters 'height' and 'width' should not be smaller than 32.")
        base_model = tf.keras.applications.EfficientNetB4(include_top=include_top, weights=weights, input_shape=input_shape, pooling=pooling)
        layer_names = ["block2a_expand_activation", "block3a_expand_activation", "block4a_expand_activation", "block6a_expand_activation", "top_activation"]
    elif name.lower() == "efficientnetb5":
        if height <= 31 or width <= 31:
            raise ValueError("Parameters 'height' and 'width' should not be smaller than 32.")
        base_model = tf.keras.applications.EfficientNetB5(include_top=include_top, weights=weights, input_shape=input_shape, pooling=pooling)
        layer_names = ["block2a_expand_activation", "block3a_expand_activation", "block4a_expand_activation", "block6a_expand_activation", "top_activation"]
    elif name.lower() == "efficientnetb6":
        if height <= 31 or width <= 31:
            raise ValueError("Parameters 'height' and 'width' should not be smaller than 32.")
        base_model = tf.keras.applications.EfficientNetB6(include_top=include_top, weights=weights, input_shape=input_shape, pooling=pooling)
        layer_names = ["block2a_expand_activation", "block3a_expand_activation", "block4a_expand_activation", "block6a_expand_activation", "top_activation"]
    elif name.lower() == "efficientnetb7":
        if height <= 31 or width <= 31:
            raise ValueError("Parameters 'height' and 'width' should not be smaller than 32.")
        base_model = tf.keras.applications.EfficientNetB7(include_top=include_top, weights=weights, input_shape=input_shape, pooling=pooling)
        layer_names = ["block2a_expand_activation", "block3a_expand_activation", "block4a_expand_activation", "block6a_expand_activation", "top_activation"]
    elif name.lower() == "mobilenet":
        if height <= 31 or width <= 31:
            raise ValueError("Parameters 'height' and 'width' should not be smaller than 32.")
        base_model = tf.keras.applications.MobileNet(include_top=include_top, weights=weights, input_shape=input_shape, pooling=pooling, alpha=alpha, depth_multiplier=depth_multiplier, dropout=dropout)
        layer_names = ["conv_pw_1_relu", "conv_pw_3_relu", "conv_pw_5_relu", "conv_pw_11_relu", "conv_pw_13_relu"]
    elif name.lower() == "mobilenetv2":
        if height <= 31 or width <= 31:
            raise ValueError("Parameters 'height' and 'width' should not be smaller than 32.")
        base_model = tf.keras.applications.MobileNetV2(include_top=include_top, weights=weights, input_shape=input_shape, pooling=pooling, alpha=alpha)
        layer_names = ["block_1_expand_relu", "block_3_expand_relu", "block_6_expand_relu", "block_13_expand_relu", "out_relu"]
    elif name.lower() == "nasnetlarge":
        if height <= 31 or width <= 31:
            raise ValueError("Parameters 'height' and 'width' should not be smaller than 32.")
        base_model = tf.keras.applications.NASNetLarge(include_top=include_top, weights=weights, input_shape=input_shape, pooling=pooling)
        layer_names = ["zero_padding2d", "cropping2d_1", "cropping2d_2", "cropping2d_3", "activation_650"]
    elif name.lower() == "nasnetmobile":
        if height <= 31 or width <= 31:
            raise ValueError("Parameters 'height' and 'width' should not be smaller than 32.")
        base_model = tf.keras.applications.NASNetMobile(include_top=include_top, weights=weights, input_shape=input_shape, pooling=pooling)
        layer_names =["zero_padding2d_4", "cropping2d_5", "cropping2d_6", "cropping2d_7", "activation_838"]
    elif name.lower() == "resnet50":
        if height <= 31 or width <= 31:
            raise ValueError("Parameters 'height' and 'width' should not be smaller than 32.")
        base_model = tf.keras.applications.ResNet50(include_top=include_top, weights=weights, input_shape=input_shape, pooling=pooling)
        layer_names = ["conv1_relu", "conv2_block3_out", "conv3_block4_out", "conv4_block6_out", "conv5_block3_out"]
    elif name.lower() == "resnet50v2":
        if height <= 31 or width <= 31:
            raise ValueError("Parameters 'height' and 'width' should not be smaller than 32.")
        base_model = tf.keras.applications.ResNet50V2(include_top=include_top, weights=weights, input_shape=input_shape, pooling=pooling)
        layer_names = ["conv1_conv", "conv2_block3_preact_relu", "conv3_block4_preact_relu", "conv4_block6_preact_relu", "post_relu"]
    elif name.lower() == "resnet101":
        if height <= 31 or width <= 31:
            raise ValueError("Parameters 'height' and 'width' should not be smaller than 32.")
        base_model = tf.keras.applications.ResNet101(include_top=include_top, weights=weights, input_shape=input_shape, pooling=pooling)
        layer_names = ["conv1_relu", "conv2_block3_out", "conv3_block4_out", "conv4_block23_out", "conv5_block3_out"]
    elif name.lower() == "resnet101v2":
        if height <= 31 or width <= 31:
            raise ValueError("Parameters 'height' and 'width' should not be smaller than 32.")
        base_model = tf.keras.applications.ResNet101V2(include_top=include_top, weights=weights, input_shape=input_shape, pooling=pooling)
        layer_names = ["conv1_conv", "conv2_block3_preact_relu", "conv3_block4_preact_relu", "conv4_block23_preact_relu", "post_relu"]
    elif name.lower() == "resnet152":
        if height <= 31 or width <= 31:
            raise ValueError("Parameters 'height' and 'width' should not be smaller than 32.")
        base_model = tf.keras.applications.ResNet152(include_top=include_top, weights=weights, input_shape=input_shape, pooling=pooling)
        layer_names = ["conv1_relu", "conv2_block3_out", "conv3_block8_out", "conv4_block36_out", "conv5_block3_out"]
    elif name.lower() == "resnet152v2":
        if height <= 31 or width <= 31:
            raise ValueError("Parameters 'height' and 'width' should not be smaller than 32.")
        base_model = tf.keras.applications.ResNet152V2(include_top=include_top, weights=weights, input_shape=input_shape, pooling=pooling)
        layer_names = ["conv1_conv", "conv2_block3_preact_relu", "conv3_block8_preact_relu", "conv4_block36_preact_relu", "post_relu"]
    elif name.lower() == "vgg16":
        if height <= 31 or width <= 31:
            raise ValueError("Parameters 'height' and 'width' should not be smaller than 32.")
        base_model = tf.keras.applications.VGG16(include_top=include_top, weights=weights, input_shape=input_shape, pooling=pooling)
        layer_names = ["block2_conv2", "block3_conv3", "block4_conv3", "block5_conv3", "block5_pool"]
    elif name.lower() == "vgg19":
        if height <= 31 or width <= 31:
            raise ValueError("Parameters 'height' and 'width' should not be smaller than 32.")
        base_model = tf.keras.applications.VGG19(include_top=include_top, weights=weights, input_shape=input_shape, pooling=pooling)
        layer_names = ["block2_conv2", "block3_conv4", "block4_conv4", "block5_conv4", "block5_pool"]
    elif name.lower() == "xception":
        if height <= 70 or width <= 70:
            raise ValueError("Parameters 'height' and width' should not be smaller than 71.")
        base_model = tf.keras.applications.Xception(include_top=include_top, weights=weights, input_shape=input_shape, pooling=pooling)
        layer_names = ["block2_sepconv2_act", "block3_sepconv2_act", "block4_sepconv2_act", "block13_sepconv2_act", "block14_sepconv2_act"]
    else:
        raise ValueError("'name' should be one of 'densenet121', 'densenet169', 'densenet201', 'efficientnetb0', 'efficientnetb1', 'efficientnetb2', \
                'efficientnetb3', 'efficientnetb4', 'efficientnetb5', 'efficientnetb6', 'efficientnetb7','mobilenet', 'mobilenetv2', 'nasnetlarge', 'nasnetmobile', \
                'resnet50', 'resnet50v2', 'resnet101', 'resnet101v2', 'resnet152', 'resnet152v2', 'vgg16', 'vgg19' or 'xception'.")
    
    layers = [base_model.get_layer(layer_name).output for layer_name in layer_names]

    return base_model, layers, layer_names