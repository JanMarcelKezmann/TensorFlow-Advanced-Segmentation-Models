import tensorflow.keras.backend as K

################################################################################
# Helper Functions
################################################################################
def average(x, class_weights=None):
    if class_weights is not None:
        x = x * class_weights
    return K.mean(x)

def gather_channels(*xs):
    return xs

def round_if_needed(x, threshold):
    if threshold is not None:
        x = K.greater(x, threshold)
        x = K.cast(x, K.floatx())
    return x

################################################################################
# Metric Functions
################################################################################
def iou_score(y_true, y_pred, class_weights=1., smooth=1e-5, threshold=None):    
    # y_true = K.one_hot(K.squeeze(K.cast(y_true, tf.int32), axis=-1), n_classes)

    y_true, y_pred = gather_channels(y_true, y_pred)
    y_pred = round_if_needed(y_pred, threshold)
    axes = [1, 2] if K.image_data_format() == "channels_last" else [2, 3]
    
    intersection = K.sum(y_true * y_pred, axis=axes)
    union = K.sum(y_true + y_pred, axis=axes) - intersection

    score = (intersection + smooth) / (union + smooth)
    score = average(score, class_weights)

    return score

def dice_coefficient(y_true, y_pred, beta=1.0, class_weights=1., smooth=1e-5, threshold=None):
    # print(y_pred)
    y_true, y_pred = gather_channels(y_true, y_pred)
    y_pred = round_if_needed(y_pred, threshold)
    axes = [1, 2] if K.image_data_format() == "channels_last" else [2, 3]
    
    tp = K.sum(y_true * y_pred, axis=axes)
    fp = K.sum(y_pred, axis=axes) - tp
    fn = K.sum(y_true, axis=axes) - tp

    score = ((1.0 + beta) * tp + smooth) / ((1.0 + beta) * tp + (beta ** 2.0) * fn + fp + smooth)
    # print("Score, wo avg: " + str(score))
    score = average(score, class_weights)
    # print("Score: " + str(score))

    return score

def precision(y_true, y_pred, class_weights=1., smooth=1e-5, threshold=None):
    y_true, y_pred = gather_channels(y_true, y_pred)
    y_pred = round_if_needed(y_pred, threshold)
    axes = [1, 2] if K.image_data_format() == "channels_last" else [2, 3]
    
    tp = K.sum(y_true * y_pred, axis=axes)
    fp = K.sum(y_pred, axis=axes) - tp

    score = (tp + smooth) / (tp + fp + smooth)
    score = average(score, class_weights)

    return score

def recall(y_true, y_pred, class_weights=1., smooth=1e-5, threshold=None):
    y_true, y_pred = gather_channels(y_true, y_pred)
    y_pred = round_if_needed(y_pred, threshold)
    axes = [1, 2] if K.image_data_format() == "channels_last" else [2, 3]
    
    tp = K.sum(y_true * y_pred, axis=axes)
    fn = K.sum(y_true, axis=axes) - tp

    score = (tp + smooth) / (tp + fn + smooth)
    score = average(score, class_weights)

    return score

def tversky(y_true, y_pred, alpha=0.7, class_weights=1., smooth=1e-5, threshold=None):
    y_true, y_pred = gather_channels(y_true, y_pred)
    y_pred = round_if_needed(y_pred, threshold)
    axes = [1, 2] if K.image_data_format() == "channels_last" else [2, 3]
    
    tp = K.sum(y_true * y_pred, axis=axes)
    fp = K.sum(y_pred, axis=axes) - tp
    fn = K.sum(y_true, axis=axes) - tp

    score = (tp + smooth) / (tp + alpha * fn + (1 - alpha) * fp + smooth)
    score = average(score, class_weights)

    return score


################################################################################
# Loss Functions
################################################################################
def categorical_crossentropy(y_true, y_pred, class_weights=1.):
    y_true, y_pred = gather_channels(y_true, y_pred)

    axis = 3 if K.image_data_format() == "channels_last" else 1
    y_pred /= K.sum(y_pred, axis=axis, keepdims=True)

    y_pred = K.clip(y_pred, K.epsilon(), 1.0 - K.epsilon())

    loss = y_true * K.log(y_pred) * class_weights
    return - K.mean(loss)

def binary_crossentropy(y_true, y_pred):
    return K.mean(K.binary_crossentropy(y_true, y_pred))

def categorical_focal_loss(y_true, y_pred, gamma=2.0, alpha=0.25):
    y_true, y_pred = gather_channels(y_true, y_pred)
    y_true = K.cast(y_true, K.floatx())
    y_pred = K.clip(y_pred, K.epsilon(), 1.0 - K.epsilon())

    loss = - y_true * (alpha * K.pow((1 - y_pred), gamma) * K.log(y_pred))

    return K.mean(loss)

# def categorical_focal_dice_loss(y_true, y_pred, gamma=2.0, alpha=0.25, beta=1.0, class_weights=1., smooth=1e-5, threshold=None):
#     dice_score = dice_coefficient(y_true, y_pred, beta=beta, class_weights=class_weights, smooth=smooth, threshold=threshold)

#     cat_focal_loss = categorical_focal_loss(y_true, y_pred, gamma=gamma, alpha=alpha)
#     return dice_loss + cat_focal_loss

def binary_focal_loss(y_true, y_pred, gamma=2.0, alpha=0.25):
    y_pred = K.clip(y_pred, K.epsilon(), 1.0 - K.epsilon())

    loss_a = - y_true * (alpha * K.pow((1 - y_pred), gamma) * K.log(y_pred))
    loss_b = - (1 - y_true) * ((1 - alpha) * K.pow((y_pred), gamma) * K.log(1 - y_pred))
    
    return K.mean(loss_a + loss_b)

def combo(y_true, y_pred, alpha=0.5, beta=1.0, ce_ratio=0.5, class_weights=1., smooth=1e-5, threshold=None):
    # alpha < 0.5 penalizes FP more, alpha > 0.5 penalizes FN more

    y_true, y_pred = gather_channels(y_true, y_pred)
    y_pred = round_if_needed(y_pred, threshold)
    axes = [1, 2] if K.image_data_format() == "channels_last" else [2, 3]
    
    tp = K.sum(y_true * y_pred, axis=axes)
    fp = K.sum(y_pred, axis=axes) - tp
    fn = K.sum(y_true, axis=axes) - tp

    dice = ((1.0 + beta) * tp + smooth) / ((1.0 + beta) * tp + (beta ** 2.0) * fn + fp + smooth)

    y_pred = K.clip(y_pred, K.epsilon(), 1.0 - K.epsilon())

    ce = - (alpha * (y_true * K.log(y_pred))) + ((1 - alpha) * (1.0 - y_true) * K.log(1.0 - y_pred))
    ce = K.mean(ce, axis=axes)

    combo = (ce_ratio * ce) - ((1 - ce_ratio) * dice)
    loss = average(combo, class_weights)

    return loss
