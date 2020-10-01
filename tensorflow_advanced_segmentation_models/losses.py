from .base import Loss
from .base.functional import *

################################################################################
# Losses
################################################################################
class JaccardLoss(Loss):
    def __init__(self, class_weights=None, smooth=1e-5):
        super().__init__(name="jaccard_loss")
        self.class_weights = class_weights if class_weights is not None else 1.0
        self.smooth = smooth

    def __call__(self, y_true, y_pred):
        return 1.0 - iou_score(
            y_true,
            y_pred,
            class_weights=self.class_weights,
            smooth=self.smooth,
            threshold=None
        )

class DiceLoss(Loss):
    def __init__(self, beta=1.0, class_weights=None, smooth=1e-5):
        super().__init__(name="dice_loss")
        self.beta = beta
        self.class_weights = class_weights if class_weights is not None else 1.0
        self.smooth = smooth

    def __call__(self, y_true, y_pred):
        # print(y_pred)
        return 1.0 - dice_coefficient(
            y_true,
            y_pred,
            beta=self.beta,
            class_weights=self.class_weights,
            smooth=self.smooth,
            threshold=None
        )


class TverskyLoss(Loss):
    def __init__(self, alpha=0.7, class_weights=None, smooth=1e-5):
        super().__init__(name="tversky_loss")
        self.alpha = alpha
        self.class_weights = class_weights if class_weights is not None else 1.0
        self.smooth = smooth

    def __call__(self, y_true, y_pred):
        return 1.0 - tversky(
            y_true,
            y_pred,
            alpha=self.alpha,
            class_weights=self.class_weights,
            smooth=self.smooth,
            threshold=None
        )


class FocalTverskyLoss(Loss):
    def __init__(self, alpha=0.7, gamma=1.25, class_weights=None, smooth=1e-5):
        super().__init__(name="focal_tversky_loss")
        self.alpha = alpha
        self.gamma = gamma
        self.class_weights = class_weights if class_weights is not None else 1.0
        self.smooth = smooth

    def __call__(self, y_true, y_pred):
        return K.pow((1.0 - tversky(
            y_true,
            y_pred,
            alpha=self.alpha,
            class_weights=self.class_weights,
            smooth=self.smooth,
            threshold=None
        )),
        self.gamma
        )


class BinaryCELoss(Loss):
    def __init__(self):
        super().__init__(name="binary_crossentropy")

    def __call__(self, y_true, y_pred):
        return binary_crossentropy(
            y_true,
            y_pred
        )


class CategoricalCELoss(Loss):
    def __init__(self, class_weights=None):
        super().__init__(name="categorical_crossentropy")
        self.class_weights = class_weights

    def __call__(self, y_true, y_pred):
        return categorical_crossentropy(
            y_true,
            y_pred,
            class_weights=self.class_weights
        )

class CategoricalFocalLoss(Loss):
    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__(name="focal_loss")
        self.alpha = alpha
        self.gamma = gamma

    def __call__(self, y_true, y_pred):
        return categorical_focal_loss(
            y_true,
            y_pred,
            alpha=self.alpha,
            gamma=self.gamma
        )

class BinaryFocalLoss(Loss):
    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__(name='binary_focal_loss')
        self.alpha = alpha
        self.gamma = gamma

    def __call__(self, y_true, y_pred):
        return binary_focal_loss(y_true, y_pred, alpha=self.alpha, gamma=self.gamma)


class ComboLoss(Loss):
    def __init__(self, alpha=0.5, beta=1.0, ce_ratio=0.5, class_weights=None, smooth=1e-5):
        super().__init__(name="combo_loss")
        self.alpha = alpha
        self.beta = beta
        self.ce_ratio = ce_ratio
        self.class_weights = class_weights if class_weights is not None else 1.0
        self.smooth = smooth

    def __call__(self, y_true, y_pred):
        return combo(
            y_true,
            y_pred,
            alpha=self.alpha,
            beta=self.beta,
            ce_ratio=self.ce_ratio,
            class_weights=self.class_weights,
            smooth=self.smooth,
            threshold=None
        )