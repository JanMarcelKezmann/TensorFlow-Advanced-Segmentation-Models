"""
The MIT License

Copyright (c) 2018, Pavel Yakubovskiy

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
"""
class KerasObject:
    _backend = None
    _models = None
    _layers = None
    _utils = None

    def __init__(self, name=None):
        if (self.backend is None or
                self.utils is None or
                self.models is None or
                self.layers is None):
            raise RuntimeError('You cannot use `KerasObjects` with None submodules.')

        self._name = name

    @property
    def __name__(self):
        if self._name is None:
            return self.__class__.__name__
        return self._name

    @property
    def name(self):
        return self.__name__

    @name.setter
    def name(self, name):
        self._name = name

    @classmethod
    def set_submodules(cls, backend, layers, models, utils):
        cls._backend = backend
        cls._layers = layers
        cls._models = models
        cls._utils = utils

    @property
    def submodules(self):
        return {
            'backend': self.backend,
            'layers': self.layers,
            'models': self.models,
            'utils': self.utils,
        }

    @property
    def backend(self):
        return self._backend

    @property
    def layers(self):
        return self._layers

    @property
    def models(self):
        return self._models

    @property
    def utils(self):
        return self._utils


class Metric(KerasObject):
    pass


class Loss(KerasObject):

    def __add__(self, other):
        if isinstance(other, Loss):
            return SumOfLosses(self, other)
        else:
            raise ValueError('Loss should be inherited from `Loss` class')

    def __radd__(self, other):
        return self.__add__(other)

    def __mul__(self, value):
        if isinstance(value, (int, float)):
            return MultipliedLoss(self, value)
        else:
            raise ValueError('Loss should be inherited from `BaseLoss` class')

    def __rmul__(self, other):
        return self.__mul__(other)


class MultipliedLoss(Loss):

    def __init__(self, loss, multiplier):

        # resolve name
        if len(loss.__name__.split('+')) > 1:
            name = '{}({})'.format(multiplier, loss.__name__)
        else:
            name = '{}{}'.format(multiplier, loss.__name__)
        super().__init__(name=name)
        self.loss = loss
        self.multiplier = multiplier

    def __call__(self, gt, pr):
        return self.multiplier * self.loss(gt, pr)


class SumOfLosses(Loss):

    def __init__(self, l1, l2):
        name = '{}_plus_{}'.format(l1.__name__, l2.__name__)
        super().__init__(name=name)
        self.l1 = l1
        self.l2 = l2

    def __call__(self, gt, pr):
        return self.l1(gt, pr) + self.l2(gt, pr)