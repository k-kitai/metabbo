#===============================================================================
# Copyright (c) 2021 Koki Kitai
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to
# deal in the Software without restriction, including without limitation the
# rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
# sell copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
# IN THE SOFTWARE.
#===============================================================================
from   abc import ABC, abstractmethod
import numpy as np

class MetaModel(ABC):
    '''
    Abstraction of surrogate models for black-box optimization

    A model should implement a class method `train` and a method `predict`.
    '''

    @classmethod
    @abstractmethod
    def train(cls, xs, ys):
        pass

    @abstractmethod
    def predict(self, xs):
        pass

    def predict_argsort(self, xs, larger_is_better=True, *args, **kwargs):
        scores = self.predict(xs, *args, **kwargs)
        return np.argsort(scores * [1,-1][larger_is_better])

