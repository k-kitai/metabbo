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
from   metabbo.helper import Logger
from   metabbo import FiniteSetSampling, MetaModel
from   metabbo import BinarySpaceSampling, MetaSamplingModel

import logging
import numpy as np
import os
import tempfile
from   unittest import TestCase

class NullModel(MetaModel):
    @classmethod
    def train(cls, xs, ys, to_minimize=False):
        return cls()

    def predict(self, xs):
        return np.random.uniform(size=len(xs))

class TestFiniteSetSampling(TestCase):
    def __init__(self, *args, **kwargs):
        super(TestFiniteSetSampling, self).__init__(*args, **kwargs)

    def test_running(self):
        xs = np.random.randint(10, size=(100, 10))
        def evaluator(xs):
            return np.dot(xs, np.array([1,-1,1,-1,1,-1,1,-1,1,-1]))
        sampling = FiniteSetSampling(xs, evaluator, logger=None)
        sampling.run(NullModel, num_probe=1, num_sampling=10)
        self.assertEqual(10, np.sum(sampling.observed))

    def test_logger(self):
        db_temp_path = tempfile.mktemp()

        xs = np.random.randint(10, size=(100, 10))
        def evaluator(xs):
            return np.dot(xs, np.array([1,-1,1,-1,1,-1,1,-1,1,-1]))
        logger = Logger([int]*10, db_temp_path)
        sampling = FiniteSetSampling(xs, evaluator, logger=logger)
        sampling.run(NullModel, num_probe=1, num_sampling=10)
        self.assertEqual(10, np.sum(sampling.observed))
        self.assertEqual(10, len(logger.xs))
        self.assertEqual(NullModel.__name__, logger.infos[0]['model'])
        del sampling
        del logger

        logger = Logger([int]*10, db_temp_path)
        sampling = FiniteSetSampling(xs, evaluator, logger=logger)
        self.assertEqual(10, np.sum(sampling.observed))
        self.assertEqual(10, len(logger.xs))
        self.assertEqual(NullModel.__name__, logger.infos[0]['model'])
        del sampling
        del logger

        os.remove(db_temp_path)

def two_complement(x, scaling=True):
    '''
    Evaluation function for binary array
    of two's complement representation.

    example (when scaling=False):
    [0,0,0,1] => 1
    [0,0,1,0] => 2
    [0,1,0,0] => 4
    [1,0,0,0] => -8
    [1,1,1,1] => -1
    '''
    val, n = 0, len(x)
    for i in range(n):
        val += (1<<(n-i-1)) * x[i] * (1 if (i>0) else -1)
    return val * (2**(1-n) if scaling else 1)

class NullBinarySpaceSamplingModel(MetaSamplingModel):

    def __init__(self, n, to_minimize=False):
        self.n = n
        self.to_minimize = to_minimize

    @classmethod
    def train(cls, xs, ys, to_minimize=False, n=8):
        return cls(n, to_minimize)

    def predict(self, xs):
        return np.random.uniform(size=len(xs))

    def sample(self, m=1):
        return np.random.randint(2, size=(m, self.n))

class TestBinarySpaceSampling(TestCase):
    def __init__(self, *args, **kwargs):
        super(TestBinarySpaceSampling, self).__init__(*args, **kwargs)

    def test_running(self):
        sampling = BinarySpaceSampling(16, evaluator=two_complement, to_minimize=True)
        sampling.run(NullBinarySpaceSamplingModel, num_probe=3, num_sampling=7, train_args={"n": 16})
        self.assertEqual(21, len(sampling.xs))

    def test_logger(self):
        db_temp_path = tempfile.mktemp()
        logger = Logger([int]*16, db_temp_path)
        sampling = BinarySpaceSampling(16, evaluator=two_complement, logger=logger, to_minimize=True)
        sampling.run(NullBinarySpaceSamplingModel, num_probe=3, num_sampling=7, train_args={"n": 16})
        self.assertEqual(21, len(sampling.xs))
        del sampling
        del logger

        logger = Logger([int]*16, db_temp_path)
        sampling = BinarySpaceSampling(16, evaluator=two_complement, logger=logger, to_minimize=True)
        self.assertEqual(21, len(sampling.xs))
        del sampling
        del logger

        os.remove(db_temp_path)
