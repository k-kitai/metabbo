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
from   metabbo import FiniteSetSampling

import logging
import numpy as np
from   unittest import TestCase

class TestLogger(TestCase):

    def test_logging(self):
        logger = Logger([int, int, int, float, float, float])
        logger.log(
            [    [1,1,1,1.0,1.0,1.0],     [2,2,2,2.0,2.0,2.0]],
            [                    1.0,                     2.0],
            [{"index": 0, "order": "first"}, {"index": 1, "order": "second"}]
        )
        del logger

    def test_attribute(self):
        logger = Logger([int, float])
        logger.log([[1, 2.0], [2, 3.0]], [2.0, 6.0], [{"say": "Hello"}, {"objective": "World"}])
        logger.log1([3, 4.0], 12.0, {"tension": "!"})
        xs = logger.xs
        self.assertEqual(3, len(xs))
        self.assertEqual([1, 2.0], xs[0])
        self.assertEqual([2, 3.0], xs[1])
        self.assertEqual([3, 4.0], xs[2])
        ys = logger.ys
        self.assertEqual(3, len(ys))
        self.assertEqual(2.0, ys[0])
        self.assertEqual(6.0, ys[1])
        self.assertEqual(12.0, ys[2])
        infos = logger.infos
        self.assertEqual(3, len(infos))
        self.assertEqual("Hello", infos[0]["say"])
        self.assertEqual("World", infos[1]["objective"])
        self.assertEqual("!", infos[2]["tension"])
