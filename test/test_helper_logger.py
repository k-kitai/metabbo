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
from   metabbo.helper.logger import Logger, MetadataType, NullMetadataType
from   metabbo import FiniteSampling

import logging
import numpy as np
import os
from   unittest import TestCase

class IntIntText(MetadataType):

    @classmethod
    def sql_formats(cls):
        return ["n1 int", "n2 int", "s text"]


    def __init__(self, n1, n2, s):
        self.n1 = n1
        self.n2 = n2
        self.s = s

    def to_strs(self):
        return [
            "%d"%self.n1,
            "%d"%self.n2,
            "\"%s\""%self.s
        ]

class TestLogger(TestCase):

    def test_create(self):
        logger = Logger([int, int, int, float, float, float], IntIntText)


    def test_logging(self):
        logger = Logger([int, int, int, float, float, float], IntIntText)
        logger.log(
            [    [1,1,1,1.0,1.0,1.0],     [2,2,2,2.0,2.0,2.0]],
            [                    1.0,                     2.0],
            [IntIntText(1,2,"Hello"), IntIntText(3,4,"World")]
        )
        assert len(logger.xs) == 2
        assert len(logger.ys) == 2
        assert len(logger.infos) == 2
        return logger

    def test_save(self):
        logger = self.test_logging()
        try:
            logger.save("test_save.db")
        except Exception as e:
            if e.args[0].startswith("File"):
                logging.info("File already exists. Assuming the test_save had successed before.")
            else:
                raise e
        os.remove("test_save.db")

    def test_load(self):
        logger = self.test_logging()
        logger.save("test_load.db")
        del logger

        logger = Logger.load("test_load.db", [int, int, int, float, float, float], IntIntText)
        assert len(logger.xs) == 2
        assert len(logger.ys) == 2
        assert len(logger.infos) == 2
        assert logger.infos[0].s == "Hello"
        assert logger.infos[1].s == "World"
        os.remove("test_load.db")

        return logger

    def test_logger_compat(self):
        logger = Logger([int, int, int, float, float, float], NullMetadataType)

        try:
            xs = np.random.randint(5, size=(10,6))
            FiniteSampling(xs, lambda xs: np.sum(xs, axis=1), num_random_init=5, randseed=123, logger=logger)
            self.fail()  # An exception should be thrown before this line
        except AssertionError as ae:
            if len(ae.args) > 0 and ae.args[0] == None:
                raise ae
            else:
                pass

        xs = [x.tolist() + y.tolist() for x,y in zip(np.random.randint(5, size=(10,3)), np.random.randn(10,3))]
        FiniteSampling(xs, lambda xs: np.sum(xs, axis=1), num_random_init=5, randseed=123, logger=logger)

