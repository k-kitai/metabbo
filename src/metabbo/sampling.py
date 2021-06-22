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
import logging
import numpy as np

from . import helper

class FiniteSetSampling():
    '''
    Managing a sampling process from a set of finite points to obtain the best

    Args:
        xs: numpy.ndarray or a list of candidates
            All the candidates under the scope of sampling.

        evaluator: function or callable object
            Given a list of a subset of `xs`, this should return their
            respective score.

        num_random_init: int, default = 10
            The number of data points to be evaluated at first for the training
            of regression models.

        randseed: int, default = 123
            The random seed for deciding the `num_random_init` data points.

        logger: helper.Logger, default = None
            If povided, load the sampling records from the logger and extend
            them with new samples made in the `run` method.
    '''
    def __init__(self, xs, evaluator, logger=None):
        self.xs = np.array(xs, dtype=object)
        self.ys = np.repeat(-np.inf, len(xs))
        self.evaluator = evaluator
        self.observed = np.repeat(False, len(xs))
        self.arange = np.arange(len(xs))
        self.current_step = 1

        self.x_types = list(map(type, self.xs[0]))

        # Connect to logger if provided
        if logger == None:
            self.logger = None
        else:
            assert isinstance(logger, helper.logger.Logger) and logger.x_types == self.x_types
            self.logger = logger
            raw_xs = xs.tolist()

            if len(logger.xs) > 0:
                self.current_step = max(map(lambda d: d['step'], logger.infos)) + 1
            else:
                self.current_step = 1

            for x, y in zip(logger.xs, logger.ys):
                try:
                    n = raw_xs.index(x)
                    self.ys[n] = y
                    self.observed[n] = True
                except:
                    pass


    def run(self, metamodel_cls, num_probe=1, num_sampling=10, larger_is_better=True, train_args={}, predict_args={}):
        '''
        Sampling from remaining data points based on the prediction by the
        given model.

        Args:
            metamodel_cls: ClassName of an implementation of MetaClass

            num_probe: int
                The number of data points sampled at once.

            num_sampling: int
                The number of samplings in total.

            larger_is_better: bool, default = True
                If True, the optimization is a maximization task.

            train_args: dictionary, optional
                The arbitrary keyword arguments passed to the training of the model.

            predict_args: dictionary, optional
                The arbitrary keyword arguments passed to the prediction by the model.

        Returns:
            nothing
        '''
        for _ in range(num_sampling):
            if np.all(self.observed):
                logging.error("Data points are already exhausted")
                break
            model = metamodel_cls.train(
                self.xs[self.observed], self.ys[self.observed],
                **train_args
            )
            nexts = self.arange[~self.observed][
                model.predict_argsort(
                    self.xs[~self.observed], larger_is_better, **predict_args
                )[:num_probe]
            ]
            self.observed[nexts] = True

            self.ys[nexts] = self.evaluator(self.xs[nexts])

            if self.logger:
                self.logger.log(self.xs[nexts], self.ys[nexts], [{"step": self.current_step, "model": metamodel_cls.__name__} for _ in range(len(nexts))])
            self.current_step += 1
        return

