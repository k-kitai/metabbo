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

        logger: helper.Logger, default = None
            If povided, load the sampling records from the logger and extend
            them with new samples made in the `run` method.

        to_minimize: bool, default = False
            If True, the optimization is treated as a minimization task.

    '''
    def __init__(self, xs, evaluator, logger=None, to_minimize=False):
        self.xs = np.array(xs, dtype=object)
        self.ys = np.repeat(-np.inf, len(xs))
        self.evaluator = evaluator
        self.observed = np.repeat(False, len(xs))
        self.arange = np.arange(len(xs))
        self.current_step = 1
        self.to_minimize = to_minimize

        self.x_types = list(map(type, self.xs[0]))

        # Connect to logger if provided
        if logger == None:
            self.logger = None
        else:
            assert isinstance(logger, helper.logger.Logger) and logger.x_types == self.x_types
            self.logger = logger
            raw_xs = self.xs.tolist()

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


    def run(self, metamodel_cls, num_probe=1, num_sampling=10, train_args={}, predict_args={}):
        '''
        Sampling from remaining data points based on the prediction by the
        given model.

        Args:
            metamodel_cls: ClassName of an implementation of MetaClass

            num_probe: int
                The number of data points sampled at once.

            num_sampling: int
                The number of samplings in total.

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
                to_minimize=self.to_minimize,
                **train_args
            )
            nexts = self.arange[~self.observed][
                model.predict_argsort(
                    self.xs[~self.observed], **predict_args
                )[:num_probe]
            ]
            self.observed[nexts] = True

            self.ys[nexts] = self.evaluator(self.xs[nexts].tolist())

            if self.logger:
                self.logger.log(self.xs[nexts], self.ys[nexts], [{"step": self.current_step, "model": metamodel_cls.__name__} for _ in range(len(nexts))])
            self.current_step += 1
        return

class BinarySpaceSampling():
    '''
    Managing a sampling process over the binary space

    Args:
        n: int
            The dimension of the search space

        evaluator: function or callable object
            Given a list of a subset of `xs`, this should return their
            respective score.

        logger: helper.Logger, default = None
            If povided, load the sampling records from the logger and extend
            them with new samples made in the `run` method.

        to_minimize: bool, default = False
            If True, the optimization is treated as a minimization task.

    '''
    def __init__(self, n, evaluator, logger=None, to_minimize=False):
        self.n = n
        self.xs = []
        self.ys = []
        self.evaluator = evaluator
        self.current_step = 1
        self.to_minimize = to_minimize

        self.x_types = [int for _ in range(n)]

        # Connect to logger if provided
        if logger == None:
            self.logger = None
        else:
            assert isinstance(logger, helper.logger.Logger) and logger.x_types == self.x_types
            self.logger = logger

            if len(logger.xs) > 0:
                self.current_step = max(map(lambda d: d['step'], logger.infos)) + 1
            else:
                self.current_step = 1

            for x, y in zip(logger.xs, logger.ys):
                try:
                    self.xs.append(x)
                    self.ys.append(y)
                except:
                    pass

    def run(self, metamodel_cls, num_probe=1, num_sampling=10, train_args={}, predict_args={}):
        '''
        Sampling data points by the `sample` method of the given model

        Args:
            metamodel: model with which sampling is done
                A name of class which implements MetaModel, or an instance of a class
                which implements MetaSamplingModel.

            num_probe: int
                The number of data points sampled at once.

            num_sampling: int
                The number of samplings in total.

            train_args: dictionary, optional
                The arbitrary keyword arguments passed to the training of the model.

            predict_args: dictionary, optional
                The arbitrary keyword arguments passed to the prediction by the model.

        Returns:
            nothing
        '''
        clsname = hasattr(metamodel_cls, "__name__") and metamodel_cls.__name__  or metamodel_cls.__class__.__name__
        for _ in range(num_sampling):
            model = metamodel_cls.train(
                self.xs, self.ys,
                to_minimize=self.to_minimize,
                **train_args
            )
            nexts = np.array(model.sample(num_probe)).tolist()
            nexts = list(filter(lambda x: x not in self.xs, nexts))
            if len(nexts) == 0:
                continue
            self.xs.extend(nexts)
            for x in nexts:
                self.ys.append(self.evaluator(x))

            if self.logger:
                self.logger.log(nexts, self.ys[-len(nexts):], [{"step": self.current_step, "model": clsname} for _ in range(len(nexts))])
            self.current_step += 1
        return

