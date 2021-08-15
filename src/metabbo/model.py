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
from   typing import Any, List

class MetaModel(ABC):
    '''
    Abstraction of surrogate models for black-box optimization
    '''
    pass

class MetaSelectiveModel(MetaModel):
    @abstractmethod
    def argsort(self, xs) -> List[int]:
        '''
        This method returns the permutation list of indices,
        whose order represents the preference for the next evaluations.
        
        Returns:
            sorted_indices: [int]
            
            metadata: [dict]
                Metadata for each element of sorted_indices. Just empty list is ok.
        '''
        pass

class MetaGenerativeModel(MetaModel):
    '''
    SamplingModels return possible solutions without candidates provided
    '''
    @abstractmethod
    def sample(self) -> List[List[Any]]:
        '''
        This method suggests the candidate configurations
        for the next evaluations.

        Returns:
            samples: [Vector]
                Samples for next evaluations.
            
            metadata: [dict]
                Metadata for each element of samples. Just empty list is ok.
        '''
        pass
