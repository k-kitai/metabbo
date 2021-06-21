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
import os
import sqlite3

class MetadataType(ABC):

    @classmethod
    @abstractmethod
    def sql_formats(cls):
        '''This method should return a list of element typenames for SQL
        '''
        pass

    @abstractmethod
    def to_strs(self):
        '''This method should return a list of str exression of all elements
        '''
        pass

class NullMetadataType(MetadataType):
    @classmethod
    def sql_formats(cls):
        return []

    def to_strs():
        return []

class Logger:
    def __init__(self, x_types, metadata_type):
        assert issubclass(metadata_type, MetadataType)
        self.xs = []
        self.ys = []
        self.infos = []
        self.x_types = x_types
        self.metadata_type = metadata_type

    def _format_x(self, x):
        return [f(el) for (f, el) in zip(self.x_types, x)]

    def log1(self, x, y, info):
        assert isinstance(info, self.metadata_type)
        self.xs.append(self._format_x(x))
        self.ys.append(float(y))
        self.infos.append(info)

    def log(self, xs, ys, infos):
        assert len(xs) == len(ys) and len(xs) == len(infos), "xs, ys, and infos must have the same length."
        assert all(filter(lambda i: isinstance(i, self.metadata_type), infos))

        self.xs.extend(map(self._format_x, xs))
        self.ys.extend(map(float, ys))
        self.infos.extend(infos)

    def save(self, fname):
        if os.path.exists(fname):
            raise Exception("File {} already exists".format(fname))
        db = sqlite3.connect(fname, timeout=60)

        db.execute("CREATE TABLE Log ("
            + ", ".join(["x%d %s"%(n, t.__name__) for (n, t) in enumerate(self.x_types)])
            + ", y float, "
            + ", ".join(self.metadata_type.sql_formats())
            + ");")

        for x, y, info in zip(self.xs, self.ys, self.infos):
            db.execute("INSERT INTO Log VALUES ("
                + ", ".join(map(lambda el: {int: "%d", float: "%e"}[type(el)]%el, x))
                + ", %e, " % y
                + ", ".join(info.to_strs())
                + ");")
        db.commit()

    @classmethod
    def load(cls, fname, x_types, metadata_type):
        obj = cls(x_types, metadata_type)
        with sqlite3.connect(fname) as conn:
            xlen = len(obj.x_types)
            for xyinfo in conn.execute("SELECT * from Log"):
                x, y, info = xyinfo[:xlen], xyinfo[xlen], metadata_type(*xyinfo[xlen+1:])
                obj.log1(x, y, info)

        return obj

