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
import json
import os
import sqlite3

class Logger:
    def __init__(self, x_types, fname=":memory:"):
        self.x_types = x_types
        self.x_types_format = ", ".join(map(lambda t: {int: "%d", float: "%e"}[t], x_types))

        #if os.path.exists(fname):
        #    raise Exception("File {} already exists".format(fname))
        self.db = sqlite3.connect(fname, timeout=60)

        self.db.execute("CREATE TABLE IF NOT EXISTS Logs ("
            + ", ".join(["x%d %s"%(n, t.__name__) for (n, t) in enumerate(self.x_types)])
            + ", y float, info text);")

        self.db.commit()

    def _format_x(self, x):
        return [f(el) for (f, el) in zip(self.x_types, x)]

    def log1(self, x, y, info={}):
        self.db.execute("INSERT INTO Logs VALUES (" + self.x_types_format%tuple(x) + ", %e, '%s');"%(y, json.dumps(info)))

    def log(self, xs, ys, infos=None):
        assert len(xs) == len(ys) and (infos == None or len(xs) == len(infos)), "xs and ys(, and infos) must have the same length."
        for x, y, info in zip(xs, ys, infos or [{} for _ in range(len(xs))]):
            self.log1(x, y, info)

    def __del__(self):
        self.db.commit()
        self.db.close()

