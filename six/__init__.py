# -*- coding: utf-8 -*-
"""six: Python 2 and 3 compatibility library."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import sys
import types

PY2 = sys.version_info[0] == 2
PY3 = sys.version_info[0] == 3

# 为Python 3环境定义兼容性变量
try:
    # 尝试使用Python 2的类型
    unicode = unicode  # type: ignore
    long = long  # type: ignore
    unichr = unichr  # type: ignore
except NameError:
    # 在Python 3中定义替代品
    unicode = str
    long = int
    unichr = chr

if PY3:
    text_type = str
    binary_type = bytes
    string_types = (str,)
    integer_types = (int,)
    class_types = (type,)
    text_to_binary = lambda s, enc: s.encode(enc)
else:
    text_type = unicode
    binary_type = str
    string_types = (str, unicode)
    integer_types = (int, long)
    class_types = (type, types.ClassType)
    text_to_binary = lambda s, enc: s

# Metaclass decorator
def add_metaclass(metaclass):
    """Class decorator for creating a class with a metaclass."""
    def wrapper(cls):
        orig_vars = cls.__dict__.copy()
        slots = orig_vars.get('__slots__')
        if slots is not None:
            if isinstance(slots, str):
                slots = [slots]
            for slots_var in slots:
                orig_vars.pop(slots_var, None)
        orig_vars.pop('__dict__', None)
        orig_vars.pop('__weakref__', None)
        return metaclass(cls.__name__, cls.__bases__, orig_vars)
    return wrapper

# Import from moves module
from . import moves

# Add compatibility methods
if PY2:
    iteritems = lambda d: d.iteritems()
    iterkeys = lambda d: d.iterkeys()
    itervalues = lambda d: d.itervalues()
    int2byte = lambda i: chr(i) if isinstance(i, int) else bytes(i)
    unichr = unichr
    u = lambda s: unicode(s, "unicode_escape")
    integer_types = (int, long)
    def advance_iterator(it):
        return it.next()
else:
    iteritems = lambda d: d.items()
    iterkeys = lambda d: d.keys()
    itervalues = lambda d: d.values()
    int2byte = lambda i: bytes([i])
    unichr = chr
    u = lambda s: s
    integer_types = (int,)
    def advance_iterator(it):
        return next(it)
