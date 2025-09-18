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

# 元类装饰器
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

# 从moves模块导入
from . import moves

# 添加兼容性方法
if PY2:
    iteritems = lambda d: d.iteritems()
    iterkeys = lambda d: d.iterkeys()
    itervalues = lambda d: d.itervalues()
else:
    iteritems = lambda d: d.items()
    iterkeys = lambda d: d.keys()
    itervalues = lambda d: d.values()
