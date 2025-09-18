# -*- coding: utf-8 -*-
"""six._moves_lib: Implementation of moved functions."""
from __future__ import absolute_import

import sys
from functools import wraps

PY2 = sys.version_info[0] == 2
PY3 = sys.version_info[0] == 3

# 处理input函数
if PY3:
    input = input
else:
    input = raw_input

# 处理range函数
if PY3:
    def srange(*args):
        return list(range(*args))
    xrange = range
else:
    srange = range
    xrange = xrange

# 处理filter, map, zip函数
if PY3:
    filter = filter
    map = map
    zip = zip
else:
    def filter(func, it):
        return list(__builtins__.filter(func, it))
    
    def map(func, *it):
        return list(__builtins__.map(func, *it))
    
    def zip(*it):
        return list(__builtins__.zip(*it))

# 处理reduce函数
if PY3:
    from functools import reduce
else:
    reduce = reduce

# 处理intern函数
if PY3:
    intern = sys.intern
else:
    intern = intern

# 处理reload函数
if PY3:
    from importlib import reload as reload_module
else:
    reload_module = reload

# 处理unichr函数
if PY3:
    unichr = chr
else:
    unichr = unichr

# 处理bytearray函数
if PY3:
    bytearray2 = bytearray
else:
    def bytearray2(*args):
        return bytearray(*args)

# 处理ascii函数
if PY3:
    ascii = ascii
else:
    def ascii(obj):
        return repr(obj).encode('ascii', 'backslashreplace').decode('ascii')