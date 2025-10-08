# -*- coding: utf-8 -*-
"""six: Python 2 and 3 compatibility library."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import sys
import types

# Python 2 and 3 compatibility
PY2 = sys.version_info[0] == 2
PY3 = not PY2

if PY2:
    text_type = unicode  # type: ignore
    binary_type = str
    string_types = (str, unicode)  # type: ignore
    integer_types = (int, long)  # type: ignore
    class_types = (type, types.ClassType)
    text_to_binary = lambda s, enc: s
else:
    text_type = str
    binary_type = bytes
    string_types = (str,)
    integer_types = (int,)
    class_types = (type,)
    text_to_binary = lambda s, enc: s.encode(enc)

# 创建moves模块
class MovedModule(object):
    """Lazy loading of moved objects."""
    def __init__(self, name, old, new=None):
        self.name = name
        self.old = old
        self.new = new or old

    def __getattr__(self, attr):
        try:
            if PY3:
                return getattr(__import__(self.new), attr)
            else:
                return getattr(__import__(self.old), attr)
        except (ImportError, AttributeError):
            raise AttributeError("module %r has no attribute %r" % (self.name, attr))

class Moves(object):
    """A container for modules that have been moved between Python 2 and 3."""
    def __init__(self):
        self.__path__ = []

    def __dir__(self):
        return ['configparser', 'cPickle', 'builtins', 'http_cookiejar', 'http_client', 'html_parser', 'input', 'map', 'filter', 'range', 'zip', 'tkinter', 'reload_module', 'queue', 'reduce', 'shlex_quote']

    def __getattr__(self, attr):
        module_moves = {
            'configparser': MovedModule('configparser', 'ConfigParser', 'configparser'),
            'cPickle': MovedModule('cPickle', 'cPickle', 'pickle'),
            'builtins': MovedModule('builtins', '__builtin__', 'builtins'),
            'http_cookiejar': MovedModule('http_cookiejar', 'cookielib', 'http.cookiejar'),
            'http_client': MovedModule('http_client', 'httplib', 'http.client'),
            'html_parser': MovedModule('html_parser', 'HTMLParser', 'html.parser'),
            'queue': MovedModule('queue', 'Queue', 'queue'),
            'reload_module': MovedModule('reload_module', 'reload', 'importlib.reload') if PY3 else MovedModule('reload_module', 'reload', 'reload'),
            'tkinter': MovedModule('tkinter', 'Tkinter', 'tkinter'),
        }
        if attr in module_moves:
            return module_moves[attr]
        else:
            raise AttributeError("module 'six.moves' has no attribute '%s'" % attr)

moves = Moves()

# 为了确保six可以被当作包导入，设置__path__
if not hasattr(sys.modules[__name__], '__path__'):
    sys.modules[__name__].__path__ = []

# 确保PY2和PY3属性可以被外部访问
sys.modules[__name__].PY2 = PY2
sys.modules[__name__].PY3 = PY3
sys.modules[__name__].moves = moves

# 添加兼容性方法
if PY2:
    sys.modules[__name__].iteritems = lambda d: d.iteritems()
    sys.modules[__name__].iterkeys = lambda d: d.iterkeys()
    sys.modules[__name__].itervalues = lambda d: d.itervalues()
else:
    sys.modules[__name__].iteritems = lambda d: d.items()
    sys.modules[__name__].iterkeys = lambda d: d.keys()
    sys.modules[__name__].itervalues = lambda d: d.values()
