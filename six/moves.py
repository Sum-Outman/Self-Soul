# -*- coding: utf-8 -*-
"""six.moves: Python 2 and 3 module compatibility."""
from __future__ import absolute_import

import sys

PY2 = sys.version_info[0] == 2
PY3 = sys.version_info[0] == 3

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

# Create common moved modules
configparser = MovedModule('configparser', 'ConfigParser', 'configparser')
cPickle = MovedModule('cPickle', 'cPickle', 'pickle')
builtins = MovedModule('builtins', '__builtin__', 'builtins')
http_cookiejar = MovedModule('http_cookiejar', 'cookielib', 'http.cookiejar')
http_client = MovedModule('http_client', 'httplib', 'http.client')
html_parser = MovedModule('html_parser', 'HTMLParser', 'html.parser')
queue = MovedModule('queue', 'Queue', 'queue')
tkinter = MovedModule('tkinter', 'Tkinter', 'tkinter')
# Add thread-related modules
_thread = MovedModule('_thread', 'thread', '_thread')
_threading_local = MovedModule('_threading_local', 'threading_local', '_threading_local')
shlex_quote = MovedModule('shlex_quote', 'pipes', 'shlex').quote

# Import lib module
from . import _moves_lib

# Import functions from lib module
input = _moves_lib.input
srange = _moves_lib.srange
xrange = _moves_lib.xrange
filter = _moves_lib.filter
map = _moves_lib.map
zip = _moves_lib.zip
reduce = _moves_lib.reduce
intern = _moves_lib.intern
reload_module = _moves_lib.reload_module
unichr = _moves_lib.unichr
bytearray2 = _moves_lib.bytearray2
ascii = _moves_lib.ascii

# Add itertools functions
import itertools
if sys.version_info[0] == 3:
    zip_longest = itertools.zip_longest
else:
    zip_longest = itertools.izip_longest