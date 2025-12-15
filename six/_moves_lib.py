# -*- coding: utf-8 -*-
"""six._moves_lib: Implementation of moved functions."""
from __future__ import absolute_import

import sys
from functools import wraps

# Handle urllib modules
if sys.version_info[0] == 3:
    import urllib.request as urllib_request
    import urllib.error as urllib_error
    import urllib.parse as urllib_parse
    import http.client as http_client
else:
    import urllib2 as urllib_request
    import urllib2 as urllib_error
    import urllib as urllib_parse
    import httplib as http_client

# Create urllib module structure
sys.modules['six.moves.urllib'] = type('module', (), {})
sys.modules['six.moves.urllib.request'] = urllib_request
sys.modules['six.moves.urllib.error'] = urllib_error
sys.modules['six.moves.urllib.parse'] = urllib_parse

# Add range and zip functions to six.moves
sys.modules['six.moves'].__dict__['range'] = range
sys.modules['six.moves'].__dict__['zip'] = zip

# Add html_entities to six.moves
import html
import html.entities
html_entities = html.entities
sys.modules['six.moves'].__dict__['html_entities'] = html_entities

PY2 = sys.version_info[0] == 2
PY3 = sys.version_info[0] == 3

# Handle input function
if PY3:
    input = input
else:
    input = raw_input  # type: ignore

# Handle range function
if PY3:
    def srange(*args):
        return list(range(*args))
    xrange = range
else:
    srange = range
    xrange = xrange  # type: ignore

# Handle filter, map, zip functions
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

# Handle reduce function
if PY3:
    from functools import reduce
else:
    reduce = reduce

# Handle intern function
if PY3:
    intern = sys.intern
else:
    intern = intern  # type: ignore

# Handle reload function
if PY3:
    from importlib import reload as reload_module
else:
    reload_module = reload  # type: ignore

# Handle unichr function
if PY3:
    unichr = chr
else:
    unichr = unichr  # type: ignore

# Handle bytearray function
if PY3:
    bytearray2 = bytearray
else:
    def bytearray2(*args):
        return bytearray(*args)

# Handle ascii function
if PY3:
    ascii = ascii
else:
    def ascii(obj):
        return repr(obj).encode('ascii', 'backslashreplace').decode('ascii')