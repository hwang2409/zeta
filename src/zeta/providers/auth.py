"""Compatibility alias for shared OAuth primitives."""

import sys

from .. import oauth as _oauth

sys.modules[__name__] = _oauth
