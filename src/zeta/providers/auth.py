"""Compatibility imports for shared OAuth primitives."""

from .. import oauth as _oauth
from ..oauth import *

_redact_multipart = _oauth._redact_multipart
