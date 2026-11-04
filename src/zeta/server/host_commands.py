from __future__ import annotations

import html
import posixpath
import re
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType

from .protocol import ProtocolError

_NAME=re.compile(r'[A-Za-z][A-Za-z0-9._-]{0,63}$'); _ENV=re.compile(r'[A-Za-z_][A-Za-z0-9_]{0,127}$')
@dataclass(frozen=True, slots=True)
class HostCommand: name:str; executable:str; summary:str
@dataclass(frozen=True, slots=True)
class SessionHostCommands:
 commands:tuple[HostCommand,...]; environment:Mapping[str,str]; instruction:str
 @classmethod
 def parse(cls,value:object):
  if value is None:value=[]
  if not isinstance(value,list) or len(value)>4: raise ProtocolError(-32602,'host_commands must be an array of at most 4 items')
  out=[]; env={}; names=set(); total=0
  for i,item in enumerate(value):
   p=f'host_commands[{i}]'
   if not isinstance(item,dict) or set(item)!={'name','executable','summary','environment'}: raise ProtocolError(-32602,f'{p} is invalid')
   n,e,s,m=(item[k] for k in ('name','executable','summary','environment'))
   if not isinstance(n,str) or not _NAME.fullmatch(n) or n in names: raise ProtocolError(-32602,f'{p}.name is invalid')
   if not isinstance(e,str) or not e.startswith('/') or e!=posixpath.normpath(e) or '\x00' in e or '\n' in e or len(e.encode())>4096: raise ProtocolError(-32602,f'{p}.executable is invalid')
   if not isinstance(s,str) or not s or '\n' in s or len(s.encode())>256: raise ProtocolError(-32602,f'{p}.summary is invalid')
   if not isinstance(m,dict): raise ProtocolError(-32602,f'{p}.environment is invalid')
   names.add(n); out.append(HostCommand(n,e,s))
   for k,v in m.items():
    if not isinstance(k,str) or not _ENV.fullmatch(k) or k in env or k in {'BASH_ENV','ENV','SHELLOPTS','BASHOPTS','CDPATH','GLOBIGNORE','ZETA_TOOL_SUBPROCESS'} or k.startswith(('LD_','DYLD_')): raise ProtocolError(-32602,f'{p}.environment key is invalid')
    if not isinstance(v,str) or '\x00' in v or len(v.encode())>4096: raise ProtocolError(-32602,f'{p}.environment value is invalid')
    env[k]=v; total+=len(k.encode())+len(v.encode())
  if len(env)>16 or total>16384: raise ProtocolError(-32602,'host_commands.environment exceeds its limit')
  ins=''
  if out: ins='<zeta-session-host-commands>\n'+'\n'.join(f'- {html.escape(c.name)}: execute `{html.escape(c.executable)}`; {html.escape(c.summary)}' for c in out)+'\n</zeta-session-host-commands>'
  return cls(tuple(out),MappingProxyType(env),ins)
