## Normative compatibility rule

A rule matches when `subject.startswith(rule.prefix)`. A matching deny overrides
every matching allow regardless of declaration order. If no deny matches, any
matching allow permits. If nothing matches, deny with reason `default deny`.
Reasons are the selected rule's text; among rules of the same effect, the first
declaration wins. Thus `decide([], "anything")` returns
`(False, "default deny")`.
