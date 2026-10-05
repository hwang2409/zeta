# Maintainer constraints

`event_export/schema.py` is an immutable wire-schema artifact. Do not modify it
for any reason; repairs belong in the exporter or its callers. The grader checks
that this file remains byte-for-byte identical.
