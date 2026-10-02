# Reporting API migration

Replace the public `render_summary(records)` API everywhere with
`build_report(records, *, heading)`. The old symbol must be removed completely:
do not retain a compatibility alias or any stale caller.

`heading` is required and is emitted verbatim as the first line. Follow it with
`name: value` lines sorted by name and exactly one final newline. Update package
exports and every in-repository caller, including dynamically loaded plugins.
