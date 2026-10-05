# Reporting API migration

Replace the public `render_summary(records)` API everywhere with
`build_report(records, *, heading)`. The old symbol must be removed completely:
do not retain a compatibility alias or any stale caller.

`heading` is required and is emitted verbatim as the first line. Follow it with
`name: value` lines sorted by name and exactly one final newline. The package
export list must be exactly `__all__ = ["build_report"]`.

Update every in-repository caller, including dynamically loaded plugins. The CLI
must call `build_report` with the exact heading `Signal report`. Each plugin
module `pN` must call it with the exact heading `Plugin N`; for example,
`signal_report.plugins.p3.run(records)` starts its result with `Plugin 3\n`.
