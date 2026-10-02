# Signal report API migration

Replace the public `render_summary(records)` API everywhere with `build_report(records, *, heading)`. There must be no compatibility alias and no stale use of the old name. `heading` is required, emitted verbatim as the first line, followed by sorted `name: value` lines and one final newline. Update all in-repo callers and exports. Inspect `evidence/` first; plugins are loaded dynamically.
