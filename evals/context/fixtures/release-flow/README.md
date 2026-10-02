# Release flow

Implement the incomplete release flow after reading `design/` in numeric order. The API is `run_release(raw, sink, *, dry_run=False)`. It must parse a mapping, validate unique step names and known dependencies, produce a stable topological plan (lexical tie-break), and execute it. A dry run returns the plan without calling the sink. A real run is atomic from the caller's perspective: on a sink failure, call `sink.rollback(completed_names)` and re-raise. Never run a dependent before its prerequisites.
