# Attention panel

`zeta panel` is a local, read-only view of live top-level Zeta sessions. It groups sessions by project and derives lanes from the session files that Zeta already writes for child agents and background tasks. It does not use a server or daemon. Use `zeta panel --list` for a plain-text summary.

A top-level orchestrator can call `request_attention` when it needs a decision that only the user can make. The request stores a small atomic JSON record in the session's `attention/` directory. The context must be self-contained. The orchestrator can then continue other work instead of repeating that it is waiting.

In the full-screen panel, use arrows or `j`/`k` to move, Enter to open an attention item, `r` to refresh, and `q` to quit. The panel also refreshes every two seconds. Session reads are bounded and run outside the terminal event loop.

Opening an item creates a new discussion session from the original active branch at the exact recorded entry. The original session remains unchanged and continues to run. Discussion forks have read-only tools plus `resolve_attention`; they cannot use edit, write, shell, agent, or background-task tools. Resolving sends the user's decision only to the original session and records the resolution. A recently resolved item stays dimmed in the panel for ten minutes, and reopening it resumes the same discussion fork.

This v0 is local to one Zeta home. It does not provide cross-machine delivery, external notifications, editable forks, or multiple decisions per item.
