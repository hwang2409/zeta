# Decisions in the TUI

A top-level orchestrator calls `request_attention` when it needs a decision
that only the user can make. The request stores a small atomic JSON record in
the session's `attention/` directory and the orchestrator keeps working. The
context in `why` must be self-contained; the orchestrator does not repeat that
it is waiting.

You answer these decisions inside the TUI you already use, not a separate tool.

## The status-bar count and the popup

When one or more live sessions have an open decision, the status bar shows a
compact count, for example `● 2 decisions`. The count disappears when nothing
is pending. Press `Ctrl+T` or run `/decisions` to open the decisions popup.

The popup is a finder-style overlay. It lists one row per open decision with
the asking session's project and id, the title, a short `why`, and any offered
options. Move with `↑/↓` (or `j/k`). For the selected decision:

- Press a number `1`–`9` to send one of the offered options.
- Press `a` to type one short line, then Enter to send it.
- Press `Enter` to discuss the decision (see below).
- Press `Esc` to close the popup. This never switches sessions.

A quick answer (an option or a typed line) is delivered to the asking
orchestrator's inbox exactly like a discussion-fork decision, and the item
closes. Quick answers work for decisions from any live session.

## Discussing a decision

Discussing opens (or reopens) a read-only discussion fork for the decision and
switches the TUI to it. The fork is created from the original session's active
branch at the exact recorded entry; the original session is unchanged and keeps
running. The fork has read-only tools plus `resolve_attention`; it cannot edit,
write, run shell, spawn agents, or start background tasks. The switch restores
the terminal and resumes the fork in place, the same way `/new` starts a fresh
session.

While in a fork, the TUI shows a `discussion · fork of <orchestrator>` header
and the subagent view (open it from the composer) lists two special entries:

- `discussion: <title>` — this fork.
- `(main)` — the asking orchestrator.

Select `(main)` and press Enter to return to the orchestrator and close the
fork. If you already sent a decision, the fork closes immediately. If you have
not, the TUI asks once (`leave without a decision? the item stays open`); press
Enter again to leave or `Esc` to stay. An item left open can be discussed again
later with a fresh fork from the same point.

Discussion forks are excluded from project memory and search, exactly as
before.

## Scope

This is local to one Zeta home. It does not provide cross-machine delivery,
external notifications, editable forks, or multiple decisions per item. If a
decision's target session is not running when a quick answer is sent, the
targeted message waits in that session's inbox until it resumes; it is never
delivered to another session.
