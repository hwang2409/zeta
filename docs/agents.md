# Agents and orchestration

Custom agent profiles are Markdown files in `~/.zeta/agents/` or
`<project>/.zeta/agents/`. Their YAML frontmatter defines the profile. For
example, a worker that can converse with its parent uses:

```yaml
---
name: worker
description: Implements one bounded contract
accepts_follow_ups: true
---
```

`accepts_follow_ups` defaults to `false` for custom profiles. Set it to
`true` when a worker or reviewer should accept follow-ups. Packaged `general`
and `run` agents accept follow-ups; packaged `explore` and `plan` agents do
not.

The TUI `/runs` command lists every live background child that accepts
follow-ups, including its agent type. When viewing an eligible child, plain
text entered in the composer is sent to that child; slash commands still go to
the main session.

The parent sends a message with `agent_send`. Delivery occurs at the child's
next turn boundary, never during a tool call. Eligible background children also
receive `ask_parent`. It records one durable `child_question` notification for
the direct parent and returns immediately. The child continues its work. The
parent answers with an ordinary `agent_send` message and can reference the
notification's `question_id` in that message. If the direct parent has ended,
the question is routed to the next live ancestor. A pending question is
withdrawn when its child finishes or is canceled.
