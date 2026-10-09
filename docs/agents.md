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

`accepts_follow_ups` defaults to `false` for custom profiles. Keep it false (or
omit it) for reviewers so each review is independent and one-shot. Packaged
`general` and `run` agents accept follow-ups; packaged `explore` and `plan`
agents do not.

The parent sends a message with `agent_send`. Delivery occurs at the child's
next turn boundary, never during a tool call. Eligible children also receive
`ask_parent`. It records one durable `child_question` notification for the
direct parent and waits for at most 300 seconds by default. The parent can
answer with `agent_send`, optionally passing the notification's `question_id`.
If no answer arrives, the child continues with a stated assumption; a later
answer remains a normal follow-up. Each child can have only one waiting
question at a time.
