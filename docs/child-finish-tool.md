# explicit completion for background subagents

status: proposed

date: 2026-10-20

## summary

background `worker` and `reviewer` children should remain live until they call a
new `finish` tool. a turn that ends without `finish` is not completion. zeta
nudges the child once, then records it as idle and waits without using model
tokens.

this behavior starts behind a global setting and profile opt-in. phoebe will use
it before it becomes the default. foreground children and other profiles keep
the current one-shot behavior during the rollout.

## motivation

current completion is inferred from provider turn shape, not agent intent. this
has caused real failures:

- an eval runner replied `WAIT` while waiting for its parent's approval. zeta
  treated the no-tool turn as completion, closed the child, and killed its
  background run after 39 of 96 calls.
- a reviewer finished seconds after it reported. the parent's follow-up then
  failed with `no live run/child ... canceled, finished, or never started`. the
  parent had to start a new reviewer and lost the original context.
- a provider `server_error` currently exits outside the normal finish path and
  kills child-owned tasks. pr #521 adds retry for that error class. this design
  separately defines the terminal result after retries are exhausted.

an agent should say when its contract is complete. silence at a turn boundary
should mean “available for more work,” not “destroy this run.”

## current behavior

- `consume_child` saves an assistant message as final when `TURN_END` reports
  zero tool calls (`src/zeta/agent/runner.py:160-165`). it then returns a
  completed result from that text (`src/zeta/agent/runner.py:235-251`).
- `consume_run` delivers queued `agent_send` prompts and notification turns,
  then finishes when neither exists (`src/zeta/agent/runner.py:254-273`,
  `src/zeta/agent/runner.py:344-384`).
- a follow-up loop exists only when a profile accepts follow-ups and the child
  is background (`src/zeta/agent/conversation_channel.py:24-27`). packaged
  `general` and `run` accept them; `explore` and `plan` do not
  (`src/zeta/agent/presets.py:43-100`). the local profiles also opt in
  (`~/.zeta/agents/worker.md:6`, `~/.zeta/agents/reviewer.md:6`).
- eligible children receive non-blocking `ask_parent`; it durably publishes a
  question and says that the answer will arrive as a follow-up
  (`src/zeta/tools/ask_parent/__init__.py:22-100`). `agent_send` appends the
  prompt durably and wakes the child (`src/zeta/agent/conversation_channel.py:82-141`,
  `src/zeta/tools/agent_send/__init__.py:15-97`).
- completion is persisted as a notification, the live child marker is removed,
  and the parent is woken (`src/zeta/agent/conversation_channel.py:143-185`).
  terminal receipts are bounded to 10,000 bytes
  (`src/zeta/agent/receipt.py:21-27`).
- if an inferred final turn still owns children or tasks, the finish gate asks
  the model to wait, cancel, or call `agent_handoff`
  (`src/zeta/agent/background.py:33-72`, `src/zeta/agent/finish_gate.py:195-229`).
  its defaults are 8 model turns, 600 seconds for the model bound, and 3,600
  seconds for a running task (`src/zeta/agent/background.py:91-122`). pr #512
  lets a running task outlive the short bound but retains the one-hour safety
  bound.
- child closure aborts the registry and waits for it to kill background tasks;
  the resulting killed task ids are included in the completion path
  (`src/zeta/agent/background.py:880-931`).
- live markers and lifecycle files drive `agent_status` and tui run controls
  (`src/zeta/tools/agent/__init__.py:740-772`,
  `src/zeta/tui/cards/agent.py:726-843`). restart recovery cancels unfinished
  descendants instead of recreating their loops
  (`src/zeta/agent/background.py:479-580`).
- a live child holds an asyncio run and its backend, store, registry, and tool
  state. it holds at least one session-directory fd in `ConversationStore` and
  one cwd fd in its cloned `ToolRegistry` (`src/zeta/core/store/_store.py:120-156`,
  `src/zeta/tools/registry.py:187-264`, `src/zeta/tools/registry.py:658-672`).
  pr #498 fixed those registry fds leaking after children ended; an idle child
  still holds them intentionally until it ends.

## scope

this design distinguishes a completed contract from a quiet turn, preserves
context for follow-ups, waits without model calls, keeps terminal data bounded,
and replaces the opted-in finish gate. it does not change foreground children,
resume processes after zeta exits, add a workflow framework, or change provider
retry policy; pr #521 owns retries.

## proposed design

### eligibility and setting

add this global-only setting:

```toml
[agents]
explicit_finish = false
```

add profile frontmatter `explicit_finish: true`. the tool is available only
when all of these are true:

1. `[agents].explicit_finish` is true in global settings;
2. the selected profile has `explicit_finish: true`;
3. the child is background and `accepts_follow_ups: true`.

project settings cannot enable it during the experiment. foreground children
never receive the tool. the first opted-in profiles are the user's `worker` and
`reviewer`; packaged presets remain unchanged until the phoebe trial succeeds.
the effective value is captured in the agent-catalog and child-marker snapshots
so resume and inspection do not reinterpret an old run under new settings.

### the `finish` tool

register one session tool named `finish` for eligible children:

```json
{
  "type": "object",
  "properties": {
    "status": {"enum": ["done", "blocked", "failed"]},
    "summary": {"type": "string", "minLength": 1, "maxLength": 8000},
    "head_sha": {"type": "string", "pattern": "^[0-9a-f]{7,64}$"},
    "verdict": {"enum": ["pass", "fail", "inconclusive"]}
  },
  "required": ["status", "summary"],
  "additionalProperties": false
}
```

`head_sha` is for implementation work. `verdict` is for reviews. both are
optional because neither applies to every contract. the tool requires no
approval and is not parallel-safe.

`finish` records terminal intent. its successful result tells the model that no
more work or prose is needed. zeta lets the current provider turn close so the
provider message remains valid, ignores later assistant prose as the report,
and then finalizes from the recorded tool arguments. a second `finish` in the
same turn is rejected. other tool calls after a successful `finish` are rejected
as “child is finishing.”

report status and runtime state are separate:

| `finish.status` | lifecycle state | receipt error | meaning |
| --- | --- | --- | --- |
| `done` | `completed` | false | contract completed |
| `blocked` | `completed` | false | contract cannot proceed; parent action is needed |
| `failed` | `failed` | true | contract ended unsuccessfully |

add `report_status`, `summary`, and supplied optional fields to structured
completion content. notification text starts with `done:`, `blocked:`, or
`failed:` and then the summary. use the existing receipt builder and notification
byte ceilings; trim only the summary, never the status or structured scalar
fields. do not persist both free-form final prose and the summary. this keeps one
bounded canonical receipt for the original agent call, lifecycle final result,
and parent completion notice.

### no-tool turn and idle state

on the first normal turn that ends without a successful `finish`, send one
harness nudge in a new model turn:

> you have not called `finish`. if the contract is complete, call it now. if
> you need a parent answer, call `ask_parent`. otherwise end this turn and zeta
> will keep you idle for follow-ups without using tokens.

this nudge occurs at most once for the child's lifetime. if the nudge turn also
ends without `finish`, or any later woken turn ends without it, set lifecycle
state to `idle`, set `current_step` to `idle, waiting for follow-up`, and wait.
`idle` is non-terminal: keep the parent marker, pending-prompt queue, loop,
registry, and store open.

these events wake an idle child and produce one ordinary model turn:

- a durable `agent_send` follow-up, including an answer to `ask_parent`;
- exit of one of the child's own background tasks;
- completion, failure, or cancellation of one of its direct child agents.

coalesce simultaneous events into that turn. task and child events use the
existing notification input, so the model sees what changed before it acts.

these events do not wake it:

- `agent_status`, `agent_output`, tui viewing, or other reads;
- unrelated parent or sibling notifications;
- progress output from a still-running task;
- elapsed time before a terminal timeout;
- repeated status polling.

### idle timeout and capacity

an idle child auto-finishes after 30 minutes with no wake event and no running
owned task. every listed wake event resets the clock. time spent while a task is
running does not count. when the last task exits, its wake is delivered; if the
following turn returns to idle, a new 30-minute clock starts.

the synthesized completion has lifecycle state `completed`, report status
`blocked`, reason `idle_timeout`, and summary:

> auto-finished after 30 minutes idle without an explicit finish report.

append a bounded excerpt of the child's last assistant text when one exists.
include normal turns, elapsed time, and tool-call statistics. do not claim that
the contract succeeded.

cap idle explicit-finish children at 16 per root agent tree. active model turns
and children with running tasks do not count toward this idle cap. if a child
would become the seventeenth idle child, auto-finish that child as `blocked`
with reason `idle_capacity`; do not evict an older child silently. the parent
can cancel any live child with `agent_cancel`.

there is no separate “finish now” parent tool in the first version. the parent
can send “finish now” with `agent_send`, which gives the child a chance to write
an accurate report, or cancel it when no report is needed.

### running work and replacement of the finish gate

`finish` refuses while the child owns a running process task or live child
agent. the error lists bounded ids and tells the child to:

- wait for task or child notifications;
- cancel work with `task_kill` or `agent_cancel`; or
- hand off child agents with `agent_handoff`.

register the existing `agent_handoff` beside `finish` for eligible children.
it transfers live descendant agents to the parent. process tasks remain tied to
the child's registry and cannot be handed off; they must exit or be killed.
only after ownership is clear can `finish` succeed.

for opted-in children, delete the 8-turn and 600-second finish-gate model loop.
remove its inferred-final-message entry condition. retain one safety rule from
pr #512: a process task that is still running 3,600 seconds after the child
first becomes idle is killed, and its exit wakes the child to report. this is a
resource safety deadline, not a finish gate and not a model-turn deadline. if
the child does not call `finish` after that wake, the normal 30-minute idle
clock applies.

this leaves one normal path: explicit `finish`; and three harness-owned terminal
paths: idle timeout/capacity, cancellation, and unrecoverable error.

### errors, cancellation, exit, and restart

- **provider error:** apply configured retries, including pr #521 when merged.
  if retries are exhausted, emit a failed receipt containing the sanitized
  provider error and killed-task metadata. do not nudge or enter idle because
  no healthy model turn exists. close the child and kill process tasks.
- **cancellation:** `agent_cancel` remains immediate. mark `canceled`, close
  pending questions, cancel descendants, kill process tasks, and publish the
  current bounded canceled receipt. a prior `finish` that has not yet been
  durably published does not outrank cancellation.
- **parent exit:** graceful parent shutdown cancels live children as it does
  now. a crash is reconciled as canceled on recovery. child conversations stay
  available for inspection, but `agent_send` cannot revive them.
- **restart:** do not rehydrate idle loops in the first version. durable idle
  state makes the reason visible during recovery, but recovery converts it to a
  canceled terminal result. this avoids pretending process tasks, in-memory
  backends, approvals, and tool leases survived.
- **foreground children:** unchanged. they still complete from their final
  no-tool assistant turn and do not receive `finish`.

### parent visibility

`agent_status` must accept and return lifecycle state `idle`, plus
`report_status` and `idle_since` when present. an idle child remains in live
results and still accepts `agent_send`. its tui card shows `idle · waiting for
follow-up` and idle duration. it is not styled as completed.

completion uses the existing notification/wake path. the parent sees the
bounded canonical report and structured status exactly once. the original
background agent tool result remains `running`; the later notification remains
the terminal receipt, as today.

### resource and token cost

an idle child makes no provider request, so it uses no model tokens. it retains:

- one asyncio waiter and child/watcher state;
- its backend and conversation objects in memory;
- at least two open directory fds: the child session store and registry cwd;
- tool registry state and any approval/store leases.

exact memory depends on provider and transcript size, so the design does not
promise a byte figure. the 16-child idle cap bounds the known baseline to at
least 32 retained directory fds per root tree, plus provider and tool-specific
resources. metrics must record the observed fd count and rss delta at idle entry
and exit during the phoebe trial. pr #498's close path remains mandatory on all
terminal paths.

## alternatives considered

### infer completion from final prose

rejected. `WAIT`, “done,” or “ready” are ambiguous and caused the reported
failures.

### keep the current finish gate and add `finish`

rejected. two completion authorities can disagree and preserve the old
8-turn, 600-second machinery.

### finish immediately on any `finish` call, even with running work

rejected. closing the registry would kill work that the report implies is safe.
refusal makes ownership explicit and testable.

### make `summary` the only argument

rejected. parents need to distinguish success, a blocker, and a failed contract
without parsing prose. `head_sha` and `verdict` cover the two first profiles
without creating arbitrary metadata.

### block inside `ask_parent`

rejected. the durable non-blocking question path works; idle adds the missing
wait state.

### resume idle children after restart

rejected for the first release. rebuilding a valid model loop, tool registry,
approval state, and task ownership is a larger durability protocol. persisted
conversation remains available even when the live lease ends.

### no idle timeout or cap

rejected. idle children use no tokens but retain fds, memory, leases, and a live
address for follow-ups. pr #498 showed that descriptor lifetime is operationally
important.

## rollout and measurement

1. add tests first; confirm each semantic test fails on the pre-implementation
   revision for the expected reason.
2. implement `[agents].explicit_finish = false`, profile parsing/snapshotting,
   and the tool without changing any default behavior.
3. add `explicit_finish: true` only to the user's `worker` and `reviewer`
   profiles. enable the global setting only in phoebe's environment.
4. delete the current finish gate only from the opted-in path. keep it unchanged
   for legacy profiles during the experiment.
5. run phoebe with the setting for at least 100 eligible children and seven
   days, including implementation workers and reviewers.
6. compare with the preceding 100 eligible children. record:
   - `agent_send` failures because the target already finished;
   - no-tool turns followed by later parent messages;
   - auto-finishes by `idle_timeout` and `idle_capacity`;
   - tasks killed on normal completion, safety deadline, cancellation, and
     provider error;
   - finish refusals due to owned work;
   - idle duration, concurrent idle peak, fd count, and rss delta;
   - extra nudge turns and tokens.
7. success requires zero known premature finishes, zero running tasks killed by
   successful `finish`, and no increase in unreconciled child markers. review
   timeout/cap and token data before changing defaults.
8. after review, opt packaged background `run` into the protocol. remove the
   setting and legacy finish gate only in a later migration. do not silently
   change foreground `general`, `explore`, or `plan` behavior.

## test plan

all node ids are proposed and should be created test-first.

- `tests/test_agent.py::test_explicit_finish_done_publishes_canonical_report`
  verifies schema, lifecycle mapping, structured fields, one notification, and
  receipt bounds.
- `tests/test_agent.py::test_explicit_finish_blocked_and_failed_map_to_receipts`
  verifies report status separately from runtime/error state.
- `tests/test_agent.py::test_explicit_finish_nudges_once_then_idles`
  verifies a no-tool turn, one nudge, durable idle state, and no provider call
  while waiting.
- `tests/test_agent.py::test_idle_child_wakes_for_follow_up_and_parent_answer`
  verifies `agent_send` and an `ask_parent` answer.
- `tests/test_agent.py::test_idle_child_wakes_for_task_and_child_completion_only`
  verifies task/child notifications and rejects status reads, progress, and
  unrelated notifications as wakes.
- `tests/test_agent.py::test_explicit_finish_refuses_owned_work_until_clear`
  verifies task/child refusal, cancellation, and descendant handoff.
- `tests/test_background.py::test_explicit_finish_task_safety_deadline_kills_and_wakes`
  verifies the single 3,600-second task deadline and killed-task metadata.
- `tests/test_agent.py::test_idle_timeout_auto_finishes_blocked_with_last_text`
  verifies the 30-minute reset rules and bounded synthesized report.
- `tests/test_agent.py::test_idle_capacity_auto_finishes_newest_child`
  verifies the 16-child cap without evicting an older child.
- `tests/test_agent.py::test_explicit_finish_provider_error_fails_without_idle`
  verifies exhausted retries, failed receipt, and task cleanup.
- `tests/test_agent.py::test_explicit_finish_cancel_wins_before_publication`
  verifies cancellation ordering and cleanup.
- `tests/test_agent.py::test_idle_child_parent_exit_recovers_as_canceled`
  verifies no rehydration and durable inspection data.
- `tests/test_agents.py::test_explicit_finish_requires_global_and_profile_opt_in`
  verifies the eligibility matrix, catalog snapshot, and foreground exclusion.
- `tests/test_settings.py::test_project_cannot_enable_explicit_finish`
  verifies global-only experimental control and default false.
- `tests/test_tui.py::test_idle_agent_card_and_status_remain_live`
  verifies `agent_status`, tui state, idle duration, and follow-up eligibility.
- existing finish-gate tests remain unchanged for legacy profiles; new deletion
  tests prove opted-in runs never enter `run_finish_gate`.

## open questions

1. is 30 minutes the correct initial idle timeout for phoebe, or should the
   experiment start at 60 minutes to favor follow-up continuity?
2. is 16 idle children per root tree an acceptable cap for phoebe's normal
   orchestration fan-out?
3. should `blocked` be a successful transport receipt as proposed, or should it
   set `isError: true` even though the agent reported normally?
4. after the phoebe trial, should restart rehydration be designed before the
   protocol becomes the default, or can cancellation-on-restart remain the
   permanent rule?
