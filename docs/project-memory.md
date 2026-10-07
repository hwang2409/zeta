# Project memory

Zeta keeps a small set of bounded Markdown files for each project:
`brief.md`, `state.md`, `backlog.md`, `changelog.md`, and `decisions.md`.
The files are loaded into the project context. The `zeta project memory`
command reads them and can replace one file with `--set` or `--from-file`.
See `zeta project memory --help` for the command syntax.

## Automatic updates

Automatic project-memory reconciliation is enabled by default for sessions that
have an associated project. One background worker reads only durable transcript
rows and creates updates:

- before context eviction;
- after 50,000 estimated new transcript tokens; or
- after 10 minutes without activity.

The token trigger uses transcript growth, not provider usage. Requests use the
configured memory model. Before provider assembly, Zeta removes secrets and
agent-directed instructions. Each automatic update records the source session,
transcript range, model, and usage as provenance. Remote export and import retain this provenance and automatic status; imports are undoable versions and never accept content automatically.

Configure the global `~/.zeta/settings.toml` file:

```toml
[memory]
auto = true
model = "gpt-5.6-luna"
token_threshold = 50000
idle_minutes = 10
```

Use `--no-auto-memory` to disable automatic reconciliation for one session.
`--auto-memory` enables it explicitly. These flags are available on the main
`zeta` command.

## Review and acceptance

In the TUI:

- `/memory log` shows retained update records and their provenance.
- `/memory undo` restores the latest update that has not already been undone.
- `/memory accept <file>` explicitly promotes an automatic file to trusted
  memory.

Manual edits do not remove automatic provenance. Acceptance is a user-only
operation and is recorded as `accepted_by: user`.

The equivalent CLI forms are:

```sh
zeta project memory <project> accept <file>
zeta project memory accept <file>  # from an associated project directory
```

The CLI action requires a terminal and asks for confirmation of the file name.
The command accepts exactly one standard memory file.

## Versions and synchronization

Zeta retains 128 recent version records, plus any older version needed by a
retained undo. The version store is private to `ProjectRegistry`. Code that
synchronizes or exports project memory must use the logical
`ProjectRegistry.export_memory()` and CAS-based `ProjectRegistry.import_memory()`
interfaces. Do not copy the private project-memory paths.

## Safety

The acceptance prompt prevents casual acceptance by an automatic model or by a
non-interactive CLI caller. Zeta's shell tools run as the current user, so this
is not protection against a hostile shell that edits project-memory files
directly. Treat shell access and project files as trusted host resources.
