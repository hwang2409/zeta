# computer use

computer use is disabled by default. enable it in your home `settings.toml`:

```toml
[computer]
enabled = true
```

by default, computer use denies apps that are not in the macos bundle allow-list. add an app under `[computer.macos.bundle_ids]` when you want deny mode with a limited set of apps.

for whole-computer mode, explicitly set:

```toml
[computer]
enabled = true
default_app_access = "allow"
```

whole-computer mode still refuses actions while the screen is locked or secure input is active. the kill switch remains enabled.
