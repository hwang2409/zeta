//! Client-side GUI preferences.
//!
//! Appearance (theme, font family, font size) is an APP-GLOBAL user
//! preference — never per-session and never routed through the server. The
//! prefs file lives under `$ZETA_HOME/gui-prefs.json` (falling back to
//! `~/.zeta/gui-prefs.json` when `ZETA_HOME` is unset). Loading is
//! best-effort: a missing, unreadable, or corrupt file resolves to
//! `Appearance::default()` so a fresh install always paints opencode at
//! JetBrains Mono / 13px on the first frame.
//!
//! Writes are atomic (tmp + rename in the parent directory) so a crash mid
//! write cannot leave the prefs file half-flushed and unparseable next
//! launch.

use std::{env, fs, io, path::PathBuf};

use gpui::SharedString;
use serde::{Deserialize, Serialize};

use crate::theme::{
    self, clamp_font_size, Appearance, ThemeId, DEFAULT_FONT_FAMILY, DEFAULT_FONT_SIZE,
    FONT_FAMILIES,
};

const FILE_NAME: &str = "gui-prefs.json";

#[derive(Debug, Serialize, Deserialize, Default)]
struct StoredPrefs {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    theme: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    font_family: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    font_size: Option<f32>,
}

fn prefs_dir() -> PathBuf {
    if let Some(home) = env::var_os("ZETA_HOME") {
        return PathBuf::from(home);
    }
    PathBuf::from(env::var_os("HOME").unwrap_or_default()).join(".zeta")
}

/// Absolute path of the prefs file — exposed for tests and error messages.
pub fn prefs_path() -> PathBuf {
    prefs_dir().join(FILE_NAME)
}

/// Load the persisted appearance, falling back to defaults on any error
/// so the app boots regardless of what's on disk.
pub fn load() -> Appearance {
    load_from(&prefs_path())
}

fn load_from(path: &std::path::Path) -> Appearance {
    let raw = match fs::read_to_string(path) {
        Ok(text) => text,
        Err(_) => return Appearance::default(),
    };
    let stored: StoredPrefs = match serde_json::from_str(&raw) {
        Ok(v) => v,
        Err(_) => return Appearance::default(),
    };
    coerce(stored)
}

fn coerce(stored: StoredPrefs) -> Appearance {
    let theme = stored
        .theme
        .as_deref()
        .and_then(ThemeId::from_slug)
        .unwrap_or_default();
    let font_family = stored
        .font_family
        .filter(|name| FONT_FAMILIES.contains(&name.as_str()))
        .map(SharedString::from)
        .unwrap_or_else(|| SharedString::new_static(DEFAULT_FONT_FAMILY));
    let font_size = stored
        .font_size
        .map(clamp_font_size)
        .unwrap_or(DEFAULT_FONT_SIZE);
    Appearance {
        theme,
        font_family,
        font_size,
    }
}

/// Persist the current appearance atomically. Ignores I/O errors so a
/// read-only `ZETA_HOME` (rare, but possible in test sandboxes) does not
/// break the settings flow — a future launch just falls back to defaults.
pub fn save(app: &Appearance) {
    if let Err(err) = save_to(&prefs_path(), app) {
        eprintln!("zeta: could not save gui prefs: {err}");
    }
}

fn save_to(path: &std::path::Path, app: &Appearance) -> io::Result<()> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    let stored = StoredPrefs {
        theme: Some(app.theme.slug().to_string()),
        font_family: Some(app.font_family.to_string()),
        font_size: Some(f32::from(app.font_size)),
    };
    let json = serde_json::to_string_pretty(&stored)?;
    let tmp = path.with_extension("json.tmp");
    fs::write(&tmp, json)?;
    fs::rename(&tmp, path)?;
    Ok(())
}

/// Overlay `app` onto the current theme via `theme::apply_with` AND
/// persist the choice. The one call site the settings overlay hits when
/// the user picks a new appearance option — routing everything through
/// one function keeps the "live update + save" pair impossible to
/// half-do.
pub fn commit(cx: &mut gpui::App, app: Appearance) {
    theme::apply_with(cx, &app);
    save(&app);
}

#[cfg(test)]
mod tests {
    use super::*;
    use gpui::px;
    use std::env;

    fn scoped_home(tag: &str) -> PathBuf {
        let dir = env::temp_dir().join(format!(
            "zeta-prefs-{tag}-{}-{:p}",
            std::process::id(),
            &tag
        ));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();
        dir
    }

    #[test]
    fn missing_file_returns_defaults() {
        let dir = scoped_home("missing");
        let path = dir.join(FILE_NAME);
        assert!(!path.exists());
        let app = load_from(&path);
        assert_eq!(app.theme, ThemeId::default());
        assert_eq!(app.font_family.as_ref(), DEFAULT_FONT_FAMILY);
        assert_eq!(app.font_size, DEFAULT_FONT_SIZE);
    }

    #[test]
    fn corrupt_file_returns_defaults() {
        let dir = scoped_home("corrupt");
        let path = dir.join(FILE_NAME);
        fs::write(&path, "{this-is-not-valid-json").unwrap();
        let app = load_from(&path);
        assert_eq!(app.theme, ThemeId::default());
        assert_eq!(app.font_family.as_ref(), DEFAULT_FONT_FAMILY);
        assert_eq!(app.font_size, DEFAULT_FONT_SIZE);
    }

    #[test]
    fn unknown_theme_slug_falls_back_to_default() {
        let stored = StoredPrefs {
            theme: Some("unknown-theme".to_string()),
            font_family: Some("Menlo".to_string()),
            font_size: Some(14.),
        };
        let app = coerce(stored);
        assert_eq!(app.theme, ThemeId::default());
        assert_eq!(app.font_family.as_ref(), "Menlo");
        assert_eq!(app.font_size, px(14.));
    }

    #[test]
    fn out_of_range_font_size_clamps_and_uncurated_family_falls_back() {
        let stored = StoredPrefs {
            theme: Some("gruvbox-dark".to_string()),
            font_family: Some("Papyrus".to_string()),
            font_size: Some(99.),
        };
        let app = coerce(stored);
        assert_eq!(app.theme, ThemeId::GruvboxDark);
        assert_eq!(app.font_family.as_ref(), DEFAULT_FONT_FAMILY);
        assert_eq!(app.font_size, px(theme::MAX_FONT_SIZE_PX));
    }

    #[test]
    fn round_trip_persists_and_reloads() {
        let dir = scoped_home("roundtrip");
        let path = dir.join(FILE_NAME);
        let original = Appearance {
            theme: ThemeId::VscodeDarkPlus,
            font_family: SharedString::new_static("Menlo"),
            font_size: px(16.),
        };
        save_to(&path, &original).unwrap();
        let round = load_from(&path);
        assert_eq!(round.theme, ThemeId::VscodeDarkPlus);
        assert_eq!(round.font_family.as_ref(), "Menlo");
        assert_eq!(round.font_size, px(16.));

        // Rewriting must overwrite atomically, not leave the tmp file behind.
        let next = Appearance {
            theme: ThemeId::GruvboxLight,
            font_family: SharedString::new_static("JetBrains Mono"),
            font_size: px(12.),
        };
        save_to(&path, &next).unwrap();
        assert!(!path.with_extension("json.tmp").exists());
        let after = load_from(&path);
        assert_eq!(after.theme, ThemeId::GruvboxLight);
        assert_eq!(after.font_family.as_ref(), "JetBrains Mono");
        assert_eq!(after.font_size, px(12.));
    }
}
