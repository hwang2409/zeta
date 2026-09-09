//! Session ergonomics data; independent of transport and GPUI.
use serde::{Deserialize, Serialize};

pub const MAX_IMAGE_BYTES: usize = 512 * 1024;

#[derive(Debug, Clone, Deserialize, PartialEq)]
pub struct Branch {
    pub id: String,
    pub label: String,
    pub depth: usize,
    pub current: bool,
}

#[derive(Debug, Clone, Deserialize, PartialEq)]
pub struct SessionSettings {
    pub model: String,
    pub approval_mode: String,
}

#[derive(Debug, Clone, Serialize, PartialEq)]
pub struct ImageAttachment {
    pub name: String,
    pub mime_type: String,
    pub data: String,
    #[serde(skip)]
    pub size: usize,
}

impl ImageAttachment {
    pub fn from_bytes(name: String, bytes: &[u8]) -> Result<Self, String> {
        use base64::Engine;
        if name.is_empty()
            || name.chars().count() > 128
            || name == "."
            || name == ".."
            || name.chars().any(|c| {
                c.is_control()
                    || matches!(c, '/' | '\\' | '\u{061c}' | '\u{200e}' | '\u{200f}' | '\u{202a}'..='\u{202e}' | '\u{2066}'..='\u{2069}')
            })
        {
            return Err("image name must be a filename of at most 128 characters".into());
        }
        if bytes.is_empty() || bytes.len() > MAX_IMAGE_BYTES {
            return Err("images must be between 1 byte and 512 KiB".into());
        }
        let mime_type = if bytes.starts_with(b"\x89PNG\r\n\x1a\n") {
            "image/png"
        } else if bytes.starts_with(b"\xff\xd8\xff") {
            "image/jpeg"
        } else if bytes.starts_with(b"GIF87a") || bytes.starts_with(b"GIF89a") {
            "image/gif"
        } else if bytes.starts_with(b"RIFF") && bytes.get(8..12) == Some(b"WEBP") {
            "image/webp"
        } else {
            return Err("choose a PNG, JPEG, GIF, or WebP image".into());
        };
        let extension = std::path::Path::new(&name)
            .extension()
            .and_then(|extension| extension.to_str())
            .unwrap_or_default()
            .to_ascii_lowercase();
        if !matches!(
            (mime_type, extension.as_str()),
            ("image/png", "png")
                | ("image/jpeg", "jpg" | "jpeg")
                | ("image/gif", "gif")
                | ("image/webp", "webp")
        ) {
            return Err("image extension does not match its type".into());
        }
        Ok(Self {
            name,
            mime_type: mime_type.into(),
            data: base64::engine::general_purpose::STANDARD.encode(bytes),
            size: bytes.len(),
        })
    }

    pub fn from_path(path: &std::path::Path) -> Result<Self, String> {
        use std::io::Read;
        let mut bytes = Vec::new();
        std::fs::File::open(path)
            .map_err(|error| error.to_string())?
            .take(MAX_IMAGE_BYTES as u64 + 1)
            .read_to_end(&mut bytes)
            .map_err(|error| error.to_string())?;
        Self::from_bytes(
            path.file_name()
                .unwrap_or_default()
                .to_string_lossy()
                .into_owned(),
            &bytes,
        )
    }
}

pub const APPROVAL_MODES: [&str; 3] = ["ask", "allow", "deny"];

#[derive(Debug, Clone, Default, PartialEq)]
pub struct SessionView {
    pub available: bool,
    pub branches: Vec<Branch>,
    pub message_ids: std::collections::HashMap<usize, String>,
    pub attachments: std::collections::HashMap<usize, Vec<(String, usize)>>,
    pub notice: Option<String>,
    pub models: Vec<String>,
    pub model_providers: std::collections::BTreeMap<String, String>,
    pub current_model: String,
    pub selected_model: usize,
    pub selected_mode: usize,
}

impl SessionView {
    pub fn approval_mode(&self) -> &'static str {
        APPROVAL_MODES[self.selected_mode.min(APPROVAL_MODES.len() - 1)]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn attachment_names_reject_paths_controls_and_false_extensions() {
        let png = b"\x89PNG\r\n\x1a\n";
        for name in [
            "",
            ".",
            "..",
            "../bad.png",
            "..\\bad.png",
            "bad/path.png",
            "bad\\path.png",
            "bad.jpg",
            "bad",
            "bad.png.exe",
            "bad\u{7f}.png",
        ] {
            assert!(
                ImageAttachment::from_bytes(name.into(), png).is_err(),
                "{name:?}"
            );
        }
        for control in "\u{061c}\u{200e}\u{200f}\u{202a}\u{202b}\u{202c}\u{202d}\u{202e}\u{2066}\u{2067}\u{2068}\u{2069}".chars() {
            assert!(ImageAttachment::from_bytes(format!("bad{control}.png"), png).is_err());
        }
        for (bytes, names) in [
            (png.as_slice(), vec!["safe.PNG", "写真.png"]),
            (b"\xff\xd8\xff".as_slice(), vec!["safe.jpg", "safe.JPEG"]),
            (b"GIF89a".as_slice(), vec!["safe.gif"]),
            (b"RIFF\x04\x00\x00\x00WEBP".as_slice(), vec!["safe.webp"]),
        ] {
            for name in names {
                assert_eq!(
                    ImageAttachment::from_bytes(name.into(), bytes)
                        .unwrap()
                        .name,
                    name
                );
            }
        }
    }
}
