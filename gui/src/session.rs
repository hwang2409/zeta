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

#[derive(Debug, Clone, Default, PartialEq)]
pub struct SessionView {
    pub available: bool,
    pub branches: Vec<Branch>,
    pub message_ids: std::collections::HashMap<usize, String>,
    pub attachments: std::collections::HashMap<usize, Vec<(String, usize)>>,
    pub notice: Option<String>,
    pub models: Vec<String>,
    pub settings_open: bool,
    pub selected_model: usize,
    pub selected_mode: usize,
    pub settings_field: usize,
}
