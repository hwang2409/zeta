//! Color roles shared by the transcript, editor, and syntax runs. No platform state.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Appearance {
    Light,
    Dark,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Palette {
    pub background: u32,
    pub panel: u32,
    pub border: u32,
    pub nested_border: u32,
    pub text: u32,
    pub muted: u32,
    pub accent: u32,
    pub error: u32,
    pub code_chip: u32,
}

impl Appearance {
    pub fn palette(self) -> Palette {
        match self {
            Self::Light => Palette {
                background: 0xffffff,
                panel: 0xf7f7f8,
                border: 0xc8c8cc,
                nested_border: 0xe0e0e4,
                text: 0x202024,
                muted: 0x62626a,
                accent: 0x002fa7,
                error: 0xb42318,
                code_chip: 0xededf0,
            },
            Self::Dark => Palette {
                background: 0x17171a,
                panel: 0x202024,
                border: 0x48484e,
                nested_border: 0x34343a,
                text: 0xededf0,
                muted: 0xa6a6af,
                accent: 0x9bb9ff,
                error: 0xff9b94,
                code_chip: 0x303036,
            },
        }
    }

    /// Syntax themes supply foregrounds only. Ensure AA against our own surface.
    pub fn syntax_color(self, color: u32) -> u32 {
        let palette = self.palette();
        let mut result = color;
        while contrast(result, palette.background) < 4.5 {
            let channel = |shift: u32| {
                let value = (result >> shift) & 255u32;
                let target = (palette.text >> shift) & 255u32;
                if value < target {
                    value + (target - value).div_ceil(4)
                } else {
                    value - (value - target).div_ceil(4)
                }
            };
            let next = (channel(16) << 16) | (channel(8) << 8) | channel(0);
            if next == result {
                break;
            }
            result = next;
        }
        result
    }
}

fn luminance(color: u32) -> f64 {
    let channel = |shift: u32| {
        let value = f64::from((color >> shift) & 255u32) / 255.;
        if value <= 0.04045 {
            value / 12.92
        } else {
            ((value + 0.055) / 1.055).powf(2.4)
        }
    };
    0.2126 * channel(16) + 0.7152 * channel(8) + 0.0722 * channel(0)
}

pub fn contrast(first: u32, second: u32) -> f64 {
    let a = luminance(first);
    let b = luminance(second);
    (a.max(b) + 0.05) / (a.min(b) + 0.05)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn both_explicit_palettes_have_aa_text_and_code() {
        for appearance in [Appearance::Light, Appearance::Dark] {
            let p = appearance.palette();
            for surface in [p.background, p.panel, p.code_chip] {
                assert!(contrast(p.text, surface) >= 4.5);
            }
            for color in [p.muted, p.accent, p.error] {
                assert!(contrast(color, p.background) >= 4.5);
            }
            for syntax in [0xffffff, 0x000000, 0x888888, 0xff00ff] {
                assert!(contrast(appearance.syntax_color(syntax), p.background) >= 4.5);
            }
        }
    }
}
