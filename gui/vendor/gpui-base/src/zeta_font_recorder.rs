//! Test-only recorder for text sizes at the rendered text path.

use gpui::Pixels;
use std::cell::RefCell;

thread_local! {
    static SAMPLES: RefCell<Vec<Sample>> = const { RefCell::new(Vec::new()) };
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Role {
    Generic,
    TextView,
    Body,
    Heading1,
    Heading2,
    Heading3,
    InlineCode,
    Fence,
    Tooltip,
    TooltipShortcut,
    SessionMenu,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Sample {
    pub role: Role,
    pub font_size: Pixels,
}

pub fn clear() {
    SAMPLES.with(|samples| samples.borrow_mut().clear());
}

pub fn record(font_size: Pixels) {
    record_role(Role::Generic, font_size);
}

pub fn record_role(role: Role, font_size: Pixels) {
    SAMPLES.with(|samples| samples.borrow_mut().push(Sample { role, font_size }));
}

pub fn samples() -> Vec<Sample> {
    SAMPLES.with(|samples| samples.borrow().clone())
}
