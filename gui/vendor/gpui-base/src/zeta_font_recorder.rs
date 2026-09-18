//! Test-only recorder for text sizes at the rendered text path.

use gpui::Pixels;
use std::cell::RefCell;

thread_local! {
    static SAMPLES: RefCell<Vec<Pixels>> = const { RefCell::new(Vec::new()) };
}

pub fn clear() {
    SAMPLES.with(|samples| samples.borrow_mut().clear());
}

pub fn record(font_size: Pixels) {
    SAMPLES.with(|samples| samples.borrow_mut().push(font_size));
}

pub fn samples() -> Vec<Pixels> {
    SAMPLES.with(|samples| samples.borrow().clone())
}
