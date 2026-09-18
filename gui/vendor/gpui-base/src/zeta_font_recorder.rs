//! Test-only recorder for text sizes at the rendered text path.

use gpui::Pixels;
use std::cell::{Cell, RefCell};

thread_local! {
    static SAMPLES: RefCell<Vec<Pixels>> = const { RefCell::new(Vec::new()) };
    static ENABLED: Cell<bool> = const { Cell::new(false) };
}

pub fn clear() {
    SAMPLES.with(|samples| samples.borrow_mut().clear());
    ENABLED.with(|enabled| enabled.set(true));
}

pub fn record(font_size: Pixels) {
    ENABLED.with(|enabled| {
        if enabled.get() {
            SAMPLES.with(|samples| samples.borrow_mut().push(font_size));
        }
    });
}

pub fn samples() -> Vec<Pixels> {
    SAMPLES.with(|samples| samples.borrow().clone())
}
