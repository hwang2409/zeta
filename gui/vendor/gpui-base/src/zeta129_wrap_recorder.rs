//! ZETA-129 inner-wrap recorder — TEST-ONLY.
//!
//! `InlineFlow::prepaint` (`src/text/inline_flow.rs`) probes each inner
//! text fragment's `shape_text` call with the SAME wrap_width the
//! fragment's `prepaint_as_root` actually uses (`MaxContent` under the
//! ZETA-129 fix, `Definite(fragment_size.width - padding * 2.)` under
//! `ZETA_GUI_INLINE_FLOW_DEFINITE=1`), and pushes each sample here so a
//! downstream test can assert the invariant "an inline text fragment
//! NEVER wraps inside its own fragment" (i.e. every sample carries
//! `wrap_boundaries == 0`).
//!
//! Gated on `test-support`; release builds compile the recorder and its
//! call site out entirely. See `gui/vendor/README.md` for the full
//! rationale.

use gpui::{Pixels, SharedString};
use std::cell::RefCell;

#[derive(Debug, Clone)]
pub struct Sample {
    pub text: SharedString,
    pub wrap_boundaries: usize,
    pub font_size: Pixels,
}

thread_local! {
    static SAMPLES: RefCell<Vec<Sample>> = const { RefCell::new(Vec::new()) };
}

pub fn clear() {
    SAMPLES.with(|s| s.borrow_mut().clear());
}

pub fn samples() -> Vec<Sample> {
    SAMPLES.with(|s| s.borrow().clone())
}

pub(crate) fn record(sample: Sample) {
    SAMPLES.with(|s| s.borrow_mut().push(sample));
}
