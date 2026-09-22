use super::Mode;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum RegisterShape {
    Charwise,
    Linewise,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) struct Snapshot {
    pub(super) text: String,
    pub(super) cursor: usize,
    pub(super) mode: Mode,
}
