use super::*;
use chrono::{DateTime, Utc};
use gpui::{FocusHandle, MouseButton};
use gpui_kit::component::menu::{DropdownMenu, PopupMenuItem};
use gpui_kit::component::v_virtual_list;
use std::rc::Rc;
use zeta_gui::client::SessionMetadata;

fn one_line(text: &str) -> String {
    text.split_whitespace().collect::<Vec<_>>().join(" ")
}

pub fn session_label(session: &SessionMetadata, transcript: Option<&[TranscriptEntry]>) -> String {
    let name = one_line(&session.name);
    if !name.is_empty() {
        return name;
    }
    let preview = transcript
        .and_then(|rows| {
            rows.iter().find_map(|row| match row {
                TranscriptEntry::User(text) if !text.trim().is_empty() => Some(text.as_str()),
                _ => None,
            })
        })
        .unwrap_or(&session.first_message_preview);
    let preview = one_line(preview);
    if preview.is_empty() {
        "New conversation".into()
    } else {
        preview
    }
}

pub fn relative_age(timestamp: &str, now: DateTime<Utc>) -> String {
    let Ok(time) = DateTime::parse_from_rfc3339(timestamp) else {
        return String::new();
    };
    let seconds = (now - time.with_timezone(&Utc)).num_seconds().max(0);
    match seconds {
        0..60 => "now".into(),
        60..3600 => format!("{}m", seconds / 60),
        3600..86400 => format!("{}h", seconds / 3600),
        86400..2592000 => format!("{}d", seconds / 86400),
        _ => time.format("%b %d").to_string(),
    }
}

/// Fetch or create a persistent focus handle keyed by `id` for a sidebar row.
/// Stored on `ZetaView::sidebar_row_focus` so the handle survives redraws AND
/// tests can look it up by the same key the renderer uses — `window`'s keyed
/// state pool only reads at layout/paint time and would panic outside those.
/// `id` is unique per row (session id / branch id).
///
/// The handle is created as a real tab stop with `tab_index(0)`. `cx.focus_handle()`
/// defaults `tab_stop=false`, and `.tab_index(0)` on the div only touches its own
/// `Interactivity` — it does NOT push through to a `tracked_focus_handle` (see
/// gpui `elements/div.rs`, `paint_state`: the sync only fires when there is no
/// tracked handle). Without setting the flag here, `window.focus_next` would
/// skip every sidebar row even though `tab_stops` contains their handles.
fn row_focus_handle(view: &ZetaView, cx: &mut App, id: &str) -> FocusHandle {
    let mut map = view.sidebar_row_focus.borrow_mut();
    if let Some(handle) = map.get(id) {
        return handle.clone();
    }
    let handle = cx.focus_handle().tab_stop(true).tab_index(0);
    map.insert(id.to_owned(), handle.clone());
    handle
}

impl ZetaView {
    pub fn render_sidebar(&self, window: &mut Window, cx: &mut Context<Self>) -> impl IntoElement {
        // Prune stale focus handles keyed by session id / branch id. Each new
        // branch head is a fresh UUID, so without this the map grows for the
        // app's lifetime. Retain only entries whose key is still live in the
        // current render; surviving rows keep their handle identity across
        // redraws so tab focus does not jump when unrelated rows change.
        self.prune_sidebar_focus_handles();
        // Sidebar rows sit in a virtual list. Every row lands on the
        // SIDEBAR_ROW_HEIGHT floor so the wiki-run column reads as an even
        // rhythm regardless of session name length.
        let sizes = Rc::new(vec![
            gpui::size(
                theme::SIDEBAR_WIDTH,
                theme::SIDEBAR_ROW_HEIGHT
            );
            self.state.sessions.len()
        ]);
        let sessions = v_virtual_list(cx.entity(), "sessions", sizes, |view, range, window, cx| {
            let now = Utc::now();
            range
                .map(|index| {
                    let session = &view.state.sessions[index];
                    let id = session.session_id.clone();
                    let active = view.state.active_session.as_ref() == Some(&id);
                    let rows = if active {
                        Some(view.state.transcript.as_slice())
                    } else {
                        view.state.saved_transcripts.get(&id).map(Vec::as_slice)
                    };
                    let label = session_label(session, rows);
                    let age = relative_age(&session.updated_at, now);
                    view.render_session_row(index, id, label, age, active, window, cx)
                })
                .collect()
        })
        .track_scroll(&self.sidebar_scroll)
        .flex_1()
        .min_h_0();
        let can_open_settings = self.can_change_session()
            && self.state.session_view.available
            && self.state.active_session.is_some();
        div()
            .v_flex()
            .w(theme::SIDEBAR_WIDTH)
            .h_full()
            .flex_shrink_0()
            .bg(cx.theme().sidebar)
            .border_r_1()
            // Sidebar edge routes through the SUBTLE border tier so the
            // column reads as a seam, not a hard rule. Contract line 3.
            .border_color(cx.theme().sidebar_border)
            .child(self.render_sidebar_header(can_open_settings, cx))
            .child(self.render_sidebar_new_session(cx))
            .when(self.state.sessions_truncated, |sidebar| {
                sidebar.child(
                    div()
                        .debug_selector(|| "sidebar-truncated".into())
                        .px(theme::SIDEBAR_ROW_PADDING_X)
                        .py(theme::SIDEBAR_ROW_PADDING_Y)
                        .text_color(theme::palette::text_faint())
                        .child("Showing a partial session list"),
                )
            })
            .when(self.state.active_session.is_none(), |sidebar| {
                sidebar.child(
                    div()
                        .debug_selector(|| "sidebar-hint-no-session".into())
                        .px(theme::SIDEBAR_ROW_PADDING_X)
                        .py(theme::SIDEBAR_ROW_PADDING_Y)
                        .text_color(theme::palette::text_faint())
                        .child("Create or select a session to use Settings"),
                )
            })
            .child(sessions)
            .when(
                self.state.session_view.available
                    && !self.state.session_view.message_ids.is_empty(),
                |sidebar| {
                    // The fork hint anchors the whole sidebar tail (fork
                    // hint + Branches heading + branch rows). At the tail
                    // the sidebar sits directly above the composer strip
                    // in the main column, and the wiki-run "breathing but
                    // compact" rhythm asks for a visible seam before the
                    // tail so the two surfaces read as separate. A top
                    // border at the sidebar-border (subtle) tier plus
                    // 8px vertical padding gives that seam without
                    // fighting the sessions list rhythm above.
                    sidebar.child(
                        div()
                            .debug_selector(|| "sidebar-hint-fork".into())
                            .px(theme::SIDEBAR_ROW_PADDING_X)
                            .pt_2()
                            .pb_1()
                            .mt_2()
                            .border_t_1()
                            .border_color(cx.theme().sidebar_border)
                            .text_color(theme::palette::text_faint())
                            .child("Hover over your message to fork from it"),
                    )
                },
            )
            .children(self.render_branches(window, cx))
    }

    /// Drop focus handles whose key (session id / `branch:<id>`) is no longer
    /// live. Called at the top of every `render_sidebar`. Without this the
    /// handle map is insert-only and grows unbounded — normal branch churn
    /// mints fresh UUIDs, and stale entries would pin a `FocusHandle` (and
    /// its entry in the window focus map) for the app's lifetime.
    fn prune_sidebar_focus_handles(&self) {
        let branches_live = self.state.session_view.available;
        self.sidebar_row_focus.borrow_mut().retain(|key, _| {
            if let Some(branch_id) = key.strip_prefix("branch:") {
                branches_live
                    && self
                        .state
                        .session_view
                        .branches
                        .iter()
                        .any(|branch| branch.id == branch_id)
            } else {
                self.state
                    .sessions
                    .iter()
                    .any(|session| session.session_id.as_str() == key)
            }
        });
    }

    fn render_sidebar_header(
        &self,
        can_open_settings: bool,
        cx: &mut Context<Self>,
    ) -> gpui::AnyElement {
        // Header band: single-line title with the Settings action pinned to
        // the right. Sits at the header-band-1 height so it aligns with the
        // run-header rhythm across the main column.
        div()
            .debug_selector(|| "sidebar-header".into())
            .h_flex()
            .items_center()
            .justify_between()
            .h(theme::HEADER_BAND1_MIN_HEIGHT)
            .px(theme::SIDEBAR_ROW_PADDING_X)
            .border_b_1()
            .border_color(cx.theme().sidebar_border)
            .child(
                div()
                    .font_weight(gpui::FontWeight::SEMIBOLD)
                    .text_color(cx.theme().foreground)
                    .child("zeta"),
            )
            .child(
                Button::new("settings")
                    .debug_selector(|| "settings-button".into())
                    .ghost()
                    .compact()
                    .label("Settings")
                    .disabled(!can_open_settings)
                    .on_click(cx.listener(|view, _, _, cx| view.open_settings(cx))),
            )
            .into_any_element()
    }

    fn render_sidebar_new_session(&self, cx: &mut Context<Self>) -> gpui::AnyElement {
        div()
            .debug_selector(|| "sidebar-new-session".into())
            .px(theme::SIDEBAR_ROW_PADDING_X)
            .py(theme::SIDEBAR_ROW_PADDING_Y)
            .child(
                // Ghost variant keeps the button transparent at rest and
                // borrows the wiki "flat action" look — hover fills to the
                // element tint, focus adds an accent ring.
                Button::new("new-session")
                    .ghost()
                    .label("New session")
                    .w_full()
                    .h(theme::SIDEBAR_ROW_HEIGHT)
                    .font_weight(gpui::FontWeight::SEMIBOLD)
                    .disabled(!self.can_change_session())
                    .on_click(cx.listener(|view, _, _, cx| view.new_session(cx))),
            )
            .into_any_element()
    }

    #[allow(clippy::too_many_arguments)]
    fn render_session_row(
        &self,
        index: usize,
        id: String,
        label: String,
        age: String,
        active: bool,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) -> gpui::AnyElement {
        // Sidebar rows are focusable controls, not pointer-only divs.
        // Contract line 81 pins the current row to NO fill + accent text +
        // gutter dot, which rules out Kit's `.ghost().selected()` treatment
        // (that fills the row with active-tint). Instead we build a
        // focusable div with a keyed focus handle, Enter/Space activation
        // and a visible keyboard cursor (solid accent fill when focused).
        let can_switch = self.can_change_session();
        let focus_handle = row_focus_handle(self, cx, &id);
        let focused = focus_handle.is_focused(window);
        // Semantic color tier — the current item paints in the accent hue,
        // switchable rows in the normal foreground, disabled rows in muted.
        // Weight stays SEMIBOLD across every primary label (contract line
        // 81: `ticket 600`) — hierarchy comes from color, not size.
        let label_color = if active {
            cx.theme().primary
        } else if can_switch {
            cx.theme().foreground
        } else {
            cx.theme().muted_foreground
        };
        // Solid-accent keyboard cursor whenever the row has focus — even the
        // current row must show the cursor so keyboard-only users see WHICH
        // row would activate on Enter/Space. The current-item "no fill" rule
        // (contract line 81) applies to the UNFOCUSED selected state; a
        // focused row overrides it. The fill inverts to canvas text so
        // contrast holds against the accent.
        let (row_bg, row_fg) = if focused {
            (cx.theme().primary, cx.theme().primary_foreground)
        } else {
            (gpui::transparent_black(), label_color)
        };
        let click_id = id.clone();
        let key_id = id.clone();
        let can_activate = can_switch && !active;
        let row = div()
            .id(("session-row", index))
            .debug_selector(|| "session-row".into())
            .track_focus(&focus_handle)
            .tab_index(0)
            .aria_label(label.clone())
            .role(gpui::accesskit::Role::Button)
            .h_flex()
            .items_center()
            .w_full()
            .min_w_0()
            .flex_1()
            .min_h(theme::SIDEBAR_ROW_HEIGHT)
            .h(theme::SIDEBAR_ROW_HEIGHT)
            .pr(theme::SIDEBAR_ROW_PADDING_X)
            .py(theme::SIDEBAR_ROW_PADDING_Y)
            .bg(row_bg)
            .text_color(row_fg)
            .when(can_switch && !active && !focused, |row| {
                row.cursor_pointer()
                    .hover(|style| style.bg(cx.theme().muted))
            })
            .child(self.render_row_gutter(active, cx))
            .child(
                div()
                    .flex_1()
                    .min_w_0()
                    .truncate()
                    .font_weight(gpui::FontWeight::SEMIBOLD)
                    .child(label),
            )
            .child(
                div()
                    .flex_shrink_0()
                    .pl_2()
                    .text_color(if focused {
                        cx.theme().primary_foreground
                    } else {
                        theme::palette::text_faint()
                    })
                    .child(age),
            )
            .when(can_activate, |row| {
                let click_id = click_id.clone();
                row.on_mouse_down(
                    MouseButton::Left,
                    cx.listener(move |view, _, _, cx| {
                        view.activate_session(click_id.clone(), cx);
                    }),
                )
            })
            .when(can_activate, |row| {
                let key_id = key_id.clone();
                row.on_key_down(cx.listener(move |view, event: &gpui::KeyDownEvent, _, cx| {
                    if is_activation_key(&event.keystroke.key) {
                        view.activate_session(key_id.clone(), cx);
                        cx.stop_propagation();
                    }
                }))
            })
            .into_any_element();

        if !self.session_management {
            return row;
        }
        // Right-side dropdown for rename/delete. Kept as a Button so the
        // menu integration and keyboard accessibility come from Kit.
        let entity = cx.entity().downgrade();
        let menu_id = id;
        div()
            .h_flex()
            .w_full()
            .items_center()
            .child(row)
            .child(
                Button::new(format!("session-menu-{menu_id}"))
                    .debug_selector(|| "session-menu".into())
                    .ghost()
                    .compact()
                    .icon(IconName::Ellipsis)
                    .tooltip("Session actions")
                    .h(theme::SIDEBAR_ROW_HEIGHT)
                    .flex_shrink_0()
                    .disabled(!self.can_rename_session())
                    .dropdown_menu(move |menu, _, cx| {
                        let delete_enabled = entity
                            .upgrade()
                            .is_some_and(|view| view.read(cx).can_change_session());
                        let rename_enabled = entity
                            .upgrade()
                            .is_some_and(|view| view.read(cx).can_rename_session());
                        let rename_view = entity.clone();
                        let delete_view = entity.clone();
                        let rename_id = menu_id.clone();
                        let delete_id = menu_id.clone();
                        menu.item(
                            PopupMenuItem::new("Rename")
                                .disabled(!rename_enabled)
                                .on_click(move |_, window, cx| {
                                    let _ = rename_view.update(cx, |view, cx| {
                                        view.open_session_edit(rename_id.clone(), true, window, cx)
                                    });
                                }),
                        )
                        .separator()
                        .item(
                            PopupMenuItem::new("Delete")
                                .disabled(!delete_enabled)
                                .on_click(move |_, window, cx| {
                                    let _ = delete_view.update(cx, |view, cx| {
                                        view.open_session_edit(delete_id.clone(), false, window, cx)
                                    });
                                }),
                        )
                    }),
            )
            .into_any_element()
    }

    /// Shared left-gutter helper. Reserves a fixed lane for two overlapping
    /// signals — the current-item accent dot (contract line 81) and the
    /// lost-connection attention rail (contract line 83). Called from both
    /// session and branch rows so the danger rail lights up on every
    /// sidebar row simultaneously, not just sessions.
    fn render_row_gutter(&self, current: bool, cx: &mut Context<Self>) -> gpui::AnyElement {
        let attention = matches!(self.state.connection, ConnectionState::Lost(_));
        div()
            .debug_selector(|| "session-gutter".into())
            .flex()
            .items_center()
            .justify_start()
            .flex_shrink_0()
            .w(theme::SIDEBAR_GUTTER_WIDTH)
            .h_full()
            .when(attention, |gutter| {
                gutter
                    .border_l(theme::ATTENTION_RAIL_WIDTH)
                    .border_color(cx.theme().danger)
            })
            .when(current, |gutter| {
                gutter.child(
                    div()
                        .debug_selector(|| "session-current-dot".into())
                        .ml(theme::SIDEBAR_CURRENT_DOT_INSET)
                        .w(theme::SIDEBAR_CURRENT_DOT_SIZE)
                        .h(theme::SIDEBAR_CURRENT_DOT_SIZE)
                        .rounded_full()
                        .bg(cx.theme().primary),
                )
            })
            .into_any_element()
    }

    pub fn render_branches(
        &self,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) -> Option<impl IntoElement> {
        if !self.state.session_view.available || self.state.session_view.branches.is_empty() {
            return None;
        }
        let can_switch = self.can_change_session();
        let heading = div()
            .debug_selector(|| "sidebar-branches-heading".into())
            .px(theme::SIDEBAR_ROW_PADDING_X)
            .pt_3()
            .pb_1()
            .text_color(theme::palette::text_faint())
            .child("Branches");
        // The branches strip is the bottom-most sidebar surface — the
        // composer strip in the main column sits directly beside its lower
        // edge. `pb_2` gives the last branch row a small floor so the
        // strip does not visually collide with the composer chrome.
        let mut section = div()
            .id("branches-list")
            .v_flex()
            .flex_shrink_0()
            .max_h(px(200.))
            .pb_2()
            .overflow_y_scroll()
            .child(heading);
        for branch in &self.state.session_view.branches {
            let id = branch.id.clone();
            let label = branch.label.clone();
            let current = branch.current;
            let depth = branch.depth.min(4);
            let focus_handle = row_focus_handle(self, cx, &format!("branch:{id}"));
            let focused = focus_handle.is_focused(window);
            // Same tier as sessions — every primary label sits at 600
            // weight, with the current one taking the accent color.
            let label_color = if current {
                cx.theme().primary
            } else if can_switch {
                cx.theme().foreground
            } else {
                cx.theme().muted_foreground
            };
            // Same focus-cursor contract as session rows: any focused row —
            // even the current one — paints the accent cursor so keyboard
            // users see the active tab-stop. Unfocused rows keep the
            // no-fill accent-text look for the current branch.
            let (row_bg, row_fg) = if focused {
                (cx.theme().primary, cx.theme().primary_foreground)
            } else {
                (gpui::transparent_black(), label_color)
            };
            let click_id = id.clone();
            let key_id = id.clone();
            let debug_click_id = id.clone();
            let can_activate = can_switch && !current;
            section = section.child(
                div()
                    .id(gpui::SharedString::from(format!("branch-row-{id}")))
                    .debug_selector({
                        let id = debug_click_id;
                        move || format!("branch-row-{id}")
                    })
                    .track_focus(&focus_handle)
                    .tab_index(0)
                    .aria_label(label.clone())
                    .role(gpui::accesskit::Role::Button)
                    .h_flex()
                    .items_center()
                    .w_full()
                    .min_w_0()
                    .h(theme::SIDEBAR_NESTED_ROW_HEIGHT)
                    .pr(theme::SIDEBAR_ROW_PADDING_X)
                    .bg(row_bg)
                    .text_color(row_fg)
                    .when(can_switch && !current && !focused, |row| {
                        row.cursor_pointer()
                            .hover(|style| style.bg(cx.theme().muted))
                    })
                    // Depth indent stands independent of the fixed dot
                    // gutter — contract line 81 pins the dot to `left 4px`
                    // regardless of nesting. Merging the two into one span
                    // (as we did in round 1) drifted the dot rightward as
                    // depth grew; splitting them holds the position.
                    .child(div().flex_shrink_0().w(px((depth * 40) as f32)).h_full())
                    .child(self.render_row_gutter(current, cx))
                    .child(
                        div()
                            .flex_1()
                            .min_w_0()
                            .truncate()
                            .font_weight(gpui::FontWeight::SEMIBOLD)
                            .child(label),
                    )
                    .when(can_activate, |row| {
                        let click_id = click_id.clone();
                        row.on_mouse_down(
                            MouseButton::Left,
                            cx.listener(move |view, _, _, cx| {
                                view.switch_branch(click_id.clone(), cx);
                            }),
                        )
                    })
                    .when(can_activate, |row| {
                        let key_id = key_id.clone();
                        row.on_key_down(cx.listener(
                            move |view, event: &gpui::KeyDownEvent, _, cx| {
                                if is_activation_key(&event.keystroke.key) {
                                    view.switch_branch(key_id.clone(), cx);
                                    cx.stop_propagation();
                                }
                            },
                        ))
                    }),
            );
        }
        Some(section)
    }

    fn activate_session(&mut self, id: String, cx: &mut Context<Self>) {
        if self.can_change_session() && self.state.active_session.as_ref() != Some(&id) {
            self.pending_command = true;
            self.queue(CommandMessage::Resume(id));
            cx.notify();
        }
    }
}

/// A row activates on Enter or Space — the standard button-role keyboard
/// contract every accessible list of buttons owes its users.
fn is_activation_key(key: &str) -> bool {
    matches!(key, "enter" | "space")
}
