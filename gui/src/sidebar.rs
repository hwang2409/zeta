use super::*;
use chrono::{DateTime, Utc};
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

impl ZetaView {
    pub fn render_sidebar(&self, cx: &mut Context<Self>) -> impl IntoElement {
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
        let sessions = v_virtual_list(cx.entity(), "sessions", sizes, |view, range, _, cx| {
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
                    view.render_session_row(index, id, label, age, active, cx)
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
            .border_color(cx.theme().border)
            .child(self.render_sidebar_header(can_open_settings, cx))
            .child(self.render_sidebar_new_session(cx))
            .when(self.state.sessions_truncated, |sidebar| {
                sidebar.child(
                    div()
                        .debug_selector(|| "sidebar-truncated".into())
                        .px(theme::SIDEBAR_ROW_PADDING_X)
                        .py(theme::SIDEBAR_ROW_PADDING_Y)
                        .text_size(px(12.))
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
                        .text_size(px(12.))
                        .text_color(theme::palette::text_faint())
                        .child("Create or select a session to use Settings"),
                )
            })
            .child(sessions)
            .when(
                self.state.session_view.available
                    && !self.state.session_view.message_ids.is_empty(),
                |sidebar| {
                    sidebar.child(
                        div()
                            .debug_selector(|| "sidebar-hint-fork".into())
                            .px(theme::SIDEBAR_ROW_PADDING_X)
                            .py(theme::SIDEBAR_ROW_PADDING_Y)
                            .text_size(px(12.))
                            .text_color(theme::palette::text_faint())
                            .child("Hover over your message to fork from it"),
                    )
                },
            )
            .children(self.render_branches(cx))
    }

    fn render_sidebar_header(
        &self,
        can_open_settings: bool,
        cx: &mut Context<Self>,
    ) -> gpui::AnyElement {
        // Header band: single-line title with the Settings action pinned to
        // the right. Sits at the header-band-1 height so it aligns with the
        // status-strip rhythm across the main column.
        div()
            .debug_selector(|| "sidebar-header".into())
            .h_flex()
            .items_center()
            .justify_between()
            .h(theme::HEADER_BAND1_MIN_HEIGHT)
            .px(theme::SIDEBAR_ROW_PADDING_X)
            .border_b_1()
            .border_color(cx.theme().border)
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

    fn render_session_row(
        &self,
        index: usize,
        id: String,
        label: String,
        age: String,
        active: bool,
        cx: &mut Context<Self>,
    ) -> gpui::AnyElement {
        // Custom row rather than a `.ghost().selected(...)` Button: contract
        // line 81 requires the current row to paint NO fill (just accent
        // text + a small dot in the gutter). Kit's selected variant fills
        // with active-tint, so we build the row manually and route hover /
        // click through plain div handlers.
        let can_switch = self.can_change_session();
        let (label_color, label_weight) = if active {
            (cx.theme().primary, gpui::FontWeight::SEMIBOLD)
        } else if can_switch {
            (cx.theme().foreground, gpui::FontWeight::NORMAL)
        } else {
            (cx.theme().muted_foreground, gpui::FontWeight::NORMAL)
        };
        let click_id = id.clone();
        let row = div()
            .id(("session-row", index))
            .debug_selector(|| "session-row".into())
            .h_flex()
            .items_center()
            .w_full()
            .min_w_0()
            .flex_1()
            .min_h(theme::SIDEBAR_ROW_HEIGHT)
            .h(theme::SIDEBAR_ROW_HEIGHT)
            .pr(theme::SIDEBAR_ROW_PADDING_X)
            .py(theme::SIDEBAR_ROW_PADDING_Y)
            .when(can_switch && !active, |row| {
                row.cursor_pointer()
                    .hover(|style| style.bg(cx.theme().muted))
            })
            .child(self.render_session_gutter(active, cx))
            .child(
                div()
                    .flex_1()
                    .min_w_0()
                    .truncate()
                    .text_color(label_color)
                    .font_weight(label_weight)
                    .child(label),
            )
            .child(
                div()
                    .flex_shrink_0()
                    .pl_2()
                    .text_size(px(12.))
                    .text_color(theme::palette::text_faint())
                    .child(age),
            )
            .when(can_switch && !active, |row| {
                let click_id = click_id.clone();
                row.on_mouse_down(
                    gpui::MouseButton::Left,
                    cx.listener(move |view, _, _, cx| {
                        if view.can_change_session()
                            && view.state.active_session.as_ref() != Some(&click_id)
                        {
                            view.pending_command = true;
                            view.queue(CommandMessage::Resume(click_id.clone()));
                            cx.notify();
                        }
                    }),
                )
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

    fn render_session_gutter(&self, active: bool, cx: &mut Context<Self>) -> gpui::AnyElement {
        // The left gutter carries the current-item accent dot and doubles as
        // the attention-rail slot. Reserving the width unconditionally keeps
        // the label's left edge stable when the dot appears or disappears.
        let attention = matches!(self.state.connection, ConnectionState::Lost(_));
        div()
            .debug_selector(|| "session-gutter".into())
            .flex()
            .items_center()
            .justify_center()
            .flex_shrink_0()
            .w(theme::SIDEBAR_GUTTER_WIDTH)
            .h_full()
            .when(attention, |gutter| {
                gutter
                    .border_l(theme::ATTENTION_RAIL_WIDTH)
                    .border_color(cx.theme().danger)
            })
            .when(active, |gutter| {
                gutter.child(
                    div()
                        .debug_selector(|| "session-current-dot".into())
                        .w(theme::SIDEBAR_CURRENT_DOT_SIZE)
                        .h(theme::SIDEBAR_CURRENT_DOT_SIZE)
                        .rounded_full()
                        .bg(cx.theme().primary),
                )
            })
            .into_any_element()
    }

    pub fn render_branches(&self, cx: &mut Context<Self>) -> Option<impl IntoElement> {
        if !self.state.session_view.available || self.state.session_view.branches.is_empty() {
            return None;
        }
        let can_switch = self.can_change_session();
        let heading = div()
            .debug_selector(|| "sidebar-branches-heading".into())
            .px(theme::SIDEBAR_ROW_PADDING_X)
            .pt_3()
            .pb_1()
            .text_size(px(11.))
            .text_color(theme::palette::text_faint())
            .child("Branches");
        let mut section = div()
            .id("branches-list")
            .v_flex()
            .flex_shrink_0()
            .max_h(px(200.))
            .overflow_y_scroll()
            .child(heading);
        for branch in &self.state.session_view.branches {
            let id = branch.id.clone();
            let label = branch.label.clone();
            let current = branch.current;
            let depth = branch.depth.min(4);
            let (label_color, label_weight) = if current {
                (cx.theme().primary, gpui::FontWeight::SEMIBOLD)
            } else if can_switch {
                (cx.theme().foreground, gpui::FontWeight::NORMAL)
            } else {
                (cx.theme().muted_foreground, gpui::FontWeight::NORMAL)
            };
            let click_id = id.clone();
            let debug_click_id = id.clone();
            section = section.child(
                div()
                    .id(gpui::SharedString::from(format!("branch-row-{id}")))
                    .debug_selector({
                        let id = debug_click_id;
                        move || format!("branch-row-{id}")
                    })
                    .h_flex()
                    .items_center()
                    .w_full()
                    .min_w_0()
                    .h(theme::SIDEBAR_NESTED_ROW_HEIGHT)
                    .pr(theme::SIDEBAR_ROW_PADDING_X)
                    .when(can_switch && !current, |row| {
                        row.cursor_pointer()
                            .hover(|style| style.bg(cx.theme().muted))
                    })
                    // Nested rows indent 40px per depth level (contract line
                    // 81 pins one indent step at the same size as a row).
                    .child(
                        div()
                            .flex_shrink_0()
                            .w(px(
                                (depth * 40) as f32 + f32::from(theme::SIDEBAR_GUTTER_WIDTH)
                            ))
                            .h_full()
                            .flex()
                            .items_center()
                            .justify_center()
                            .when(current, |gutter| {
                                gutter.child(
                                    div()
                                        .debug_selector(|| "branch-current-dot".into())
                                        .w(theme::SIDEBAR_CURRENT_DOT_SIZE)
                                        .h(theme::SIDEBAR_CURRENT_DOT_SIZE)
                                        .rounded_full()
                                        .bg(cx.theme().primary),
                                )
                            }),
                    )
                    .child(
                        div()
                            .flex_1()
                            .min_w_0()
                            .truncate()
                            .text_color(label_color)
                            .font_weight(label_weight)
                            .child(label),
                    )
                    .when(can_switch && !current, |row| {
                        let click_id = click_id.clone();
                        row.on_mouse_down(
                            gpui::MouseButton::Left,
                            cx.listener(move |view, _, _, cx| {
                                view.switch_branch(click_id.clone(), cx);
                            }),
                        )
                    }),
            );
        }
        Some(section)
    }
}
