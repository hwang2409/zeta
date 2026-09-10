use super::*;
use chrono::{DateTime, Utc};
use gpui_kit::component::menu::{DropdownMenu, PopupMenuItem};
use gpui_kit::component::{v_virtual_list, Selectable};
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
        let sizes = Rc::new(vec![
            gpui::size(px(264.), px(44.));
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
                    let menu_id = id.clone();
                    let row = Button::new(format!("session-{id}"))
                        .ghost()
                        .selected(active)
                        .disabled(!view.can_change_session())
                        .w_full()
                        .h(px(44.))
                        .child(
                            div()
                                .debug_selector(|| "session-row".into())
                                .h_flex()
                                .w_full()
                                .min_w_0()
                                .justify_between()
                                .gap_3()
                                .child(div().flex_1().min_w_0().truncate().child(label))
                                .child(
                                    div()
                                        .flex_shrink_0()
                                        .text_size(px(11.))
                                        .text_color(cx.theme().muted_foreground)
                                        .child(age),
                                ),
                        )
                        .on_click(cx.listener(move |view, _, _, cx| {
                            if view.can_change_session()
                                && view.state.active_session.as_ref() != Some(&id)
                            {
                                view.pending_command = true;
                                view.queue(CommandMessage::Resume(id.clone()));
                                cx.notify();
                            }
                        }))
                        .into_any_element();
                    if !view.session_management {
                        return row;
                    }
                    let entity = cx.entity().downgrade();
                    div()
                        .h_flex()
                        .w_full()
                        .items_center()
                        .child(div().flex_1().min_w_0().child(row))
                        .child(
                            Button::new(format!("session-menu-{menu_id}"))
                                .debug_selector(|| "session-menu".into())
                                .ghost()
                                .icon(IconName::Ellipsis)
                                .tooltip("Session actions")
                                .w(px(40.))
                                .h(px(40.))
                                .flex_shrink_0()
                                .disabled(!view.can_change_session())
                                .dropdown_menu(move |menu, _, cx| {
                                    let enabled = entity
                                        .upgrade()
                                        .is_some_and(|view| view.read(cx).can_change_session());
                                    let rename_view = entity.clone();
                                    let delete_view = entity.clone();
                                    let rename_id = menu_id.clone();
                                    let delete_id = menu_id.clone();
                                    menu.item(
                                        PopupMenuItem::new("Rename").disabled(!enabled).on_click(
                                            move |_, window, cx| {
                                                let _ = rename_view.update(cx, |view, cx| {
                                                    view.open_session_edit(
                                                        rename_id.clone(),
                                                        true,
                                                        window,
                                                        cx,
                                                    )
                                                });
                                            },
                                        ),
                                    )
                                    .separator()
                                    .item(
                                        PopupMenuItem::new("Delete").disabled(!enabled).on_click(
                                            move |_, window, cx| {
                                                let _ = delete_view.update(cx, |view, cx| {
                                                    view.open_session_edit(
                                                        delete_id.clone(),
                                                        false,
                                                        window,
                                                        cx,
                                                    )
                                                });
                                            },
                                        ),
                                    )
                                }),
                        )
                        .into_any_element()
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
            .w(px(280.))
            .h_full()
            .flex_shrink_0()
            .p_2()
            .gap_2()
            .bg(cx.theme().sidebar)
            .border_r_1()
            .border_color(cx.theme().border)
            .child(
                div()
                    .h_flex()
                    .items_center()
                    .justify_between()
                    .px_3()
                    .py_4()
                    .child(
                        div()
                            .text_size(px(20.))
                            .font_weight(gpui::FontWeight::BOLD)
                            .child("zeta"),
                    )
                    .child(
                        Button::new("settings")
                            .debug_selector(|| "settings-button".into())
                            .ghost()
                            .label("Settings")
                            .disabled(!can_open_settings)
                            .on_click(cx.listener(|view, _, _, cx| view.open_settings(cx))),
                    ),
            )
            .child(
                Button::new("new-session")
                    .label("New session")
                    .w_full()
                    .h(px(40.))
                    .disabled(!self.can_change_session())
                    .on_click(cx.listener(|view, _, _, cx| view.new_session(cx))),
            )
            .when(self.state.sessions_truncated, |sidebar| {
                sidebar.child(
                    div()
                        .p_2()
                        .text_size(px(12.))
                        .child("Showing a partial session list"),
                )
            })
            .when(self.state.active_session.is_none(), |sidebar| {
                sidebar.child(
                    div()
                        .px_3()
                        .text_size(px(12.))
                        .text_color(cx.theme().muted_foreground)
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
                            .px_3()
                            .text_size(px(12.))
                            .text_color(cx.theme().muted_foreground)
                            .child("Hover over your message to fork from it"),
                    )
                },
            )
            .children(self.render_branches(cx))
    }

    pub fn render_branches(&self, cx: &mut Context<Self>) -> Option<impl IntoElement> {
        if !self.state.session_view.available || self.state.session_view.branches.is_empty() {
            return None;
        }
        let can_switch = self.can_change_session();
        let heading = div()
            .px_3()
            .pt_3()
            .pb_1()
            .text_size(px(11.))
            .text_color(cx.theme().muted_foreground)
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
            let depth = branch.depth.min(8);
            section = section.child(
                Button::new(format!("branch-{id}"))
                    .debug_selector({
                        let id = id.clone();
                        move || format!("branch-row-{id}")
                    })
                    .ghost()
                    .selected(current)
                    .disabled(!can_switch || current)
                    .w_full()
                    .h(px(32.))
                    .child(
                        div()
                            .h_flex()
                            .w_full()
                            .min_w_0()
                            .items_center()
                            .gap_2()
                            .child(div().w(px((depth * 12) as f32)).flex_shrink_0())
                            .child(
                                div()
                                    .flex_shrink_0()
                                    .text_color(if current {
                                        cx.theme().primary
                                    } else {
                                        cx.theme().muted_foreground
                                    })
                                    .child(if current { "*" } else { "-" }),
                            )
                            .child(div().flex_1().min_w_0().truncate().child(label)),
                    )
                    .on_click(cx.listener(move |view, _, _, cx| {
                        if !current {
                            view.switch_branch(id.clone(), cx);
                        }
                    })),
            );
        }
        Some(section)
    }
}
