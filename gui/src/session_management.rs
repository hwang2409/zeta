use super::*;

pub enum SessionEdit {
    Rename {
        id: String,
        input: Entity<TextareaState>,
    },
    Delete {
        id: String,
        label: String,
    },
}

impl ZetaView {
    pub fn can_rename_session(&self) -> bool {
        self.state.connection == ConnectionState::Connected
            && !self.pending_command
            && self.session_edit.is_none()
    }

    pub fn open_session_edit(
        &mut self,
        id: String,
        rename: bool,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        let enabled = if rename {
            self.can_rename_session()
        } else {
            self.can_change_session()
        };
        if !self.session_management || !enabled || self.settings_open {
            return;
        }
        let Some(session) = self.state.sessions.iter().find(|row| row.session_id == id) else {
            return;
        };
        self.command_error = None;
        self.session_edit = Some(if rename {
            let name = session.name.clone();
            let input = cx.new(|cx| {
                let mut input = TextareaState::new(window, cx).placeholder("Session name");
                input.set_value(name, window, cx);
                input
            });
            window.focus(&input.focus_handle(cx), cx);
            SessionEdit::Rename { id, input }
        } else {
            window.focus(&self.session_edit_focus, cx);
            SessionEdit::Delete {
                id,
                label: sidebar::session_label(session, None),
            }
        });
        cx.notify();
    }

    pub fn close_session_edit(&mut self, window: &mut Window, cx: &mut Context<Self>) {
        if self.pending_command {
            return;
        }
        self.session_edit = None;
        window.focus(&self.composer.focus_handle(cx), cx);
        cx.notify();
    }

    pub fn commit_session_edit(&mut self, cx: &mut Context<Self>) {
        if self.pending_command
            || self.state.connection != ConnectionState::Connected
            || (matches!(self.session_edit, Some(SessionEdit::Delete { .. }))
                && (self.state.streaming || !self.state.approvals.is_empty()))
        {
            return;
        }
        let command = match &self.session_edit {
            Some(SessionEdit::Rename { id, input }) => {
                CommandMessage::RenameSession(id.clone(), input.read(cx).value().to_string())
            }
            Some(SessionEdit::Delete { id, .. }) => CommandMessage::DeleteSession(id.clone()),
            None => return,
        };
        self.pending_command = true;
        self.queue(command);
        cx.notify();
    }

    pub fn render_session_edit(&self, cx: &mut Context<Self>) -> gpui::AnyElement {
        let Some(edit) = &self.session_edit else {
            return div().into_any_element();
        };
        let rename = matches!(edit, SessionEdit::Rename { .. });
        let content = match edit {
            SessionEdit::Rename { input, .. } => div().v_flex().gap_3()
                .child(Textarea::new(input).h(px(44.)))
                .child("Leave empty to use the first message."),
            SessionEdit::Delete { label, id } => div().v_flex().gap_3()
                .child(div().truncate().child(label.clone()))
                .child("Delete this conversation and its stored files? This cannot be undone.")
                .when(self.state.active_session.as_ref() == Some(id), |view| view.child("This conversation is active. Cancel and select another session before deleting it.")),
        };
        div()
            .absolute()
            .inset_0()
            .debug_selector(|| "session-edit".into())
            .track_focus(&self.session_edit_focus)
            .occlude()
            .bg(gpui::black().opacity(0.55))
            .h_flex()
            .items_center()
            .justify_center()
            .child(
                div()
                    .v_flex()
                    .w(px(480.))
                    .p_5()
                    .gap_3()
                    .bg(cx.theme().background)
                    .border_1()
                    .border_color(cx.theme().border)
                    .child(
                        div()
                            .text_size(px(18.))
                            .font_weight(gpui::FontWeight::BOLD)
                            .child(if rename {
                                "Rename session"
                            } else {
                                "Delete session"
                            }),
                    )
                    .child(content)
                    .when_some(self.command_error.clone(), |view, error| {
                        view.child(
                            div()
                                .debug_selector(|| "session-edit-error".into())
                                .child(Alert::error("session-edit-error", error)),
                        )
                    })
                    .child(
                        div()
                            .h_flex()
                            .justify_end()
                            .gap_2()
                            .child(
                                Button::new("session-edit-cancel")
                                    .debug_selector(|| "session-edit-cancel".into())
                                    .ghost()
                                    .label("Cancel")
                                    .h(px(40.))
                                    .disabled(self.pending_command)
                                    .on_click(cx.listener(|view, _, window, cx| {
                                        view.close_session_edit(window, cx)
                                    })),
                            )
                            .child(
                                Button::new("session-edit-confirm")
                                    .debug_selector(|| "session-edit-confirm".into())
                                    .primary()
                                    .label(if self.pending_command {
                                        if rename {
                                            "Saving…"
                                        } else {
                                            "Deleting…"
                                        }
                                    } else if rename {
                                        "Save"
                                    } else {
                                        "Delete"
                                    })
                                    .h(px(40.))
                                    .disabled(self.pending_command)
                                    .on_click(
                                        cx.listener(|view, _, _, cx| view.commit_session_edit(cx)),
                                    ),
                            ),
                    ),
            )
            .into_any_element()
    }
}
