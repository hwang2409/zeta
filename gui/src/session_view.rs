//! Thin views for the session tree and settings dialog.
use super::*;
use std::rc::Rc;
use zeta_gui::session::SessionSettings;

pub const MODES: [&str; 3] = ["ask", "allow", "deny"];

struct LabelTooltip(String, Palette);
impl Render for LabelTooltip {
    fn render(&mut self, _: &mut Window, _: &mut Context<Self>) -> impl IntoElement {
        div()
            .max_w(px(480.))
            .p_2()
            .bg(gpui::rgb(self.1.panel))
            .text_color(gpui::rgb(self.1.text))
            .child(self.0.clone())
    }
}

impl ZetaView {
    pub(super) fn session_button(
        &self,
        id: String,
        label: String,
        cx: &mut Context<Self>,
        action: impl Fn(&mut Self, &mut Window, &mut Context<Self>) + 'static,
    ) -> gpui::Stateful<gpui::Div> {
        let action = Rc::new(action);
        let click = action.clone();
        let p = self.appearance.palette();
        div()
            .id(id)
            .tab_index(0)
            .min_h(px(40.))
            .px_2()
            .py_2()
            .cursor_pointer()
            .text_color(gpui::rgb(p.text))
            .hover(|row| row.bg(gpui::rgb(p.border)))
            .focus(|row| row.border_1().border_color(gpui::rgb(p.accent)))
            .child(label)
            .on_click(cx.listener(move |view, _, window, cx| click(view, window, cx)))
            .on_key_down(cx.listener(move |view, event: &KeyDownEvent, window, cx| {
                if event.keystroke.key == "enter" {
                    action(view, window, cx);
                    cx.stop_propagation();
                }
            }))
    }

    pub(super) fn render_session_tools(&self, cx: &mut Context<Self>) -> impl IntoElement {
        let p = self.appearance.palette();
        let mut section = div().flex().flex_col().gap_2();
        if !self.state.session_view.available || self.state.active_session.is_none() {
            return section;
        }
        section = section
            .child(
                self.session_button("settings".into(), "settings".into(), cx, |view, _, cx| {
                    if view.can_change_session() {
                        view.pending_command = true;
                        view.queue(CommandMessage::LoadSettings);
                        cx.notify();
                    }
                }),
            )
            .child(
                div()
                    .mt_3()
                    .text_size(px(12.))
                    .text_color(gpui::rgb(p.muted))
                    .child("branches"),
            );
        for branch in &self.state.session_view.branches {
            let id = branch.id.clone();
            let label = branch.label.clone();
            section = section.child(
                self.session_button(
                    format!("branch-{id}"),
                    String::new(),
                    cx,
                    move |view, _, cx| {
                        if view.can_change_session() {
                            view.pending_command = true;
                            view.queue(CommandMessage::SwitchBranch(id.clone()));
                            cx.notify();
                        }
                    },
                )
                .debug_selector(|| "branch-row".into())
                .w_full()
                .min_w_0()
                .h(px(40.))
                .overflow_hidden()
                .pl(px(8. + (branch.depth.min(10) * 12) as f32))
                .border_l_2()
                .border_color(gpui::rgb(if branch.current { p.accent } else { p.border }))
                .when(branch.current, |row| {
                    row.bg(gpui::rgb(p.code_chip))
                        .font_weight(gpui::FontWeight::BOLD)
                })
                .child(div().truncate().child(branch.label.clone()))
                .tooltip(move |_, cx| cx.new(|_| LabelTooltip(label.clone(), p)).into()),
            );
        }
        section
    }

    fn close_settings(&mut self, window: &mut Window, cx: &mut Context<Self>) {
        self.state.session_view.settings_open = false;
        window.focus(&self.composer.focus_handle(cx), cx);
        cx.notify();
    }

    fn apply_settings(&mut self, cx: &mut Context<Self>) {
        if self.pending_command {
            return;
        }
        let view = &self.state.session_view;
        if let Some(model) = view.models.get(view.selected_model) {
            let settings = SessionSettings {
                model: model.clone(),
                approval_mode: MODES[view.selected_mode].into(),
            };
            self.pending_command = true;
            self.queue(CommandMessage::SetSettings(settings));
            cx.notify();
        }
    }

    pub(super) fn settings_key(
        &mut self,
        event: &KeyDownEvent,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        match event.keystroke.key.as_str() {
            "escape" => self.close_settings(window, cx),
            "tab" => {
                self.state.session_view.settings_field = 1 - self.state.session_view.settings_field;
            }
            "up" | "down" | "left" | "right" => {
                let view = &mut self.state.session_view;
                let (selected, count) = if view.settings_field == 0 {
                    (&mut view.selected_model, view.models.len())
                } else {
                    (&mut view.selected_mode, MODES.len())
                };
                if count > 0 {
                    let delta = if matches!(event.keystroke.key.as_str(), "up" | "left") {
                        count - 1
                    } else {
                        1
                    };
                    *selected = (*selected + delta) % count;
                }
            }
            "enter" => self.apply_settings(cx),
            _ => {}
        }
        cx.stop_propagation();
        cx.notify();
    }

    pub(super) fn render_settings(&self, cx: &mut Context<Self>) -> impl IntoElement {
        let p = self.appearance.palette();
        let view = &self.state.session_view;
        let mut models = div()
            .id("settings-models")
            .max_h(px(240.))
            .overflow_y_scroll();
        for (index, model) in view.models.iter().enumerate() {
            models = models.child(
                self.session_button(
                    format!("model-{index}"),
                    model.clone(),
                    cx,
                    move |view, window, cx| {
                        view.state.session_view.selected_model = index;
                        view.state.session_view.settings_field = 0;
                        window.focus(&view.focus_handle, cx);
                        cx.notify();
                    },
                )
                .when(index == view.selected_model, |row| {
                    row.bg(gpui::rgb(p.code_chip))
                        .border_l_2()
                        .border_color(gpui::rgb(p.accent))
                }),
            );
        }
        let mut modes = div().flex().gap_2();
        for (index, mode) in MODES.iter().enumerate() {
            modes = modes.child(
                self.session_button(
                    format!("mode-{index}"),
                    (*mode).into(),
                    cx,
                    move |view, window, cx| {
                        view.state.session_view.selected_mode = index;
                        view.state.session_view.settings_field = 1;
                        window.focus(&view.focus_handle, cx);
                        cx.notify();
                    },
                )
                .when(index == view.selected_mode, |row| {
                    row.bg(gpui::rgb(p.code_chip))
                        .border_b_2()
                        .border_color(gpui::rgb(p.accent))
                }),
            );
        }
        div().absolute().inset_0().occlude().bg(gpui::black().opacity(0.7)).flex().items_center().justify_center()
            .child(div().w(px(480.)).max_h_full().p_6().bg(gpui::rgb(p.panel)).flex().flex_col().gap_3()
                .child(div().text_size(px(18.)).child("session settings"))
                .child(div().text_color(gpui::rgb(if view.settings_field == 0 { p.accent } else { p.text })).truncate().child(format!("model: {}", view.models.get(view.selected_model).map(String::as_str).unwrap_or("unavailable"))))
                .child(models)
                .child(div().text_color(gpui::rgb(if view.settings_field == 1 { p.accent } else { p.text })).child("approval mode"))
                .child(modes)
                .child(div().text_size(px(12.)).text_color(gpui::rgb(p.muted)).child("tool rules still apply. tab changes field; arrows select; enter applies; escape closes."))
                .when_some(self.command_error.clone(), |view, error| view.child(div().text_color(gpui::rgb(p.error)).child(error)))
                .child(div().flex().gap_3()
                    .child(self.session_button("apply-settings".into(), if self.pending_command { "applying..." } else { "apply" }.into(), cx, |view, _, cx| view.apply_settings(cx)))
                    .child(self.session_button("close-settings".into(), "close".into(), cx, |view, window, cx| view.close_settings(window, cx)))))
    }
}
