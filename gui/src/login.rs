//! Shared login state for settings, recovery actions, and the first conversation.
use serde::Deserialize;

#[derive(Debug, Clone, Deserialize, PartialEq)]
pub struct LoginProvider {
    pub provider: String,
    pub credentials_present: bool,
    #[serde(skip)]
    pub progress: LoginProgress,
}

#[derive(Debug, Clone, Deserialize)]
pub struct LoginProviders {
    pub providers: Vec<LoginProvider>,
}

#[derive(Debug, Clone, Default, Deserialize, PartialEq)]
#[serde(tag = "state", rename_all = "snake_case")]
pub enum LoginProgress {
    #[default]
    Idle,
    Starting,
    Pending {
        #[serde(default)]
        authorization_url: Option<String>,
    },
    Cancelling,
    Cancelled,
    Succeeded,
    Failed {
        error: LoginError,
    },
}

#[derive(Debug, Clone, Deserialize, PartialEq)]
pub struct LoginError {
    pub code: String,
    pub message: String,
}

impl LoginProgress {
    pub fn busy(&self) -> bool {
        matches!(
            self,
            Self::Starting | Self::Pending { .. } | Self::Cancelling
        )
    }

    pub fn failed(message: String) -> Self {
        Self::Failed {
            error: LoginError {
                code: "login_failed".into(),
                message,
            },
        }
    }
}

impl LoginProvider {
    pub fn label(&self) -> &str {
        match self.provider.as_str() {
            "claude" => "Claude",
            "codex" => "ChatGPT",
            _ => &self.provider,
        }
    }

    pub fn update(&mut self, progress: LoginProgress) -> Option<String> {
        // A queued cancellation must not reopen the browser or enable a restart.
        let url = match &progress {
            LoginProgress::Pending { authorization_url }
                if self.progress != LoginProgress::Cancelling =>
            {
                authorization_url.clone()
            }
            _ => None,
        };
        if progress == LoginProgress::Succeeded {
            self.credentials_present = true;
        }
        if self.progress != LoginProgress::Cancelling || !progress.busy() {
            self.progress = progress;
        }
        url
    }

    pub fn status(&self) -> &'static str {
        match &self.progress {
            LoginProgress::Starting => "Opening browser…",
            LoginProgress::Pending { .. } => "Waiting for browser sign-in…",
            LoginProgress::Cancelling => "Cancelling sign-in…",
            LoginProgress::Cancelled => "Sign-in cancelled",
            LoginProgress::Succeeded => "Logged in. You can send your message again.",
            LoginProgress::Failed { .. } => "Sign-in failed",
            LoginProgress::Idle if self.credentials_present => "Credentials available",
            LoginProgress::Idle => "Not logged in",
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cancellation_ignores_late_browser_url_and_success_enables_retry() {
        let mut provider = LoginProvider {
            provider: "claude".into(),
            credentials_present: false,
            progress: LoginProgress::Cancelling,
        };
        assert_eq!(
            provider.update(LoginProgress::Pending {
                authorization_url: Some("https://authorize.invalid/".into())
            }),
            None
        );
        assert_eq!(provider.progress, LoginProgress::Cancelling);
        provider.update(LoginProgress::Cancelled);
        assert!(!provider.progress.busy());
        provider.progress = LoginProgress::Starting;
        assert_eq!(
            provider.update(LoginProgress::Pending {
                authorization_url: Some("https://authorize.invalid/".into())
            }),
            Some("https://authorize.invalid/".into())
        );
        provider.update(LoginProgress::Succeeded);
        assert!(provider.credentials_present);
        assert!(!provider.progress.busy());
        assert!(provider.status().contains("send your message again"));
    }
}
