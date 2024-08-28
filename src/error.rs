use std::fmt;

#[derive(Debug, thiserror::Error)]
pub enum JinaError {
    #[error("Client error: {0}")]
    ClientError(#[from] reqwest::Error),
    #[error("HTTP error: {0}")]
    HttpError(HttpError),
}

#[derive(Debug, serde::Deserialize, serde::Serialize, thiserror::Error)]
pub struct HttpError {
    pub status: u16,
    pub payload: Option<HttpErrorPayload>,
}

#[derive(Debug, serde::Deserialize, serde::Serialize)]
pub struct HttpErrorPayload {
    pub detail: String,
}

impl fmt::Display for HttpError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        if let Some(payload) = self.payload.as_ref() {
            write!(f, "{} - {}", self.status, payload.detail)
        } else {
            write!(f, "{}", self.status)
        }
    }
}
