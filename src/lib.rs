mod embeddings;
mod error;
mod rerank;

use anyhow::{anyhow, Result};
use reqwest::header::{HeaderMap, HeaderValue, AUTHORIZATION};
use secrecy::{ExposeSecret, SecretString};
use serde::{de::DeserializeOwned, Deserialize, Serialize};

pub use crate::{embeddings::*, error::*, rerank::*};

pub const BASE_URL: &str = "https://api.jina.ai";

/// Represents usage information for the request
#[derive(Debug, Serialize, Deserialize)]
pub struct Usage {
    /// The total number of tokens used for computing the embeddings.
    pub prompt_tokens: i32,
    /// The total number of tokens used for computing the embeddings.
    pub total_tokens: i32,
}

pub struct Jina {
    client: reqwest::Client,
    api_key: SecretString,
    base_url: String,
}

pub struct JinaBuilder {
    api_key: Option<SecretString>,
    base_url: Option<String>,
}

impl Jina {
    pub fn builder() -> JinaBuilder {
        JinaBuilder {
            api_key: None,
            base_url: None,
        }
    }

    pub(crate) async fn post<P, S, D>(&self, path: P, request: S) -> Result<D, JinaError>
    where
        P: Into<String>,
        S: Serialize,
        D: DeserializeOwned,
    {
        let mut headers = HeaderMap::new();
        headers.insert(
            AUTHORIZATION,
            HeaderValue::from_str(&format!("Bearer {}", self.api_key.expose_secret()))
                .expect("couldn't create header value"),
        );

        let response = self
            .client
            .post(format!("{}{}", self.base_url, path.into()))
            .headers(headers)
            .json(&request)
            .send()
            .await?;

        let status = response.status();
        if !status.is_success() {
            let payload = response.json::<HttpErrorPayload>().await.ok();
            return Err(JinaError::HttpError(HttpError {
                status: status.as_u16(),
                payload,
            }));
        }

        let response = response.json::<D>().await?;
        Ok(response)
    }
}

impl JinaBuilder {
    pub fn api_key(mut self, api_key: String) -> Self {
        self.api_key = Some(api_key.into());
        self
    }

    pub fn base_url(mut self, base_url: String) -> Self {
        self.base_url = Some(base_url);
        self
    }

    pub fn build(self) -> Result<Jina> {
        Ok(Jina {
            client: reqwest::Client::new(),
            api_key: self.api_key.or_else(|| std::env::var("EXA_API_KEY").ok().map(SecretString::new))
                .ok_or_else(|| anyhow!("API key is required. Set it explicitly or use the EXA_API_KEY environment variable"))?,
            base_url: self.base_url.unwrap_or_else(|| BASE_URL.to_string()),
        })
    }
}
