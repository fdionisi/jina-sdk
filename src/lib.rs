mod embeddings;
mod error;
mod reader;
mod rerank;

use std::sync::Arc;

use anyhow::{anyhow, Result};
use http_client::{
    http::{
        header::{AUTHORIZATION, CONTENT_TYPE},
        method::Method,
        HeaderMap, HeaderValue,
    },
    AsyncBody, HttpClient, Request, RequestBuilderExt, Response, ResponseAsyncBodyExt,
};
use secrecy::{ExposeSecret, SecretString};
use serde::{de::DeserializeOwned, Deserialize, Serialize};

pub use crate::{embeddings::*, error::*, reader::*, rerank::*};

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
    http_client: Arc<dyn HttpClient>,
    api_key: SecretString,
    pub(crate) base_url: String,
}

pub struct JinaBuilder {
    http_client: Option<Arc<dyn HttpClient>>,
    api_key: Option<SecretString>,
    base_url: Option<String>,
}

impl Jina {
    pub fn builder() -> JinaBuilder {
        JinaBuilder {
            http_client: None,
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
        let headers = self.default_headers();

        let response = self
            .http_client
            .send(
                Request::builder()
                    .uri(format!("{}{}", self.base_url, path.into()))
                    .method(Method::POST)
                    .headers(headers)
                    .json(&request)?,
            )
            .await?;

        Self::handle_response(response).await
    }

    pub(crate) fn default_headers(&self) -> HeaderMap {
        let mut headers = HeaderMap::new();
        headers.insert(
            AUTHORIZATION,
            HeaderValue::from_str(&format!("Bearer {}", self.api_key.expose_secret()))
                .expect("couldn't create header value"),
        );
        headers.insert(
            CONTENT_TYPE,
            "application/json"
                .parse()
                .expect("couldn't create header value"),
        );
        headers
    }

    pub async fn handle_response<D>(response: Response<AsyncBody>) -> Result<D, JinaError>
    where
        D: DeserializeOwned,
    {
        let status = response.status();
        if !status.is_success() {
            let payload = response.json::<HttpErrorPayload>().await.ok();
            return Err(JinaError::HttpError(HttpError {
                status: status.as_u16(),
                payload,
            }));
        }

        let response = response.text().await?;

        Ok(serde_json::from_str(&response).unwrap())
    }
}

impl JinaBuilder {
    pub fn with_http_client(mut self, http_client: Arc<dyn HttpClient>) -> Self {
        self.http_client = Some(http_client);
        self
    }

    pub fn with_api_key<S>(mut self, api_key: S) -> Self
    where
        S: AsRef<str>,
    {
        self.api_key = Some(api_key.as_ref().to_string().into());
        self
    }

    pub fn with_base_url<S>(mut self, base_url: S) -> Self
    where
        S: AsRef<str>,
    {
        self.base_url = Some(base_url.as_ref().into());
        self
    }

    pub fn build(self) -> Result<Jina> {
        Ok(Jina {
            http_client: self.http_client.ok_or_else(|| anyhow!("you must provide an HttpClient implementation"))?,
            api_key: self.api_key.or_else(|| std::env::var("EXA_API_KEY").ok().map(SecretString::new))
                .ok_or_else(|| anyhow!("API key is required. Set it explicitly or use the EXA_API_KEY environment variable"))?,
            base_url: self.base_url.unwrap_or_else(|| BASE_URL.to_string()),
        })
    }
}
