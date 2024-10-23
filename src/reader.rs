use anyhow::{anyhow, Result};
use http_client::{
    http::{header::ACCEPT, HeaderValue, Method},
    Request, RequestBuilderExt,
};
use serde::{Deserialize, Serialize};
use serde_json::json;

use crate::Jina;

#[derive(Serialize, Deserialize)]
pub struct ReaderUsage {
    pub tokens: i64,
}

#[derive(Serialize, Deserialize)]
pub struct ReaderData {
    pub content: String,
    pub description: String,
    pub title: String,
    pub url: String,
    pub usage: ReaderUsage,
}

#[derive(Serialize, Deserialize)]
pub struct ReaderResponse {
    pub code: i64,
    pub data: ReaderData,
    pub status: i64,
}

#[derive(Serialize, Deserialize)]
pub struct ReaderRequest {
    pub url: String,
    pub return_format: Option<ReaderReturnFormat>,
    pub no_cache: Option<bool>,
    pub wait_for_selector: Option<String>,
    pub target_selector: Option<String>,
    pub timeout: Option<u16>,
    pub proxy_url: Option<String>,
    pub locale: Option<String>,
}

#[derive(Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ReaderReturnFormat {
    Default,
    Markdown,
    Html,
    Text,
    Screenshot,
    Pageshot,
}

impl TryFrom<String> for ReaderReturnFormat {
    type Error = anyhow::Error;

    fn try_from(value: String) -> Result<Self, Self::Error> {
        match value.to_lowercase().as_str() {
            "default" => Ok(ReaderReturnFormat::Default),
            "markdown" => Ok(ReaderReturnFormat::Markdown),
            "html" => Ok(ReaderReturnFormat::Html),
            "text" => Ok(ReaderReturnFormat::Text),
            "screenshot" => Ok(ReaderReturnFormat::Screenshot),
            "pageshot" => Ok(ReaderReturnFormat::Pageshot),
            _ => Err(anyhow!("Invalid ReaderReturnFormat: {}", value)),
        }
    }
}

impl ReaderReturnFormat {
    pub fn as_str(&self) -> &'static str {
        match self {
            ReaderReturnFormat::Default => "default",
            ReaderReturnFormat::Markdown => "markdown",
            ReaderReturnFormat::Html => "html",
            ReaderReturnFormat::Text => "text",
            ReaderReturnFormat::Screenshot => "screenshot",
            ReaderReturnFormat::Pageshot => "pageshot",
        }
    }
}

impl Jina {
    pub async fn reader(&self, request: ReaderRequest) -> Result<ReaderResponse> {
        let mut headers = self.default_headers();
        headers.insert(ACCEPT, HeaderValue::from_str("application/json")?);

        if let Some(ref return_format) = request.return_format {
            headers.insert("X-Return-Format", return_format.as_str().parse()?);
        }

        if let Some(ref target_selector) = request.target_selector {
            headers.insert("X-Target-Selector", target_selector.parse()?);
        }

        if let Some(ref locale) = request.locale {
            headers.insert("X-Locale", locale.parse()?);
        }

        if let Some(ref proxy_url) = request.proxy_url {
            headers.insert("X-Proxy-Url", proxy_url.to_string().parse()?);
        }

        if let Some(ref timeout) = request.timeout {
            headers.insert("X-Timeout", timeout.to_string().parse()?);
        }

        if let Some(ref no_cache) = request.no_cache {
            headers.insert("X-No-Cache", no_cache.to_string().parse()?);
        }

        if let Some(ref wait_for_selector) = request.wait_for_selector {
            headers.insert("X-Wait-For-Selector", wait_for_selector.parse()?);
        }

        let response = self
            .http_client
            .send(
                Request::builder()
                    .uri(self.base_url.clone())
                    .method(Method::POST)
                    .headers(headers)
                    .json(json! ({
                        "url": request.url
                    }))?,
            )
            .await?;

        Ok(Self::handle_response(response).await?)
    }
}
