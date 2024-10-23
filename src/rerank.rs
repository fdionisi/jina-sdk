use serde::{Deserialize, Serialize};

use crate::embeddings::TextDoc;
use crate::{Jina, JinaError, Usage};

/// The identifier of the model.
///
/// Available models and corresponding param size and dimension:
/// - `jina-reranker-v2-base-multilingual`,	278M
/// - `jina-reranker-v1-base-en`,	137M
/// - `jina-reranker-v1-tiny-en`,	33M
/// - `jina-reranker-v1-turbo-en`,	38M
/// - `jina-colbert-v1-en`,	137M
#[derive(Debug, Serialize)]
pub enum RerankerModel {
    #[serde(rename = "jina-reranker-v2-base-multilingual")]
    RerankerV2BaseMultilingual,
    #[serde(rename = "jina-reranker-v1-base-en")]
    RerankerV1BaseEn,
    #[serde(rename = "jina-reranker-v1-tiny-en")]
    RerankerV1TinyEn,
    #[serde(rename = "jina-reranker-v1-turbo-en")]
    RerankerV1TurboEn,
    #[serde(rename = "jina-colbert-v1-en")]
    ColbertV1En,
}

#[derive(Debug, Serialize)]
pub struct RerankRequest {
    pub model: RerankerModel,
    /// The search query
    #[serde(flatten)]
    pub query: QueryType,
    /// A list of text documents or strings to rerank. If a document is provided the text fields is required and all other fields will be preserved in the response.
    #[serde(flatten)]
    pub documents: DocumentType,
    /// The number of most relevant documents or indices to return, defaults to the length of `documents`
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_n: Option<usize>,
    /// If false, returns results without the doc text - the api will return a list of {index, relevance score} where index is inferred from the list passed into the request. If true, returns results with the doc text passed in - the api will return an ordered list of {index, text, relevance score} where index + text refers to the list passed into the request. Defaults to true
    #[serde(skip_serializing_if = "Option::is_none")]
    pub return_documents: Option<bool>,
}

#[derive(Debug, Serialize)]
#[serde(untagged)]
pub enum QueryType {
    String(String),
    TextDoc(TextDoc),
}

#[derive(Debug, Serialize)]
#[serde(untagged)]
pub enum DocumentType {
    Strings(Vec<String>),
    TextDocs(Vec<TextDoc>),
}

#[derive(Debug, Deserialize)]
pub struct RerankResponse {
    pub model: String,
    pub results: Vec<RankedResult>,
    pub usage: Usage,
}

#[derive(Debug, Deserialize)]
pub struct RankedResult {
    pub index: usize,
    pub document: RankedDocument,
    pub relevance_score: f32,
}

#[derive(Debug, Deserialize)]
pub struct RankedDocument {
    pub text: String,
}

impl Jina {
    pub async fn rerank(&self, request: RerankRequest) -> Result<RerankResponse, JinaError> {
        self.post("/v1/rerank", request).await
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use http_client_reqwest::HttpClientReqwest;

    use super::*;

    #[tokio::test]
    async fn test_rerank() {
        let http_client = Arc::new(HttpClientReqwest::default());
        let mut server = mockito::Server::new();
        let mock = server
            .mock("POST", "/v1/rerank")
            .with_status(200)
            .with_header("content-type", "application/json")
            .with_body(
                r#"
                {
                    "model": "test-model",
                    "results": [
                        {
                            "index": 0,
                            "document": {
                                "text": "Relevant document"
                            },
                            "relevance_score": 0.9
                        }
                    ],
                    "usage": {
                        "total_tokens": 5,
                        "prompt_tokens": 5
                    }
                }
            "#,
            )
            .create();

        let client = Jina::builder()
            .with_http_client(http_client)
            .with_api_key("test-key".to_string())
            .with_base_url(server.url())
            .build()
            .unwrap();

        let request = RerankRequest {
            model: RerankerModel::ColbertV1En,
            query: QueryType::String("Test query".to_string()),
            documents: DocumentType::Strings(vec!["Relevant document".to_string()]),
            top_n: None,
            return_documents: None,
        };

        let response = client.rerank(request).await.unwrap();

        assert_eq!(response.model, "test-model");
        assert_eq!(response.results.len(), 1);
        assert_eq!(response.results[0].index, 0);
        assert_eq!(response.results[0].document.text, "Relevant document");
        assert_eq!(response.results[0].relevance_score, 0.9);
        assert_eq!(response.usage.total_tokens, 5);

        mock.assert();
    }
}
