use serde::{Deserialize, Serialize};

use crate::{Jina, JinaError, Usage};

#[derive(Debug, Serialize, Deserialize)]
pub enum EmbeddingsModel {
    #[serde(rename = "jina-clip-v1")]
    ClipV1,
    #[serde(rename = "jina-embeddings-v2-base-en")]
    EmbeddingsV2BaseEn,
    #[serde(rename = "jina-embeddings-v2-base-es")]
    EmbeddingsV2BaseEs,
    #[serde(rename = "jina-embeddings-v2-base-de")]
    EmbeddingsV2BaseDe,
    #[serde(rename = "jina-embeddings-v2-base-zh")]
    EmbeddingsV2BaseZh,
    #[serde(rename = "jina-embeddings-v2-base-code")]
    EmbeddingsV2BaseCode,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct EmbeddingsRequest {
    /// The identifier of the model.
    ///
    /// Available models and corresponding param size and dimension:
    /// - `jina-clip-v1`,	223M,	768
    /// - `jina-embeddings-v2-base-en`,	137M,	768
    /// - `jina-embeddings-v2-base-es`,	161M,	768
    /// - `jina-embeddings-v2-base-de`,	161M,	768
    /// - `jina-embeddings-v2-base-zh`,	161M,	768
    /// - `jina-embeddings-v2-base-code`,	137M,	768
    ///
    /// For more information, please checkout our [technical blog](https://arxiv.org/abs/2307.11224).
    pub model: EmbeddingsModel,

    /// List of texts to embed
    #[serde(flatten)]
    pub input: EmbeddingsInput,

    /// The format in which you want the embeddings to be returned.
    /// Possible value are `float`, `base64`, `binary`, `ubinary` or a list containing any of them.
    /// Defaults to `float`
    #[serde(skip_serializing_if = "Option::is_none")]
    pub embedding_type: Option<EmbeddingType>,

    /// Flag to determine if the embeddings should be normalized to have a unit L2 norm
    #[serde(skip_serializing_if = "Option::is_none")]
    pub normalized: Option<bool>,
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(untagged)]
pub enum EmbeddingsInput {
    StringArray(Vec<String>),
    String(String),
    DocArray(Vec<Doc>),
    Doc(Doc),
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(untagged)]
pub enum Doc {
    Text(TextDoc),
    Image(ImageDoc),
}

#[derive(Debug, Serialize, Deserialize)]
pub struct ImageDoc {
    pub image: String,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct TextDoc {
    pub text: String,
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(untagged)]
pub enum EmbeddingType {
    Single(EmbeddingTypeEnum),
    Multiple(Vec<EmbeddingTypeEnum>),
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum EmbeddingTypeEnum {
    Float,
    Base64,
    Binary,
    Ubinary,
}

#[derive(Debug, Deserialize)]
pub struct EmbeddingsResponse {
    pub model: String,
    pub data: Vec<Embedding>,
    pub usage: Usage,
}

#[derive(Debug, Deserialize)]
pub struct Embedding {
    pub index: usize,
    pub embedding: Vec<f32>,
    pub object: String,
}

impl Jina {
    pub async fn embeddings(
        &self,
        request: EmbeddingsRequest,
    ) -> Result<EmbeddingsResponse, JinaError> {
        self.post("/v1/embeddings", request).await
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_embeddings() {
        let mut server = mockito::Server::new();
        let mock = server
            .mock("POST", "/v1/embeddings")
            .with_status(200)
            .with_header("content-type", "application/json")
            .with_body(
                r#"
                {
                    "model": "test-model",
                    "data": [
                        {
                            "index": 0,
                            "embedding": [0.1, 0.2, 0.3],
                            "object": "embedding"
                        }
                    ],
                    "usage": {
                        "total_tokens": 3,
                        "prompt_tokens": 3
                    }
                }
            "#,
            )
            .create();

        let client = Jina::builder()
            .api_key("test-key".to_string())
            .base_url(server.url())
            .build()
            .unwrap();

        let request = EmbeddingsRequest {
            model: EmbeddingsModel::ClipV1,
            input: EmbeddingsInput::String("Hello, world!".to_string()),
            embedding_type: None,
            normalized: None,
        };

        let response = client.embeddings(request).await.unwrap();

        assert_eq!(response.model, "test-model");
        assert_eq!(response.data.len(), 1);
        assert_eq!(response.data[0].index, 0);
        assert_eq!(response.data[0].embedding, vec![0.1, 0.2, 0.3]);
        assert_eq!(response.usage.total_tokens, 3);

        mock.assert();
    }
}
