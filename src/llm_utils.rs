use crate::llama_structs::{output_llama_response, LlamaResponseError, LlamaResponseMessage};
use crate::LlmConfig;
use anyhow::Context;
use async_openai::types::CreateChatCompletionResponse;
use reqwest::{
    header::{HeaderMap, HeaderValue, AUTHORIZATION, CONTENT_TYPE, USER_AGENT},
    ClientBuilder,
};

#[derive(thiserror::Error, Debug)]
pub enum ChatInnerError {
    #[error("API key retrieval failed: {0}")]
    MissingApiKey(#[from] std::env::VarError),

    #[error("Invalid authorization header: {0}")]
    InvalidAuthHeader(#[from] reqwest::header::InvalidHeaderValue),

    #[error("JSON serialization failed: {0}")]
    SerializationError(#[from] serde_json::Error),

    #[error("HTTP client build failed: {0}")]
    ClientBuildError(#[from] reqwest::Error),

    #[error("HTTP request failed: {0}")]
    RequestError(reqwest::Error),

    #[error("Response text extraction failed: {0}")]
    ResponseTextError(reqwest::Error),

    #[error("Pretty printing JSON failed: {0}")]
    PrettyPrintError(String),

    #[error("LLama response processing failed: {0}")]
    LlamaResponseProcessingError(anyhow::Error),
}

pub async fn chat_inner_async_wrapper(
    llm_config: &LlmConfig,
    system_prompt: &str,
    input: &str,
    max_token: u16,
) -> anyhow::Result<LlamaResponseMessage> {
    let api_key = std::env::var(&llm_config.api_key_str)?;
    let bearer_token = format!("Bearer {}", api_key);

    let mut headers = HeaderMap::new();
    headers.insert(CONTENT_TYPE, HeaderValue::from_static("application/json"));
    headers.insert(USER_AGENT, HeaderValue::from_static("MyClient/1.0.0"));
    headers.insert(AUTHORIZATION, HeaderValue::from_str(&bearer_token)?);

    let messages = serde_json::json!([
        { "role": "system", "content": system_prompt },
        { "role": "user", "content": input }
    ]);
    let payload = serde_json::json!({
        "temperature": 0.3,
        "max_tokens": max_token,
        "model": llm_config.model,
        "messages": messages,
    });

    let body = serde_json::to_vec(&payload)?;

    let client = ClientBuilder::new().default_headers(headers).build()?;

    let response = client.post(llm_config.base_url).body(body).send().await?;

    let response_body: CreateChatCompletionResponse = response.json().await?;

    // let pretty_json =
    //     jsonxf::pretty_print(&response_body).map_err(ChatInnerError::PrettyPrintError)?;
    // println!("raw response_body: {}", pretty_json);

    match output_llama_response(response_body) {
        Ok(res) => Ok(res),
        Err(e) => Err(anyhow::anyhow!(e.to_string())),
    }
}
