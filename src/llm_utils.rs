use crate::llama_structs::{output_llama_response, LlamaResponseMessage};
use crate::LlmConfig;
use async_openai::types::CreateChatCompletionResponse;
use reqwest::{
    header::{HeaderMap, HeaderValue, AUTHORIZATION, CONTENT_TYPE, USER_AGENT},
    ClientBuilder,
};

pub async fn chat_inner_async_wrapper_text(
    llm_config: &LlmConfig,
    system_prompt: &str,
    input: &str,
    max_token: u16,
) -> anyhow::Result<String> {
    let mut headers = HeaderMap::new();
    let api_key = std::env::var(llm_config.api_key_str)?;
    let bearer_token = format!("Bearer {}", api_key);

    headers.insert(CONTENT_TYPE, HeaderValue::from_static("application/json"));
    headers.insert(USER_AGENT, HeaderValue::from_static("MyClient/1.0.0"));
    headers.insert(AUTHORIZATION, HeaderValue::from_str(&bearer_token)?);

    let input = input.chars().take(24_000).collect::<String>();

    let messages = serde_json::json!([
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": input}
    ]);

    let uri = llm_config.base_url;
    let body = serde_json::to_vec(&serde_json::json!({
        "temperature": 0,
        "max_tokens": max_token,
       "model": llm_config.model,
        "messages": messages,
    }))?;
    //  "model": "Meta-Llama-3-8B-Instruct",

    let client = ClientBuilder::new().default_headers(headers).build()?;
    match client.post(uri).body(body.clone()).send().await {
        Ok(chat) => {
            let response_body = chat.text().await?;
            println!("response_body: {:?}", response_body.clone());
            let raw_output = serde_json::from_str::<CreateChatCompletionResponse>(&response_body)?;

            let out = raw_output.choices[0].message.content.clone().unwrap_or_default();
            Ok(out)
        }
        Err(_e) => {
            println!("Error getting response from OpenAI: {:?}", _e);
            Err(anyhow::anyhow!("Failed to get reply from OpenAI: {:?}", _e))
        }
    }
}
pub async fn chat_inner_async_wrapper(
    llm_config: &LlmConfig,
    system_prompt: &str,
    input: &str,
    max_token: u16,
) -> anyhow::Result<LlamaResponseMessage> {
    let mut headers = HeaderMap::new();
    let api_key = std::env::var(llm_config.api_key_str)?;
    let bearer_token = format!("Bearer {}", api_key);

    headers.insert(CONTENT_TYPE, HeaderValue::from_static("application/json"));
    headers.insert(USER_AGENT, HeaderValue::from_static("MyClient/1.0.0"));
    headers.insert(AUTHORIZATION, HeaderValue::from_str(&bearer_token)?);

    let messages = serde_json::json!([
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": input}
    ]);

    let uri = llm_config.base_url;
    let body = serde_json::to_vec(&serde_json::json!({
        "temperature": 0.3,
        "max_tokens": max_token,
       "model": llm_config.model,
        "messages": messages,
    }))?;
    //  "model": "Meta-Llama-3-8B-Instruct",

    let client = ClientBuilder::new().default_headers(headers).build()?;
    match client.post(uri).body(body.clone()).send().await {
        Ok(chat) => {
            let response_body = chat.text().await?;
            println!("coding response_body: {:?}", response_body.clone());

            let raw_output = serde_json::from_str::<CreateChatCompletionResponse>(&response_body)?;
            if let Some(out) = output_llama_response(raw_output) {
                Ok(out)
            } else {
                Err(anyhow::anyhow!("Empty output in Llama format"))
            }
        }
        Err(_e) => {
            println!("Error getting response from OpenAI: {:?}", _e);
            Err(anyhow::anyhow!("Failed to get reply from OpenAI: {:?}", _e))
        }
    }
}
pub async fn chat_inner_async_llama(
    system_prompt: &str,
    input: &str,
    max_token: u16,
) -> anyhow::Result<LlamaResponseMessage> {
    let mut headers = HeaderMap::new();
    let api_key = std::env::var("TOGETHER_API_KEY")?;
    let bearer_token = format!("Bearer {}", api_key);

    headers.insert(CONTENT_TYPE, HeaderValue::from_static("application/json"));
    headers.insert(USER_AGENT, HeaderValue::from_static("MyClient/1.0.0"));
    headers.insert(AUTHORIZATION, HeaderValue::from_str(&bearer_token)?);

    let messages = serde_json::json!([
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": input}
    ]);

    let uri = "https://api.together.xyz/v1/chat/completions";
    let body = serde_json::to_vec(&serde_json::json!({
        "temperature": 0.7,
        "max_tokens": max_token,
       "model": "meta-llama/Llama-3-70b-chat-hf",
        "messages": messages,
    }))?;
    //  "model": "Meta-Llama-3-8B-Instruct",

    let client = ClientBuilder::new().default_headers(headers).build()?;
    match client.post(uri).body(body.clone()).send().await {
        Ok(chat) => {
            let response_body = chat.text().await?;
            let raw_output = serde_json::from_str::<CreateChatCompletionResponse>(&response_body)?;
            if let Some(out) = output_llama_response(raw_output) {
                Ok(out)
            } else {
                Err(anyhow::anyhow!("Empty output in Llama format"))
            }
        }
        Err(_e) => {
            println!("Error getting response from OpenAI: {:?}", _e);
            Err(anyhow::anyhow!("Failed to get reply from OpenAI: {:?}", _e))
        }
    }
}
