pub mod llama_utils;

use crate::{immutable_agent::TaskOutput, LlmConfig};
use anyhow::{anyhow, Result};
use async_openai::types::{CompletionUsage, CreateChatCompletionResponse, Role};
use llama_utils::*;
use log;
use regex::Regex;
use reqwest::{
    header::{HeaderMap, HeaderValue, AUTHORIZATION, CONTENT_TYPE, USER_AGENT},
    ClientBuilder,
};
use serde::{Deserialize, Serialize};
use serde_json::json;
use serde_json::Value;
use std::collections::HashMap;
use std::result::Result as StdResult;
use thiserror::Error;

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
) -> anyhow::Result<CreateChatCompletionResponse>
where
    for<'de> CreateChatCompletionResponse: serde::Deserialize<'de>,
{
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
    response
        .json::<CreateChatCompletionResponse>()
        .await
        .map_err(Into::into)
}

#[derive(Debug, Deserialize, Serialize, Clone, PartialEq)]
pub struct ToolCall {
    pub name: String,
    pub arguments: Option<String>,
}

#[allow(non_snake_case)]
#[derive(Debug, Deserialize, Serialize, Clone, PartialEq)]
pub enum Content {
    Text(String),
    JsonStr(JsonStr),
}

#[allow(non_snake_case)]
#[derive(Debug, Deserialize, Serialize, Clone, PartialEq)]
pub enum JsonStr {
    ToolCall(ToolCall),
    JsonLoad(Value),
}

impl Content {
    pub fn content_to_string(&self) -> String {
        match self {
            Content::Text(text) => text.clone(),
            Content::JsonStr(json_data) => match json_data {
                JsonStr::ToolCall(tool_call) => {
                    format!(
                        "tool_call: {}, arguments: {:?}",
                        tool_call.name, tool_call.arguments
                    )
                }
                JsonStr::JsonLoad(val) => format!("json_load: {}", val),
            },
        }
    }

    pub fn from_str(input: &str) -> Content {
        match serde_json::from_str::<Value>(input) {
            Ok(json_value) => {
                if let Some(name) = json_value.get("name").and_then(|v| v.as_str()) {
                    let arguments = json_value.get("arguments").map(|v| v.to_string());
                    Content::JsonStr(JsonStr::ToolCall(ToolCall {
                        name: name.to_owned(),
                        arguments,
                    }))
                } else {
                    Content::JsonStr(JsonStr::JsonLoad(json_value))
                }
            }
            Err(_) => Content::Text(input.to_string()),
        }
    }
}

#[derive(Debug, Deserialize, Serialize, Clone, PartialEq)]
pub struct LlamaResponseMessage {
    pub content: Content,
    pub role: Role,
    pub usage: CompletionUsage,
}

impl LlamaResponseMessage {
    pub fn content_to_string(&self) -> String {
        self.content.content_to_string()
    }
}

#[derive(Debug, thiserror::Error)]
pub enum LlamaResponseError {
    #[error("No choices available in response")]
    NoChoices,
    #[error("Message content missing in choice")]
    MissingMessageContent,
    #[error("Usage data missing in response")]
    MissingUsageData,
    #[error("Role information missing")]
    MissingRole,
    #[error("JSON extraction from XML failed: {0}")]
    JsonExtractionError(String),
    #[error("JSON pretty print formatting failed: {0}")]
    JsonFormatError(String),
    #[error("Tool call parsing error: {0}")]
    ToolCallParseError(String),
}

pub fn output_response_by_task(
    res_obj: CreateChatCompletionResponse,
    task_type: TaskOutput,
) -> StdResult<LlamaResponseMessage, Box<dyn std::error::Error>> {
    let usage = res_obj.usage.ok_or_else(|| {
        Box::new(LlamaResponseError::MissingUsageData) as Box<dyn std::error::Error>
    })?;
    let choice = res_obj
        .choices
        .first()
        .ok_or_else(|| Box::new(LlamaResponseError::NoChoices) as Box<dyn std::error::Error>)?;
    let msg_obj = &choice.message;

    let role = msg_obj.role.clone();
    let data = msg_obj.content.as_ref().ok_or_else(|| {
        Box::new(LlamaResponseError::MissingMessageContent) as Box<dyn std::error::Error>
    })?;

    let content = match task_type {
        TaskOutput::text => Content::Text(data.to_string()),
        TaskOutput::tool_call => {
            let json_str = extract_json_from_xml_like(data)
                .map_err(|e| Box::new(LlamaResponseError::JsonExtractionError(e.to_string())))?;
            let pretty_json = jsonxf::pretty_print(&json_str)
                .map_err(|e| Box::new(LlamaResponseError::JsonFormatError(e.to_string())))?;
            println!("JSON Output:\n{}\n", pretty_json);

            match extract_tool_call_json(&json_str) {
                Ok(tc) => Content::JsonStr(JsonStr::ToolCall(tc)),
                Err(e) => {
                    log::warn!("Tool call parse error, falling back to text: {}", e);
                    Content::Text(data.to_string())
                }
            }
        }
        TaskOutput::plan => {
            let tasks = parse_planning_tasks(data).map_err(|e| {
                LlamaResponseError::ToolCallParseError(format!(
                    "Failed to parse planning tasks: {}",
                    e
                ))
            })?;

            let planning_data = json!({
                "tasks": tasks
            });

            Content::JsonStr(JsonStr::JsonLoad(planning_data))
        }
    };

    Ok(LlamaResponseMessage {
        content,
        role,
        usage,
    })
}
