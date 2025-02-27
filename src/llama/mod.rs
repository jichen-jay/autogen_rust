pub mod llama_utils;

use crate::LlmConfig;
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
use serde_json::{from_str, json, Value};
use std::collections::HashMap;
use std::result::Result as StdResult;
use thiserror::Error;

#[derive(Debug, Deserialize, Serialize, Clone, PartialEq)]
pub struct ToolCall {
    pub name: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub arguments: Option<String>,
}

#[derive(Error, Debug)]
pub enum ParseError {
    #[error("Regex creation failed: {0}")]
    RegexError(#[from] regex::Error),
    #[error("Failed to find required pattern in input")]
    CaptureError,
    #[error("JSON parsing failed: {0}")]
    JsonError(#[from] serde_json::Error),
    #[error("XML structure invalid")]
    XmlParseError,
}

#[derive(Debug, Deserialize, Serialize, Clone, PartialEq)]
pub enum Content {
    Text(String),
    Structured(StructuredText),
}

#[derive(Debug, Deserialize, Serialize, Clone, PartialEq)]
pub enum StructuredText {
    ToolCall(ToolCall),
    Tasks(Vec<Task>),
    Expanded(String),
}

impl StructuredText {
    fn try_parse<T: serde::de::DeserializeOwned>(inp: Value) -> Result<T, serde_json::Error> {
        serde_json::from_value(inp)
    }

    pub fn from_str(input: &str, desired_variant: StructuredText) -> StdResult<Self, ParseError> {
        match desired_variant {
            StructuredText::ToolCall(_) => {
                if let Ok(tc) = Self::parse_tool_call(input) {
                    return Ok(StructuredText::ToolCall(tc));
                }
            }
            StructuredText::Tasks(_) => {
                if let Ok(tasks) = Self::parse_tasks(input) {
                    return Ok(StructuredText::Tasks(tasks));
                }
            }
            StructuredText::Expanded(_) => {
                return Ok(StructuredText::Expanded(input.to_string()));
            }
        }

        if let Ok(tool_call) = Self::parse_tool_call(input) {
            return Ok(StructuredText::ToolCall(tool_call));
        }
        if let Ok(tasks) = Self::parse_tasks(input) {
            return Ok(StructuredText::Tasks(tasks));
        }

        Ok(StructuredText::Expanded(input.to_string()))
    }

    pub fn parse_tool_call(input: &str) -> Result<ToolCall, ExtractError> {
        extract_tool_call_json(input)
    }

    pub fn parse_tasks(input: &str) -> Result<Vec<Task>, ParseError> {
        parse_planning_tasks(input)
    }
}

#[derive(Debug, Deserialize, Serialize, Clone, PartialEq)]
pub struct Task {
    pub name: String,
    pub description: String,
    #[serde(default)]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool: Option<String>,
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

#[derive(Error, Debug)]
pub enum ChatInnerError {
    #[error("API key retrieval failed: {0}")]
    MissingApiKey(#[from] std::env::VarError),
    #[error("Invalid authorization header: {0}")]
    InvalidAuthHeader(#[from] reqwest::header::InvalidHeaderValue),
    #[error("JSON serialization failed: {0}")]
    SerializationError(#[from] serde_json::Error),
    #[error("HTTP client build failed: {0}")]
    ClientBuildError(#[from] reqwest::Error),
    // #[error("HTTP request failed: {0}")]
    // RequestError(#[from] reqwest::Error),
    #[error("Empty choices in response")]
    EmptyChoices,
    #[error("Missing message content")]
    MissingMessageContent,
    #[error("LLama response processing failed: {0}")]
    LlamaResponseProcessingError(String),
}

pub async fn chat_inner_async_wrapper(
    llm_config: &LlmConfig,
    system_prompt: &str,
    input: &str,
    max_token: u16,
) -> Result<(String, CompletionUsage), ChatInnerError> {
    let api_key = std::env::var(&llm_config.api_key_str)?;
    let bearer_token = format!("Bearer {}", api_key);

    let mut headers = HeaderMap::new();
    headers.insert(CONTENT_TYPE, HeaderValue::from_static("application/json"));
    headers.insert(USER_AGENT, HeaderValue::from_static("MyClient/1.0.0"));
    headers.insert(AUTHORIZATION, HeaderValue::from_str(&bearer_token)?);

    let messages = json!([
        { "role": "system", "content": system_prompt },
        { "role": "user", "content": input }
    ]);
    let payload = json!({
        "temperature": 0.3,
        "max_tokens": max_token,
        "model": llm_config.model,
        "messages": messages,
    });

    let body = serde_json::to_vec(&payload)?;
    let client = ClientBuilder::new().default_headers(headers).build()?;
    let response = client.post(llm_config.base_url).body(body).send().await?;

    let chat_response = response.json::<CreateChatCompletionResponse>().await?;
    let usage = chat_response.usage.unwrap_or_else(|| CompletionUsage {
        prompt_tokens: 0,
        completion_tokens: 0,
        total_tokens: 0,
    });

    let choice = chat_response
        .choices
        .first()
        .ok_or(ChatInnerError::EmptyChoices)?;
    let data = choice
        .message
        .content
        .as_deref()
        .ok_or(ChatInnerError::MissingMessageContent)?;

    Ok((data.to_owned(), usage))
}

impl Content {
    pub fn content_to_string(&self) -> String {
        match self {
            Content::Text(text) => text.clone(),
            Content::Structured(structured) => match structured {
                StructuredText::ToolCall(tc) => {
                    format!("ToolCall: {} ({:?})", tc.name, tc.arguments)
                }
                StructuredText::Tasks(tasks) => tasks
                    .iter()
                    .map(|t| format!("[{}] {}", t.name, t.description))
                    .collect::<Vec<_>>()
                    .join("\n"),
                StructuredText::Expanded(tex) => tex.clone(),
            },
        }
    }
}
