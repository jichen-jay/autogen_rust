use crate::llama::{
    LlamaResponseError, LlamaResponseMessage, ParseError, StructuredText, Task, ToolCall,
};
use crate::LlmConfig;
use anyhow::Context;
use anyhow::{anyhow, Result};
use async_openai::types::{CompletionUsage, CreateChatCompletionResponse, Role};
use log;
use regex::Regex;
use reqwest::{
    header::{HeaderMap, HeaderValue, AUTHORIZATION, CONTENT_TYPE, USER_AGENT},
    ClientBuilder,
};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::borrow::Cow;
use std::collections::HashMap;
use std::result::Result as StdResult;
use thiserror::Error;

pub fn extract_json_from_xml_like(xml_like_data: &str) -> StdResult<String, ParseError> {
    let start_tag = "<tool_call>";
    let end_tag = "</tool_call>";

    let trimmed = xml_like_data.trim();
    println!("trimmed xml-like-data:\n{}", trimmed.clone());
    if trimmed.starts_with(start_tag) && trimmed.ends_with(end_tag) {
        let start_pos = start_tag.len();
        let end_pos = trimmed.len() - end_tag.len();
        Ok(trimmed[start_pos..end_pos].trim().to_string())
    } else {
        Err(ParseError::XmlParseError)
    }
}

#[derive(Error, Debug)]
pub enum ExtractError {
    #[error("Invalid XML format")]
    XmlParseError,
    #[error("JSON repair failed: {0}")]
    RepairError(String),
    #[error("Regex error: {0}")]
    RegexError(#[from] regex::Error),
    #[error("JSON error: {0}")]
    JsonError(#[from] serde_json::Error),
    #[error("Missing required field: {0}")]
    MissingField(&'static str),
    #[error("Invalid string input: {0}")]
    InvalidInput(String),
}

//is it better to keep it as a free standing function or keep it within the trait of StructuredText
pub fn extract_tool_call_json(input: &str) -> StdResult<ToolCall, ExtractError> {
    let repaired_input =
        repair_json_with_tool(input).map_err(|e| ExtractError::RepairError(e.to_string()))?;
    let parsed = serde_json::from_str::<serde_json::Value>(&repaired_input)?;

    let name = parsed
        .get("name")
        .and_then(|v| v.as_str())
        .ok_or(ExtractError::MissingField("missing function name"))?
        .to_string();

    let arguments = parsed.get("arguments").and_then(|args| {
        let re_args = Regex::new(r#"(?s)"arguments"\s*:\s*(?P<args>\{.*?\})"#).ok()?;
        re_args
            .captures(&repaired_input)
            .and_then(|cap| cap.name("args"))
            .map(|m| m.as_str().to_string())
    });

    if !name.is_empty() {
        return Ok(ToolCall { name, arguments });
    }

    Err(ExtractError::InvalidInput(input.to_string()))
}

pub fn repair_json_with_tool(input: &str) -> StdResult<String, impl std::error::Error> {
    #[derive(Debug, Error)]
    enum RepairError {
        #[error("Repair tool failed with error: {0}")]
        RepairToolError(String),
        #[error("Invalid UTF-8 in repair tool output: {0}")]
        Utf8Error(#[from] std::string::FromUtf8Error),
        #[error("IO error: {0}")]
        Io(#[from] std::io::Error),
    }

    let output = std::process::Command::new("./jsonrepair")
        .args(&["-i", input])
        .output()?;

    if !output.status.success() {
        return Err(RepairError::RepairToolError(
            String::from_utf8_lossy(&output.stderr).to_string(),
        ));
    }

    String::from_utf8(output.stdout).map_err(|e| RepairError::Utf8Error(e))
}

#[derive(Debug, Deserialize, Serialize, Clone, PartialEq)]
pub struct PlanningTask {
    pub name: String,
    pub description: String,
    #[serde(default)]
    pub tool: Option<String>,
}

pub fn parse_planning_tasks(input: &str) -> StdResult<Vec<Task>, ParseError> {
    let tasks_regex =
        Regex::new(r#""tasks":\s*(\[[^\]]*\])"#).map_err(|e| ParseError::RegexError(e));

    let tasks_str = tasks_regex
        .unwrap()
        .captures(input)
        .and_then(|cap| cap.get(1))
        .map(|m| m.as_str())
        .ok_or(ParseError::CaptureError)?;

    let tasks: Vec<Task> = serde_json::from_str(tasks_str).map_err(|e| ParseError::JsonError(e))?;

    Ok(tasks)
}
