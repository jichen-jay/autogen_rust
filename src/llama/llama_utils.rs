use crate::llama::{
    output_response_by_task, JsonStr, LlamaResponseError, LlamaResponseMessage, ToolCall,
};
use crate::LlmConfig;
use anyhow::Context;
use reqwest::{
    header::{HeaderMap, HeaderValue, AUTHORIZATION, CONTENT_TYPE, USER_AGENT},
    ClientBuilder,
};

use anyhow::{anyhow, Result};
use async_openai::types::{CompletionUsage, CreateChatCompletionResponse, Role};
use log;
use regex::Regex;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::HashMap;
use std::result::Result as StdResult;
use thiserror::Error;

pub fn extract_json_from_xml_like(
    xml_like_data: &str,
) -> StdResult<String, impl std::error::Error> {
    #[derive(Error, Debug)]
    #[error("Invalid XML format: missing or mismatched tool_call tags")]
    struct XmlParseError;

    let start_tag = "<tool_call>";
    let end_tag = "</tool_call>";

    let trimmed = xml_like_data.trim();
    println!("trimmed xml-like-data:\n{}", trimmed.clone());
    if trimmed.starts_with(start_tag) && trimmed.ends_with(end_tag) {
        let start_pos = start_tag.len();
        let end_pos = trimmed.len() - end_tag.len();
        Ok(trimmed[start_pos..end_pos].trim().to_string())
    } else {
        Err(XmlParseError)
    }
}

pub fn extract_tool_call_json(input: &str) -> StdResult<ToolCall, Box<dyn std::error::Error>> {
    #[derive(thiserror::Error, Debug)]
    enum ExtractError {
        #[error("JSON repair failed: {0}")]
        RepairError(#[from] Box<dyn std::error::Error>),
        #[error("Regex creation failed: {0}")]
        RegexError(#[from] regex::Error),
        #[error("JSON parsing failed: {0}")]
        JsonError(#[from] serde_json::Error),
        #[error("No arguments found in tool call")]
        NoArguments,
        #[error("No name found in tool call")]
        NoName,
        #[error("Invalid input: neither valid JSON nor matching tool-call pattern")]
        InvalidInput,
    }

    let repaired_input =
        repair_json_with_tool(input).map_err(|e| ExtractError::RepairError(Box::new(e)))?;
    let parsed = serde_json::from_str::<serde_json::Value>(&repaired_input)?;

    let name = parsed
        .get("name")
        .and_then(|v| v.as_str())
        .ok_or(ExtractError::NoName)?
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

    let re = Regex::new(
        r#"(?i)\{"arguments":\s*(?P<arguments>\{.*\}),\s*"name":\s*"(?P<name>[^"]+)"\}"#,
    )?;

    if let Some(caps) = re.captures(input) {
        return Ok(ToolCall {
            name: caps
                .name("name")
                .ok_or(ExtractError::NoName)?
                .as_str()
                .to_string(),
            arguments: Some(
                caps.name("arguments")
                    .ok_or(ExtractError::NoArguments)?
                    .as_str()
                    .to_string(),
            ),
        });
    }

    Err(ExtractError::InvalidInput.into())
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

    let res = String::from_utf8(output.stdout)?;
    println!("repair output: {:?}", res);
    Ok(res)
}

pub fn parse_next_move_and_(
    input: &str,
    next_marker: Option<&str>,
) -> (bool, Option<String>, Vec<String>) {
    let json_regex = Regex::new(r"\{[^}]*\}").unwrap();
    let json_str = json_regex
        .captures(input)
        .and_then(|cap| cap.get(0))
        .map_or(String::new(), |m| m.as_str().to_string());

    let continue_or_terminate_regex =
        Regex::new(r#""continue_or_terminate":\s*"([^"]*)""#).unwrap();
    let continue_or_terminate = continue_or_terminate_regex
        .captures(&json_str)
        .and_then(|cap| cap.get(1))
        .map_or(String::new(), |m| m.as_str().to_string());

    let next_move = match next_marker {
        Some(marker) => {
            let next_marker_regex = Regex::new(&format!(r#""{}":\s*"([^"]*)""#, marker)).unwrap();
            Some(
                next_marker_regex
                    .captures(&json_str)
                    .and_then(|cap| cap.get(1))
                    .map_or(String::new(), |m| m.as_str().to_string()),
            )
        }
        None => None,
    };

    let key_points_array_regex = Regex::new(r#""key_points":\s*\[(.*?)\]"#).unwrap();

    let key_points_array = key_points_array_regex
        .captures(&json_str)
        .and_then(|cap| cap.get(1))
        .map_or(String::new(), |m| m.as_str().to_string());

    let key_points: Vec<String> = if !key_points_array.is_empty() {
        key_points_array
            .split(',')
            .map(|s| s.trim().trim_matches('"').to_string())
            .collect()
    } else {
        vec![]
    };

    (&continue_or_terminate == "TERMINATE", next_move, key_points)
}

#[derive(Debug, Deserialize, Serialize, Clone, PartialEq)]
pub struct PlanningTask {
    pub id: String,
    pub name: String,
    pub description: String,
    #[serde(default)]
    pub tool: Option<String>,
}

pub fn parse_planning_tasks(input: &str) -> Result<Vec<PlanningTask>> {
    let tasks_regex =
        Regex::new(r#""tasks":\s*(\[[^\]]*\])"#).context("Failed to create tasks regex")?;

    let tasks_str = tasks_regex
        .captures(input)
        .and_then(|cap| cap.get(1))
        .map(|m| m.as_str())
        .ok_or_else(|| anyhow!("Failed to find tasks array in input"))?;

    let parsed_tasks: Vec<PlanningTask> =
        serde_json::from_str(tasks_str).context("Failed to parse tasks JSON")?;

    Ok(parsed_tasks)
}
