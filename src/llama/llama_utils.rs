use crate::llama::{
    output_llama_response, JsonStr, LlamaResponseError, LlamaResponseMessage, ToolCall,
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
    if trimmed.starts_with(start_tag) && trimmed.ends_with(end_tag) {
        let start_pos = start_tag.len();
        let end_pos = trimmed.len() - end_tag.len();
        Ok(trimmed[start_pos..end_pos].trim().to_string())
    } else {
        Err(XmlParseError)
    }
}

pub fn extract_tool_call_json(input: &str) -> StdResult<JsonStr, Box<dyn std::error::Error>> {
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

    if let Some(name_val) = parsed.get("name") {
        if name_val.is_string() {
            let name = name_val.as_str().unwrap().to_string();
            // Optionally attempt to extract arguments from the tool call.
            let arguments_slice = if parsed.get("arguments").is_some() {
                let re_args = Regex::new(r#"(?s)"arguments"\s*:\s*(?P<args>\{.*?\})"#)?;
                if let Some(cap_args) = re_args.captures(&repaired_input) {
                    cap_args
                        .name("args")
                        .ok_or(ExtractError::NoArguments)?
                        .as_str()
                        .to_string()
                        .into()
                } else {
                    None
                }
            } else {
                None
            };

            return Ok(JsonStr::ToolCall(ToolCall {
                name,
                arguments: arguments_slice,
            }));
        }
    } else {
        return Ok(JsonStr::JsonLoad(parsed));
    }

    let re = Regex::new(
        r#"(?i)\{"arguments":\s*(?P<arguments>\{.*\}),\s*"name":\s*"(?P<name>[^"]+)"\}"#,
    )?;
    if let Some(caps) = re.captures(input) {
        let arguments_str = caps
            .name("arguments")
            .ok_or(ExtractError::NoArguments)?
            .as_str()
            .to_string();
        let name = caps
            .name("name")
            .ok_or(ExtractError::NoName)?
            .as_str()
            .to_string();
        return Ok(JsonStr::ToolCall(ToolCall {
            name,
            arguments: Some(arguments_str),
        }));
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

pub fn parse_planning_sub_tasks(input: &str) -> (Vec<String>, String, String) {
    let sub_tasks_regex = Regex::new(r#""sub_tasks":\s*(\[[^\]]*\])"#).unwrap();
    let sub_tasks_str = sub_tasks_regex
        .captures(input)
        .and_then(|cap| cap.get(1))
        .map_or(String::new(), |m| m.as_str().to_string());

    let solution_found_regex = Regex::new(r#""solution_found":\s*"([^"]*)""#).unwrap();
    let solution_found = solution_found_regex
        .captures(input)
        .and_then(|cap| cap.get(1))
        .map_or(String::new(), |m| m.as_str().to_string());

    if sub_tasks_str.is_empty() {
        eprintln!("Failed to extract 'sub_tasks' from input.");
        return (vec![], input.to_string(), solution_found);
    }
    let task_summary_regex = Regex::new(r#""task_summary":\s*"([^"]*)""#).unwrap();
    let task_summary = task_summary_regex
        .captures(input)
        .and_then(|cap| cap.get(1))
        .map_or(String::new(), |m| m.as_str().to_string());

    let parsed_sub_tasks: Vec<String> = match serde_json::from_str(&sub_tasks_str) {
        Ok(val) => val,
        Err(_) => {
            eprintln!("Failed to parse extracted 'sub_tasks' as JSON.");
            return (vec![], task_summary, solution_found);
        }
    };

    (parsed_sub_tasks, task_summary, solution_found)
}
