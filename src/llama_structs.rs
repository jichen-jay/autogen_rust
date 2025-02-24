use anyhow::{anyhow, Result};
use async_openai::types::{CompletionUsage, CreateChatCompletionResponse, Role};
use log;
use regex::Regex;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::HashMap;
use std::result::Result as StdResult;
use thiserror::Error;

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

pub fn output_llama_response(
    res_obj: CreateChatCompletionResponse,
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

    let json_str = extract_json_from_xml_like(data).map_err(|e| {
        Box::new(LlamaResponseError::JsonExtractionError(e.to_string()))
            as Box<dyn std::error::Error>
    })?;
    let pretty_json = jsonxf::pretty_print(&json_str).map_err(|e| {
        Box::new(LlamaResponseError::JsonFormatError(e.to_string())) as Box<dyn std::error::Error>
    })?;
    println!("Json Output original string:\n{}\n", pretty_json);

    match extract_tool_call_json(&json_str) {
        Ok(json_data) => {
            Ok::<LlamaResponseMessage, Box<dyn std::error::Error>>(LlamaResponseMessage {
                content: Content::JsonStr(json_data),
                role,
                usage,
            })
        }
        Err(e) => {
            log::warn!(
                "Falling back to text content due to tool call parse error: {}",
                e
            );
            Ok(LlamaResponseMessage {
                content: Content::Text(data.to_string()),
                role,
                usage,
            })
        }
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
