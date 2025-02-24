use anyhow::{anyhow, Result};
use async_openai::types::{CompletionUsage, CreateChatCompletionResponse, Role};
use regex::Regex;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::HashMap;

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

fn extract_json_from_xml_like(xml_like_data: &str) -> Result<String> {
    let start_tag = "<tool_call>";
    let end_tag = "</tool_call>";

    let trimmed = xml_like_data.trim();
    if trimmed.starts_with(start_tag) && trimmed.ends_with(end_tag) {
        let start_pos = start_tag.len();
        let end_pos = trimmed.len() - end_tag.len();
        Ok(trimmed[start_pos..end_pos].trim().to_string())
    } else {
        Err(anyhow!("Failed to parse tool_call"))
    }
}

pub fn output_llama_response(
    res_obj: CreateChatCompletionResponse,
) -> Result<LlamaResponseMessage> {
    let usage = res_obj.usage.ok_or_else(|| anyhow!("Missing usage"))?;
    let msg_obj = &res_obj.choices[0].message;
    let role = msg_obj.role.clone();
    let data = msg_obj
        .content
        .as_ref()
        .ok_or_else(|| anyhow!("Missing content"))?;

    let json_str = extract_json_from_xml_like(data)?;

    let pretty_json = jsonxf::pretty_print(&json_str).expect("failed to convert for pretty print");
    println!("Json Output original string:\n {}\n", pretty_json);

    match extract_tool_call_json(&json_str) {
        Ok(json_data) => Ok(LlamaResponseMessage {
            content: Content::JsonStr(json_data),
            role,
            usage,
        }),
        Err(_) => Ok(LlamaResponseMessage {
            content: Content::Text(data.to_string()),
            role,
            usage,
        }),
    }
}

use std::process::Command;

pub fn extract_tool_call_json(input: &str) -> Result<JsonStr> {
    let repaired_input = repair_json_with_tool(input)?;

    if let Ok(parsed) = serde_json::from_str::<Value>(&repaired_input) {
        if let Some(name_val) = parsed.get("name") {
            if name_val.is_string() {
                let name = name_val.as_str().unwrap().to_string();
                let arguments_slice = if parsed.get("arguments").is_some() {
                    let re_args = Regex::new(r#"(?s)"arguments"\s*:\s*(?P<args>\{.*?\})"#)?;
                    if let Some(cap_args) = re_args.captures(&repaired_input) {
                        Some(cap_args.name("args").unwrap().as_str().to_string())
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
        }
        return Ok(JsonStr::JsonLoad(parsed));
    }

    // Step 3: Fallback to fuzzy matching with regex
    let re = Regex::new(
        r#"(?i)\{"arguments":\s*(?P<arguments>\{.*\}),\s*"name":\s*"(?P<name>[^"]+)"\}"#,
    )?;
    if let Some(caps) = re.captures(input) {
        let arguments_str = caps
            .name("arguments")
            .ok_or_else(|| anyhow!("No arguments found"))?
            .as_str()
            .to_string();
        let name = caps
            .name("name")
            .ok_or_else(|| anyhow!("No name found"))?
            .as_str()
            .to_string();

        return Ok(JsonStr::ToolCall(ToolCall {
            name,
            arguments: Some(arguments_str),
        }));
    }

    Err(anyhow!(
        "Input is neither valid JSON (after repair) nor does it match the fuzzy tool-call pattern"
    ))
}

fn repair_json_with_tool(input: &str) -> Result<String> {
    let output = Command::new("./jsonrepair").args(&["-i", input]).output()?;

    if !output.status.success() {
        return Err(anyhow!(
            "Repair tool failed with error: {}",
            String::from_utf8_lossy(&output.stderr)
        ));
    }

    match String::from_utf8(output.stdout) {
        Ok(res) => {
            println!("repair output: {:?}", res.clone());
            Ok(res)
        }
        Err(e) => Err(anyhow!("Invalid UTF-8 in repair tool output: {}", e)),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json;

    #[test]
    fn test_extract_tool_call_json() {
        // Test input: a JSON string representing a chat completion response.
        let test_input = r#"
  {\"arguments\": {\"location\": \"New York\", \"unit\": \"celsius\"}, \"name\": \"get_current_weather\"}",
    "#;

        // Extract tool call information using extract_tool_call_json.
        let result = extract_tool_call_json(&test_input).expect("Failed to extract tool call json");

        // Verify that the returned JsonStr is a ToolCall variant with expected values.
        match result {
            JsonStr::ToolCall(tool_call) => {
                assert_eq!(tool_call.name, "get_current_weather");
                assert_eq!(
                    tool_call.arguments,
                    Some("{\"location\": \"New York\", \"unit\": \"celsius\"}".to_string())
                );
            }
            _ => panic!("Expected JsonStr::ToolCall variant, but received a different variant"),
        }
    }
}
