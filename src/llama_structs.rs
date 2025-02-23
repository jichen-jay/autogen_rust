use anyhow::{anyhow, Result};
use async_openai::types::{CompletionUsage, CreateChatCompletionResponse, Role};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::HashMap; // Added anyhow import

#[derive(Debug, Deserialize, Serialize, Clone, PartialEq)]
pub struct ToolCallStr {
    pub name: String,
    pub arguments: Option<String>,
}

#[allow(non_snake_case)]
#[derive(Debug, Deserialize, Serialize, Clone, PartialEq)]
pub enum Content {
    Text(String),
    ToolCallStr(ToolCallStr),
}

impl Content {
    pub fn content_to_string(&self) -> String {
        match self {
            Content::Text(tex) => tex.to_string(),
            Content::ToolCallStr(tcs) => format!("{:?}", tcs),
        }
    }

    pub fn from_str(input: &str) -> Content {
        Content::Text(input.to_string())
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
        match &self.content {
            Content::Text(text) => text.clone(),
            Content::ToolCallStr(tcs) => {
                format!("tool_call: {}, arguments: {:?}", tcs.name, tcs.arguments)
            }
        }
    }
}

fn extract_json_from_xml_like(xml_like_data: &str) -> Result<String> {
    let start_tag = "<tool_call>";
    let end_tag = "</tool_call>";

    if xml_like_data.trim().starts_with(start_tag) && xml_like_data.trim().ends_with(end_tag) {
        let start_pos = start_tag.len();
        let end_pos = xml_like_data.len() - end_tag.len();
        Ok(xml_like_data[start_pos..end_pos].trim().to_string())
    } else {
        Err(anyhow!("Failed to parse tool_call")) // Fixed error macro
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

    println!("tool_call original string\n: {:?}\n\n", json_str);
    match extract_tool_call_json(&json_str) {
        Ok((name, arguments)) => Ok(LlamaResponseMessage {
            content: Content::ToolCallStr(ToolCallStr { name, arguments }),
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

fn extract_tool_call_json(input: &str) -> Result<(String, Option<String>)> {
    let re = regex::Regex::new(
        r#"\{"arguments":\s*(?P<arguments>\{.*?\}),\s*"name":\s*"(?P<name>.*?)"\}"#,
    )?;

    let caps = re
        .captures(input)
        .ok_or_else(|| anyhow!("No captures found"))?;
    let arguments_str = caps
        .name("arguments")
        .ok_or_else(|| anyhow!("No arguments found"))?
        .as_str();
    let name = caps
        .name("name")
        .ok_or_else(|| anyhow!("No name found"))?
        .as_str()
        .to_string();

    match serde_json::from_str::<Value>(arguments_str) {
        Ok(_) => Ok((name, Some(arguments_str.to_string()))),
        Err(_) => Ok((name, None)),
    }
}
