use async_openai::types::{CompletionUsage, CreateChatCompletionResponse, Role};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::HashMap;

#[derive(Debug, Deserialize, Serialize, Clone, PartialEq)]
pub struct ToolCall {
    pub name: String,
    pub arguments: Option<HashMap<String, String>>,
}

#[derive(Debug, Deserialize, Serialize, Clone, PartialEq)]
pub struct ToolCallJson {
    pub name: String,
    pub arguments: Option<Value>,
}

#[allow(non_snake_case)]
#[derive(Debug, Deserialize, Serialize, Clone, PartialEq)]
pub enum Content {
    Text(String),
    // ToolCall(ToolCall),
    ToolCallJson(ToolCallJson),
}

impl Content {
    pub fn content_to_string(&self) -> String {
        match self {
            Content::Text(tex) => tex.to_string(),
            // Content::ToolCall(tc) => format!("{:?}", tc),
            Content::ToolCallJson(tcj) => format!("{:?}", tcj),
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
            // Content::ToolCall(tool_call) => format!(
            //     "tool_call: {}, arguments: {}",
            //     tool_call.name,
            //     tool_call
            //         .arguments
            //         .as_ref()
            //         .unwrap()
            //         .into_iter()
            //         .map(|(arg, val)| format!("{:?}: {:?}", arg, val))
            //         .collect::<Vec<String>>()
            //         .join(", ")
            // ),
            Content::ToolCallJson(tool_call_json) => format!(
                "tool_call: {}, arguments: {:?}",
                tool_call_json.name, tool_call_json.arguments
            ),
        }
    }
}

fn extract_json_from_xml_like(xml_like_data: &str) -> Option<String> {
    let start_tag = "<tool_call>";
    let end_tag = "</tool_call>";

    if xml_like_data.trim().starts_with(start_tag) && xml_like_data.trim().ends_with(end_tag) {
        let start_pos = start_tag.len();
        let end_pos = xml_like_data.len() - end_tag.len();
        Some(xml_like_data[start_pos..end_pos].trim().to_string())
    } else {
        None
    }
}

pub fn output_llama_response(
    res_obj: CreateChatCompletionResponse,
) -> Option<LlamaResponseMessage> {
    let usage = res_obj.clone().usage.unwrap();
    let msg_obj = res_obj.clone().choices[0].message.clone();
    let role = msg_obj.clone().role;
    if let Some(data) = msg_obj.content {
        if let Some(json_str) = extract_json_from_xml_like(&data) {
            println!("{:?}", json_str.clone());
            let (name, args) = extract_tool_call_json(&json_str);
            return Some(LlamaResponseMessage {
                content: Content::ToolCallJson(ToolCallJson {
                    name: name,
                    arguments: args,
                }),
                role: role,
                usage: usage,
            });
        } else {
            return Some(LlamaResponseMessage {
                content: Content::Text(data.to_owned()),
                role: role,
                usage: usage,
            });
        }
    } else {
        return Some(LlamaResponseMessage {
            content: Content::Text("empty result from LlamaResponse".to_string()),
            role: role,
            usage: usage,
        });
    }
}

fn extract_tool_call(input: &str) -> Option<ToolCall> {
    let re = regex::Regex::new(
        r#"\{"arguments":\s*(?P<arguments>\{.*?\}),\s*"name":\s*"(?P<name>.*?)"\}"#,
    )
    .unwrap();

    if let Some(caps) = re.captures(input) {
        let arguments_str = caps.name("arguments")?.as_str();
        let name = caps.name("name")?.as_str().to_string();

        if let Ok(arguments_value) = serde_json::from_str::<Value>(arguments_str) {
            let arguments_map = match arguments_value {
                Value::Object(map) => map
                    .into_iter()
                    .filter_map(|(k, v)| v.as_str().map(|v| (k, v.to_string())))
                    .collect(),
                _ => HashMap::new(),
            };

            return Some(ToolCall {
                arguments: Some(arguments_map),
                name,
            });
        }
    }

    None
}

fn extract_tool_call_json(input: &str) -> (String, Option<Value>) {
    let re = regex::Regex::new(
        r#"\{"arguments":\s*(?P<arguments>\{.*?\}),\s*"name":\s*"(?P<name>.*?)"\}"#,
    )
    .unwrap();

    if let Some(caps) = re.captures(input) {
        let arguments_str = caps.name("arguments")?.as_str();
        let name = caps.name("name")?.as_str().to_string();

        if let Ok(arguments_value) = serde_json::from_str::<Value>(arguments_str) {
            let arguments_map = match arguments_value {
                Value::Object(map) => (name.clone(), Some(map)),
                _ => (name.clone(), None),
            };
        } else {
            (name, None)
        }
    } else {
        ("function name not found".to_string(), None)
    }
}
