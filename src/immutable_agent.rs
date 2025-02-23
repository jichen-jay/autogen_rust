use crate::actor::{agent::AgentActor, AgentId, TopicId};
use crate::llama_structs::*;
use crate::llm_utils::*;
use crate::utils::*;
use crate::STORE;
use crate::{FormatterFn, LlmConfig, TOGETHER_CONFIG};
use anyhow::Result;
use async_openai::types::CompletionUsage;
use async_openai::types::Role;
use lazy_static::lazy_static;
use log::debug;
use ractor::ActorRef;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::HashMap;
use std::error::Error;
use std::sync::{Arc, Mutex, RwLock};
use tokio::io::{stdin, AsyncBufReadExt, BufReader};
use tokio::time::{timeout, Duration};
use uuid::Uuid;

lazy_static! {
    static ref END_STR: &'static str = r#"</tools> Use the following pydantic model json schema for each tool call you will make: {"properties": {"arguments": {"title": "Arguments", "type": "object"}, "name": {"title": "Name", "type": "string"}}, "required": ["arguments", "name"], "title": "FunctionCall", "type": "object"} For each function call return a json object with function name and arguments within <tool_call></tool_call> XML tags as follows:
<tool_call>
{"arguments": <args-dict>, "name": <function-name>}
</tool_call>"#;
    pub static ref TEMPLATE_SYSTEM_PROMPT_TOOL_USE: Arc<Mutex<FormatterFn>> =
        Arc::new(Mutex::new(Box::new(|args: &[&str]| {
            format!("{}{}{}", args[0], args[1], *END_STR)
        })));
    pub static ref TEMPLATE_tool_call: Arc<Mutex<FormatterFn>> =
        Arc::new(Mutex::new(Box::new(|args: &[&str]| {
            format!(
                "You're a tool use agent, here are tools avaiable to you: {}",
                args[0]
            )
        })));
}

#[derive(Debug, Clone)]
pub enum AgentResponse {
    Llama(LlamaResponseMessage),
    Proxy(String),
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct Message {
    pub content: Content,
    pub name: Option<String>,
    pub role: Role,
}

impl Default for Message {
    fn default() -> Self {
        Message {
            content: Content::Text("placeholder".to_string()),
            name: None,
            role: Role::User,
        }
    }
}

impl Message {
    pub fn new(content: Content, name: Option<String>, role: Role) -> Self {
        Message {
            content,
            name,
            role,
        }
    }
}

#[derive(Clone)]
pub struct LlmAgent {
    pub system_prompt: String,
    pub llm_config: Option<LlmConfig>,
    pub tools_map_meta: Option<Value>,
    pub description: String,
    tool_names: Vec<String>,
}

impl LlmAgent {
    pub fn build(
        system_prompt: String,
        llm_config: Option<LlmConfig>,
        tools_map_meta: Option<Value>,
        description: String,
    ) -> Self {
        let (system_prompt, tool_names) = match tools_map_meta.as_ref() {
            Some(tools_meta) => {
                let formatter = TEMPLATE_SYSTEM_PROMPT_TOOL_USE.lock().unwrap();
                let meta_str = tools_meta
                    .as_str()
                    .map(|s| s.to_owned())
                    .unwrap_or_else(|| tools_meta.to_string());
                let formatted_prompt = formatter(&[&system_prompt, &meta_str]);

                let names = match tools_meta.as_array() {
                    Some(tools_array) => tools_array
                        .iter()
                        .filter_map(|tool| {
                            tool.get("name")
                                .and_then(|n| n.as_str())
                                .or_else(|| {
                                    tool.get("function")
                                        .and_then(|f| f.get("name"))
                                        .and_then(|n| n.as_str())
                                })
                                .map(String::from)
                        })
                        .collect::<Vec<String>>(),
                    None => Vec::new(),
                };

                (formatted_prompt, names)
            }
            None => (system_prompt, Vec::new()),
        };

        Self {
            system_prompt,
            llm_config,
            description,
            tools_map_meta,
            tool_names,
        }
    }

    pub async fn default_method(&self, input: &str) -> Result<AgentResponse> {
        println!("default_method: received input: {:?}", input);

        let tool_names = self.tool_names.clone();
        println!("length of tool_names: {:?}", tool_names.len());

        match tool_names.len() {
            0 => {
                let user_prompt = format!("Here is the task for you: {:?}", input);
                let max_token = 1000u16;
                let llama_response = chat_inner_async_wrapper(
                    &TOGETHER_CONFIG,
                    &self.system_prompt,
                    &user_prompt,
                    max_token,
                )
                .await?;
                Ok(AgentResponse::Llama(llama_response))
            }
            1 if tool_names[0] == "get_user_feedback" => {
                let store = STORE.lock().unwrap();
                let tool = store
                    .get("get_user_feedback")
                    .expect("get_user_feedback not found in store");

                let output = tool.run(String::new()).expect("tool execution failed");

                Ok(AgentResponse::Proxy(output))
            }
            _ => {
                let available_tools = tool_names.join(", ");
                let user_prompt = format!(
                    "Task: {}\nAvailable tools: {}.\nIf necessary, include a tool to use in your response.",
                    input, available_tools
                );
                let max_token = 1000u16;
                let config = self.llm_config.as_ref().unwrap_or(&TOGETHER_CONFIG);

                let mut llama_response = String::new();
                let mut usage = CompletionUsage {
                    prompt_tokens: 0,
                    completion_tokens: 0,
                    total_tokens: 0,
                };

                match chat_inner_async_wrapper(config, &self.system_prompt, &user_prompt, max_token)
                    .await
                {
                    Ok(res) => match res.content {
                        Content::Text(tex) => {
                            println!("I'm inside the chat Text branch");
                            llama_response = tex;
                        }
                        //expect program to enter here, it didn't, it entered
                        Content::ToolCallStr(tcs) => {
                            println!("I'm inside the chat ToolCallStr branch");

                            usage = res.usage;
                            let func_name = tcs.name.clone();

                            let args_value = tcs.arguments.unwrap_or(String::new());

                            let binding = STORE.lock().unwrap();
                            if let Some(tool) = binding.get(&func_name) {
                                match tool.run(args_value) {
                                    Ok(tool_output) => {
                                        println!("function_call result: {}", tool_output.clone());

                                        llama_response = tool_output;
                                    }
                                    Err(e) => {
                                        eprintln!("Error executing tool {}: {}", func_name, e);
                                        llama_response =
                                            format!("Error executing tool {}: {}", func_name, e);
                                    }
                                }
                            } else {
                                eprintln!("Tool {} not found in STORE", func_name);
                                llama_response = format!("Tool {} not found", func_name);
                            }
                        }
                    },
                    Err(e) => {
                        eprintln!("Error in chat_inner_async_wrapper: {}", e);
                        llama_response =
                            "Failed to get a response from the chat system".to_string();
                    }
                };

                let wrapped_response = LlamaResponseMessage {
                    content: Content::Text(llama_response),
                    role: Role::Assistant,
                    usage: usage,
                };
                Ok(AgentResponse::Llama(wrapped_response))
            }
        }
    }
}
