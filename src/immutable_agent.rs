use crate::agent_runtime::{agent::AgentActor, AgentId, TopicId};
use crate::llama::{
    chat_inner_async_wrapper,
    llama_utils::{extract_json_from_xml_like, extract_tool_call_json, parse_planning_tasks},
    Content, LlamaResponseError, LlamaResponseMessage, StructuredText, Task, ToolCall,
};
use crate::{
    FormatterFn, LlmConfig, STORE, TEMPLATE_SYSTEM_PROMPT_PLANNER, TEMPLATE_SYSTEM_PROMPT_TOOL_USE,
    TEMPLATE_USER_PROMPT_TASK_JSON, TOGETHER_CONFIG,
};
use anyhow::Result;
use async_openai::types::CompletionUsage;
use async_openai::types::Role;
use log::debug;
use ractor::ActorRef;
use serde::{Deserialize, Serialize};
use serde_json::json;
use serde_json::{Error as SerdeError, Value};
use std::collections::HashMap;
use std::error::Error as StdError;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, RwLock};
use thiserror::Error;
use tokio::io::{stdin, AsyncBufReadExt, BufReader};
use tokio::time::{timeout, Duration};
use uuid::Uuid;

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
    pub user_prompt_formatter: Option<Arc<Mutex<FormatterFn>>>,
    pub llm_config: Option<LlmConfig>,
    pub tools_map_meta: Option<Value>,
    pub description: String,
    tool_names: Vec<String>,
}

#[derive(Debug, Error)]
pub enum BuilderError {
    #[error("Failed to serialize tools metadata: {0}")]
    MetaSerializationFailure(#[from] SerdeError),

    #[error("Tool name not found in metadata at index {index}: {data}")]
    ToolNameNotFound { index: usize, data: String },
}
#[derive(Clone)]
pub enum TaskOutput {
    text,
    tool_call,
    tasks,
}

impl LlmAgent {
    //am considering is add a user prompt formatter function in this method
    //or should I keep the formatter function as a field of agent struct
    pub fn build(
        system_prompt: String,
        user_prompt_formatter: Option<Arc<Mutex<FormatterFn>>>,
        llm_config: Option<LlmConfig>,
        tools_map_meta: Option<Value>,
        description: String,
    ) -> Result<Self, BuilderError> {
        let (system_prompt, tool_names) = match tools_map_meta.as_ref() {
            Some(tools_meta) => {
                let formatted_prompt = TEMPLATE_SYSTEM_PROMPT_TOOL_USE.to_string();

                let names = match tools_meta.as_array() {
                    Some(tools_array) => tools_array
                        .iter()
                        .enumerate()
                        .map(|(index, tool)| {
                            tool.get("name")
                                .and_then(|n| n.as_str())
                                .or_else(|| {
                                    tool.get("function")
                                        .and_then(|f| f.get("name"))
                                        .and_then(|n| n.as_str())
                                })
                                .map(String::from)
                                .ok_or_else(|| BuilderError::ToolNameNotFound {
                                    index,
                                    data: tool.to_string(),
                                })
                        })
                        .collect::<Result<Vec<String>, BuilderError>>()?,
                    None => Vec::new(),
                };

                Ok::<(std::string::String, Vec<std::string::String>), BuilderError>((
                    formatted_prompt,
                    names,
                ))
            }
            None => Ok((system_prompt, Vec::new())),
        }?;

        Ok(Self {
            system_prompt,
            user_prompt_formatter,
            llm_config,
            description,
            tools_map_meta,
            tool_names,
        })
    }

    pub async fn default_method(&self, input: &str) -> anyhow::Result<LlamaResponseMessage> {
        println!("default_method: received input: {:?}", input);

        let tool_names = self.tool_names.clone();
        println!("tool_names: {:?}", tool_names.clone());

        let default_usage = CompletionUsage {
            prompt_tokens: 0,
            completion_tokens: 0,
            total_tokens: 0,
        };

        if tool_names
            .first()
            .map(|name| name == "get_user_feedback")
            .unwrap_or(false)
        {
            let store = STORE.lock().unwrap();
            let tool = store
                .get("get_user_feedback")
                .expect("get_user_feedback not found in store");
            let output = tool.run(String::new()).expect("tool execution failed");
            return Ok(LlamaResponseMessage {
                content: Content::Text(output.to_string()),
                role: Role::Assistant,
                usage: default_usage,
            });
        }

        let task_type = if self.user_prompt_formatter.is_some()
            && self.description.to_lowercase().contains("plan")
        {
            TaskOutput::tasks
        } else if !tool_names.is_empty() {
            TaskOutput::tool_call
        } else {
            TaskOutput::text
        };

        let user_prompt = match &self.user_prompt_formatter {
            None => format!("here is your task: {}", input),
            Some(f) => {
                let formatter = f.lock().unwrap();
                formatter(&[
                    &input,
                    &self
                        .tools_map_meta
                        .clone()
                        .unwrap_or(Value::String(String::new()))
                        .to_string(),
                ])
            }
        };

        let max_token = 1000u16;

        match task_type {
            TaskOutput::text => {
                let (response_text, usage) = chat_inner_async_wrapper(
                    &TOGETHER_CONFIG,
                    &self.system_prompt,
                    &user_prompt,
                    max_token,
                )
                .await?;
                Ok(LlamaResponseMessage {
                    content: Content::Text(response_text),
                    role: Role::Assistant,
                    usage,
                })
            }
            //merge the next 2 arms to simplify logic
            TaskOutput::tasks => {
                let config = self.llm_config.as_ref().unwrap_or(&TOGETHER_CONFIG);

                let (response_text, usage, tasks) = tryhard::retry_fn(|| async {
                    let (resp, usage) = chat_inner_async_wrapper(
                        config,
                        &self.system_prompt,
                        &user_prompt,
                        max_token,
                    )
                    .await?;
                    let tasks = parse_planning_tasks(&resp)
                        .map_err(|e| anyhow::anyhow!("Failed to parse planning tasks: {:?}", e))?;
                    Ok::<(String, CompletionUsage, Vec<Task>), anyhow::Error>((resp, usage, tasks))
                })
                .retries(2)
                .await?;

                Ok(LlamaResponseMessage {
                    content: Content::Structured(StructuredText::Tasks(tasks)),
                    role: Role::Assistant,
                    usage,
                })
            }
            TaskOutput::tool_call => {
                let config = self.llm_config.as_ref().unwrap_or(&TOGETHER_CONFIG);

                let attempt = AtomicUsize::new(0);
                let (chat_response_text, usage, tool_call) = tryhard::retry_fn(|| async {
                    let count = attempt.fetch_add(1, Ordering::Relaxed) + 1;
                    println!("Attempt number: {}", count);
                    let (rsp, usage) = chat_inner_async_wrapper(
                        config,
                        &self.system_prompt,
                        &user_prompt,
                        max_token,
                    )
                    .await?;
                    let json_str = extract_json_from_xml_like(&rsp)
                        .map_err(|e| LlamaResponseError::JsonExtractionError(e.to_string()))?;
                    let tool_call = extract_tool_call_json(&json_str)
                        .map_err(|e| LlamaResponseError::ToolCallParseError(e.to_string()))?;
                    Ok::<(String, CompletionUsage, ToolCall), anyhow::Error>((
                        rsp, usage, tool_call,
                    ))
                })
                .retries(2)
                .await?;

                // Run the extracted tool call.
                let tool_content = {
                    let func_name = tool_call.name.clone();
                    let args_value = tool_call.arguments.unwrap_or_else(String::new);
                    let binding = STORE.lock().unwrap();
                    if let Some(tool) = binding.get(&func_name) {
                        match tool.run(args_value) {
                            Ok(tool_output) => {
                                println!("function_call result: {}", tool_output);
                                Content::Text(tool_output)
                            }
                            Err(e) => {
                                eprintln!("Error executing tool {}: {}", func_name, e);
                                Content::Text(format!("Error executing tool {}: {}", func_name, e))
                            }
                        }
                    } else {
                        eprintln!("Tool {} not found in STORE", func_name);
                        Content::Text(format!("Tool {} not found", func_name))
                    }
                };

                Ok(LlamaResponseMessage {
                    content: tool_content,
                    role: Role::Assistant,
                    usage,
                })
            }
        }
    }
}
