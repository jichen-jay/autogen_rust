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
use std::result::Result as StdResult;
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

#[derive(Debug, Error)]
pub enum BuilderError {
    #[error("Failed to serialize tools metadata: {0}")]
    MetaSerializationFailure(#[from] SerdeError),

    #[error("Tool name not found in metadata at index {index}: {data}")]
    ToolNameNotFound { index: usize, data: String },
}

#[derive(Error, Debug)]
pub enum DefaultMethodError {
    #[error("IO error: {0}")]
    IoError(#[from] std::io::Error),

    #[error("LLM API error: {0}")]
    LlmApiError(String),

    #[error("Tool execution error: {0}")]
    ToolExecutionError(String),

    #[error("Parsing error: {0}")]
    ParsingError(String),

    #[error("Tool not found: {0}")]
    ToolNotFound(String),
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

impl LlmAgent {
    pub fn build(
        system_prompt: String,
        user_prompt_formatter: Option<Arc<Mutex<FormatterFn>>>,
        llm_config: Option<LlmConfig>,
        tools_map_meta: Option<Value>,
        description: String,
    ) -> StdResult<Self, BuilderError> {
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

    pub async fn default_method(
        &self,
        input: &str,
    ) -> StdResult<LlamaResponseMessage, DefaultMethodError> {
        enum TaskOutput {
            text,
            tool_call,
            tasks,
        }
        println!("default_method: received input\n: {:?}\n", input);

        let tool_names = self.tool_names.clone();
        println!("tool_names: {:?}\n", tool_names.clone());

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
            let output = tool
                .run(String::new())
                .map_err(|e| DefaultMethodError::ToolExecutionError(e.to_string()))?;

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
        let config = self.llm_config.as_ref().unwrap_or(&TOGETHER_CONFIG);
        let attempt = AtomicUsize::new(0);

        let result = tryhard::retry_fn(|| async {
            let count = attempt.fetch_add(1, Ordering::Relaxed) + 1;
            println!("Attempt number: {}", count);

            let (resp, usage) =
                chat_inner_async_wrapper(config, &self.system_prompt, &user_prompt, max_token)
                    .await
                    .map_err(|e| DefaultMethodError::LlmApiError(e.to_string()))?;

            let content = match task_type {
                TaskOutput::text => Content::Text(resp.clone()),
                TaskOutput::tasks => {
                    let tasks = parse_planning_tasks(&resp).map_err(|e| {
                        DefaultMethodError::ParsingError(format!(
                            "Failed to parse planning tasks: {:?}",
                            e
                        ))
                    })?;
                    Content::Structured(StructuredText::Tasks(tasks))
                }
                TaskOutput::tool_call => {
                    let json_str = extract_json_from_xml_like(&resp).map_err(|e| {
                        DefaultMethodError::ParsingError(format!("JSON extraction error: {}", e))
                    })?;
                    let tool_call = extract_tool_call_json(&json_str).map_err(|e| {
                        DefaultMethodError::ParsingError(format!("Tool call parse error: {}", e))
                    })?;

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
                                let error_msg =
                                    format!("Error executing tool {}: {}", func_name, e);
                                eprintln!("{}", error_msg);
                                return Err(DefaultMethodError::ToolExecutionError(error_msg));
                            }
                        }
                    } else {
                        let error_msg = format!("Tool {} not found", func_name);
                        eprintln!("{}", error_msg);
                        return Err(DefaultMethodError::ToolNotFound(error_msg));
                    }
                }
            };

            Ok((resp, usage, content))
        })
        .retries(2)
        .await?;

        let (_, usage, content) = result;

        Ok(LlamaResponseMessage {
            content,
            role: Role::Assistant,
            usage,
        })
    }
}
