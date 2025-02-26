use crate::agent_runtime::{agent::AgentActor, AgentId, TopicId};
use crate::llama::{
    chat_inner_async_wrapper,
    llama_utils::{extract_json_from_xml_like, extract_tool_call_json, parse_planning_tasks},
    Content, JsonStr, LlamaResponseError, LlamaResponseMessage,
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
    task_type: TaskOutput,
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
    plan,
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
        task_type: TaskOutput,
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
            task_type,
        })
    }

    pub async fn default_method(&self, input: &str) -> anyhow::Result<LlamaResponseMessage> {
        println!("default_method: received input: {:?}", input);

        let tool_names = self.tool_names.clone();
        println!("length of tool_names: {:?}", tool_names.len());

        let default_usage = CompletionUsage {
            prompt_tokens: 0,
            completion_tokens: 0,
            total_tokens: 0,
        };

        match tool_names.len() {
            // 0 => {
            //     let user_prompt = match &self.user_prompt_formatter {
            //         None => format!("here is your task: {}", input),

            //         Some(f) => {
            //             let formatter = f.lock().unwrap();

            //             formatter(&[&input])
            //         }
            //     };

            //     let max_token = 1000u16;
            //     let chat_response = chat_inner_async_wrapper(
            //         &TOGETHER_CONFIG,
            //         &self.system_prompt,
            //         &user_prompt,
            //         max_token,
            //     )
            //     .await?;

            //     let llama_response: LlamaResponseMessage = output_response_by_task(&chat_response)?;

            //     Ok(llama_response)
            // }
            1 if tool_names[0] == "get_user_feedback" => {
                let store = STORE.lock().unwrap();
                let tool = store
                    .get("get_user_feedback")
                    .expect("get_user_feedback not found in store");

                let output = tool.run(String::new()).expect("tool execution failed");

                let content = Content::Text(output.to_string());

                Ok(LlamaResponseMessage {
                    content: content,
                    role: Role::Assistant,
                    usage: default_usage.clone(),
                })
            }
            _ => {
                let user_prompt = match &self.user_prompt_formatter {
                    None => format!("here is your task: {}", input),

                    Some(f) => {
                        let formatter = f.lock().unwrap();

                        formatter(&[&input])
                    }
                };

                let max_token = 1000u16;
                let config = self.llm_config.as_ref().unwrap_or(&TOGETHER_CONFIG);

                let mut llama_response = String::new();
                let mut usage = CompletionUsage {
                    prompt_tokens: 0,
                    completion_tokens: 0,
                    total_tokens: 0,
                };

                let chat_response =
                    chat_inner_async_wrapper(config, &self.system_prompt, &user_prompt, max_token)
                        .await?;

                let usage = chat_response.usage.unwrap_or(default_usage.clone());
                let choice = chat_response.choices.first().ok_or(Err(anyhow::Error::msg(
                    "empty choices in ChatResponse".to_string(),
                )));
                let msg_obj = &choice.unwrap().message;

                let role = msg_obj.role.clone();
                let data = msg_obj.content.as_ref()?;

                let content = match self.task_type {
                    TaskOutput::text => Content::Text(data.to_string()),
                    TaskOutput::tool_call => {
                        let json_str = extract_json_from_xml_like(data).map_err(|e| {
                            Box::new(LlamaResponseError::JsonExtractionError(e.to_string()))
                        })?;
                        let pretty_json = jsonxf::pretty_print(&json_str).map_err(|e| {
                            Box::new(LlamaResponseError::JsonFormatError(e.to_string()))
                        })?;
                        println!("JSON Output:\n{}\n", pretty_json);

                        match extract_tool_call_json(&json_str) {
                            Ok(tc) => {
                                println!("I'm inside the chat JsonStr branch");

                                let func_name = tc.name.clone();

                                let args_value = tc.arguments.unwrap_or(String::new());

                                let binding = STORE.lock().unwrap();
                                let mut tc_result = String::new();
                                if let Some(tool) = binding.get(&func_name) {
                                    match tool.run(args_value) {
                                        Ok(tool_output) => {
                                            println!(
                                                "function_call result: {}",
                                                tool_output.clone()
                                            );

                                            tc_result = tool_output;
                                        }
                                        Err(e) => {
                                            eprintln!("Error executing tool {}: {}", func_name, e);
                                            tc_result = format!(
                                                "Error executing tool {}: {}",
                                                func_name, e
                                            );
                                        }
                                    }
                                } else {
                                    eprintln!("Tool {} not found in STORE", func_name);
                                    tc_result = format!("Tool {} not found", func_name);
                                }

                                Content::Text(tc_result)
                            }
                            Err(e) => {
                                log::warn!("Tool call parse error, falling back to text: {}", e);
                                Content::Text(json_str.to_string())
                            }
                        }
                    }
                    TaskOutput::plan => {
                        let tasks = parse_planning_tasks(data).map_err(|e| {
                            LlamaResponseError::ToolCallParseError(format!(
                                "Failed to parse planning tasks: {}",
                                e
                            ))
                        })?;

                        let planning_data = json!({
                            "tasks": tasks
                        });

                        Content::JsonStr(JsonStr::JsonLoad(planning_data))
                    }
                };

                let wrapped_response = LlamaResponseMessage {
                    content: content,
                    role: Role::Assistant,
                    usage: usage,
                };
                Ok(wrapped_response)
            }
        }
    }
}
