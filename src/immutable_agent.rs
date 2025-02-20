use crate::actor::{agent::AgentActor, AgentId, TopicId};
use crate::llama_structs::*;
use crate::llm_utils::*;
use crate::utils::*;
use crate::TOGETHER_CONFIG;
use anyhow::Result;
use async_openai::types::Role;
use log::debug;
use ractor::ActorRef;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::HashMap;
use std::error::Error;
use std::sync::{Arc, RwLock};
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
    pub llm_config: Option<Value>,
    pub tools_map_meta: String,
    pub description: String,
}

impl LlmAgent {
    pub fn new(system_prompt: String, llm_config: Option<Value>) -> Self {
        Self {
            system_prompt,
            llm_config,
            tools_map_meta: String::new(),
            description: String::new(),
        }
    }

    pub async fn default_method(&self, input: &str) -> anyhow::Result<LlamaResponseMessage> {
        println!("default_method: received input: {:?}", input);

        let user_prompt = format!("Here is the task for you: {:?}", input);

        let max_token = 1000u16;
        chat_inner_async_wrapper(
            &TOGETHER_CONFIG,
            &self.system_prompt,
            &user_prompt,
            max_token,
        )
        .await
    }
}

pub async fn get_user_feedback() -> Result<String> {
    // print!("User input: ");

    let mut input = String::new();

    let mut reader = BufReader::new(stdin());

    match timeout(Duration::from_secs(10), async {
        reader
            .read_line(&mut input)
            .await
            .expect("Failed to read line");
        input
    })
    .await
    {
        Ok(mut input) => {
            if let Some('\n') = input.clone().chars().next_back() {
                input.pop();
            }
            if let Some('\r') = input.chars().next_back() {
                input.pop();
            }

            if input == "stop" {
                std::process::exit(0);
            }
            if input == "back" {
                return Err(anyhow::Error::msg("back to main"));
            }
            Ok(input)
        }
        Err(e) => {
            eprintln!("{}", e);
            std::process::exit(0);
        }
    }
}
