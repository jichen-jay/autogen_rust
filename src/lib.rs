#![allow(warnings, deprecated)]

pub mod agent_runtime;
pub mod immutable_agent;
pub mod llama;
pub mod use_tool;
use crate::use_tool::{Tool, TypeConverter};
use anyhow::Result;
use ctor::ctor;
use lazy_static::lazy_static;
use once_cell::sync::Lazy;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::HashMap;
use std::io::BufRead;
use std::io::{stdin, BufReader};
use std::sync::{Arc, Mutex};
use tokio::time::{timeout, Duration};
use tool_builder::create_tool_with_function;

lazy_static! {
    static ref END_STR: &'static str = r#"</tools> Use the following pydantic model json schema for each tool call you will make: {"properties": {"arguments": {"title": "Arguments", "type": "object"}, "name": {"title": "Name", "type": "string"}}, "required": ["arguments", "name"], "title": "FunctionCall", "type": "object"} For each function call return a json object with function name and arguments within <tool_call></tool_call> XML tags as follows:
<tool_call>
{"arguments": <args-dict>, "name": <function-name>}
</tool_call>"#;
    pub static ref TEMPLATE_SYSTEM_PROMPT_TOOL_USE: Arc<Mutex<FormatterFn>> =
        Arc::new(Mutex::new(Box::new(|args: &[&str]| {
            format!("{}{}{}", args[0], args[1], *END_STR)
        })));
    pub static ref TEMPLATE_: Arc<Mutex<FormatterFn>> =
        Arc::new(Mutex::new(Box::new(|args: &[&str]| {
            format!(
                "You're a tool use agent, here are tools avaiable to you: {}",
                args[0]
            )
        })));
}

#[derive(Debug, Deserialize, Serialize, Clone, PartialEq)]
pub struct LlmConfig {
    pub model: &'static str,
    pub base_url: &'static str,
    pub context_size: usize,
    pub api_key_str: &'static str,
}

// pub const DEEPINFRA_CONFIG: LlmConfig = LlmConfig {
//     model: "NousResearch/Hermes-3-Llama-3.1-405B",
//     context_size: 8192,
//     base_url: "https://api.deepinfra.com/v1/openai/chat/completions",
//     api_key_str: "DEEPINFRA_API_KEY",
// };

pub const TOGETHER_CONFIG: LlmConfig = LlmConfig {
    model: "google/gemma-2-9b-it",
    // model: "mistralai/Mistral-Small-24B-Instruct-2501",
    // model: "meta-llama/Llama-3.3-70B-Instruct-Turbo",
    context_size: 8192,
    base_url: "https://api.together.xyz/v1/chat/completions",
    api_key_str: "TOGETHER_API_KEY",
};

// const CODELLAMA_CONFIG: LlmConfig = LlmConfig {
//     model: "codellama/CodeLlama-34b-Instruct-hf",
//     context_size: 8192,
//     base_url: "https://api.together.xyz/v1/chat/completions",
//     api_key_str: "TOGETHER_API_KEY",
// };

// const QWEN_CONFIG: LlmConfig = LlmConfig {
//     model: "Qwen/Qwen2-72B-Instruct",
//     context_size: 32000,
//     base_url: "https://api.deepinfra.com/v1/openai/chat/completions",
//     api_key_str: "DEEPINFRA_API_KEY",
// };

// const DEEPSEEK_CONFIG: LlmConfig = LlmConfig {
//     model: "deepseek-coder",
//     context_size: 16000,
//     base_url: "https://api.deepseek.com/chat/completions",
//     api_key_str: "SEEK_API_KEY",
// };

type FormatterFn = Box<dyn (Fn(&[&str]) -> String) + Send + Sync>;

pub static STORE: Lazy<Mutex<HashMap<String, Tool>>> = Lazy::new(|| Mutex::new(HashMap::new()));

pub struct AgentType(String);

#[derive(Debug)]
pub struct FunctionToolError(String);

impl std::fmt::Display for FunctionToolError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for FunctionToolError {}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FunctionCallInput {
    pub arguments_obj: Value,
    pub function_name: String,
    pub return_type: String,
}

type MyResult<T> = Result<T, Box<dyn std::error::Error + Send + Sync>>;

#[create_tool_with_function(GET_USER_FEEDBACK_TOOL_DEF_OBJ)]
fn get_user_feedback() -> MyResult<String> {
    tokio::task::block_in_place(|| {
        let rt = tokio::runtime::Runtime::new().unwrap();
        rt.block_on(async {
            println!("Please provide your next instruction:");

            let mut input = String::new();
            let mut reader = BufReader::new(stdin());

            match timeout(Duration::from_secs(10), async {
                reader.read_line(&mut input).expect("Failed to read line");
                input
            })
            .await
            {
                Ok(mut input) => {
                    if let Some('\n') = input.chars().next_back() {
                        input.pop();
                    }
                    if let Some('\r') = input.chars().next_back() {
                        input.pop();
                    }
                    match input.as_str() {
                        "stop" => std::process::exit(0),
                        "back" => Err("back to main".into()),
                        _ => Ok(input),
                    }
                }
                Err(e) => {
                    eprintln!("{}", e);
                    std::process::exit(0);
                }
            }
        })
    })
}

#[create_tool_with_function(PROCESS_VALUE_TOOL_DEF_OBJ)]
fn process_values(a: i32, b: f32, c: bool, d: String, e: i32) -> MyResult<String> {
    if a > 10 {
        Ok(format!(
            "Processed: a = {}, b = {}, c = {}, d = {}, e = {}",
            a, b, c, d, e
        ))
    } else {
        Err(format!(
            "Processed: a = {}, b = {}, c = {}, d = {}, e = {}",
            a, b, c, d, e
        )
        .into())
    }
}

#[create_tool_with_function(GET_WEATHER_TOOL_DEF_OBJ)]
fn get_current_weather(location: String, unit: String) -> MyResult<String> {
    if location.contains("New") {
        Ok(format!("Weather for {} in 25 {} ", location, unit))
    } else {
        Err(format!("Weather for {} in {}", location, unit).into())
    }
}

pub const GET_USER_FEEDBACK_TOOL_DEF_OBJ: &str = r#"
{
    "name": "get_user_feedback"
}
"#;

pub const GET_WEATHER_TOOL_DEF_OBJ: &str = r#"
        {
            "name": "get_current_weather",
            "description": "Get the current weather in a given location",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "The city and state, e.g. San Francisco, CA"
                    },
                    "unit": {
                        "type": "string",
                        "enum": ["celsius", "fahrenheit"],
                        "description": "The unit of measurement"
                    }
                },
                "required": ["location", "unit"]
            }
        }
        "#;

pub const PROCESS_VALUE_TOOL_DEF_OBJ: &str = r#"
{
    "name": "process_values",
    "description": "Processes up to 5 different types of values",
    "parameters": {
        "type": "object",
        "properties": {
            "a": {
                "type": "i32",
                "description": "An integer value"
            },
            "b": {
                "type": "f32",
                "description": "A floating-point value"
            },
            "c": {
                "type": "bool",
                "description": "A boolean value"
            },
            "d": {
                "type": "string",
                "description": "A string value"
            },
            "e": {
                "type": "i32",
                "description": "Another integer value"
            }
        },
        "required": ["a", "b", "c", "d", "e"]
    }
}
"#;
