#![allow(warnings, deprecated)]

use anyhow::anyhow;
use anyhow::Result;
use async_openai::types::{CompletionUsage, CreateChatCompletionResponse, Role};

use autogen_rust::agent_runtime::{
    agent::{AgentActor, AgentState},
    router::{RouterActor, RouterState, RouterStatus},
    ActorContext, AgentId, MessageContext, RouterCommand, SpawnAgentResponse, TopicId,
};
use autogen_rust::{immutable_agent::*, llama::*, FormatterWrapper, LlmConfig};
use autogen_rust::{
    STORE, TEMPLATE_SYSTEM_PROMPT_TOOL_USE, TEMPLATE_USER_PROMPT_TASK_JSON,
    TEMPLATE_USER_PROMPT_TOOL_USE,
};
use env_logger;
use escape8259::{escape, unescape};
use ractor::{call_t, rpc::CallResult, spawn_named, Actor, ActorCell, ActorRef, RpcReplyPort};
use serde_json::{json, Value};
use std::collections::HashMap;
use std::marker::PhantomData;
use std::time::Duration;
use tokio::time;
use tokio_util::sync::CancellationToken;
use uuid::Uuid;

#[tokio::main]
async fn main() -> Result<()> {
    dotenv::dotenv().ok();
    env_logger::init();
    std::env::set_var("RUST_LOG", "debug");

    let tools_map_meta =         serde_json::json!([{
    "type": "function",
    "function": {
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
                "description": "The temperature unit to use. Infer this from the users location."
            }
        },
        "required": ["location", "unit"]
    }
}
},
{
"type": "function",
"function": {
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
}]).to_string();

    let TOGETHER_CONFIG: LlmConfig = LlmConfig {
        model: "google/gemma-2-9b-it",
        // model: "mistralai/Mistral-Small-24B-Instruct-2501",
        // model: "meta-llama/Llama-3.3-70B-Instruct-Turbo",
        context_size: 8192,
        base_url: "https://api.together.xyz/v1/chat/completions",
        api_key_str: "TOGETHER_API_KEY",
    };

    let max_token = 1000u16;

    let mut llama_response = String::new();
    let mut usage = CompletionUsage {
        prompt_tokens: 0,
        completion_tokens: 0,
        total_tokens: 0,
    };

    let system_prompt = TEMPLATE_SYSTEM_PROMPT_TOOL_USE.clone();

    let pretty_json = unescape(&system_prompt)?;
    println!("system_prompt:\n {}\n\n", pretty_json);

    let input =
        "Fetch the weather of New York in Celsius unit".to_string();
    let formatter = TEMPLATE_USER_PROMPT_TOOL_USE.lock().unwrap();

    let user_prompt = formatter(&[&input, &tools_map_meta]);

    let pretty_json = user_prompt.clone();
    // let pretty_json = unescape(&user_prompt)?;
    println!("user_prompt:\n {}\n\n", pretty_json);

    match chat_inner_async_wrapper(&TOGETHER_CONFIG, system_prompt, &user_prompt, max_token).await {
        Ok(res) => match res.content {
            Content::Text(tex) => {
                println!("I'm inside the chat Text branch");
                llama_response = tex;
            }
            Content::JsonStr(JsonStr::ToolCall(tc)) => {
                println!("I'm inside the chat JsonStr branch");

                usage = res.usage;
                let func_name = tc.name.clone();

                let args_value = tc.arguments.unwrap_or(String::new());

                let binding = STORE.lock().unwrap();
                if let Some(tool) = binding.get(&func_name) {
                    match tool.run(args_value) {
                        Ok(tool_output) => {
                            println!("function_call result: {}", tool_output.clone());

                            llama_response = tool_output;
                        }
                        Err(e) => {
                            eprintln!("Error executing tool {}: {}", func_name, e);
                            llama_response = format!("Error executing tool {}: {}", func_name, e);
                        }
                    }
                } else {
                    eprintln!("Tool {} not found in STORE", func_name);
                    llama_response = format!("Tool {} not found", func_name);
                }
            }
            Content::JsonStr(JsonStr::JsonLoad(jl)) => {
                println!("I'm inside the chat JsonStr branch");

                usage = res.usage;

                todo!()
            }
        },
        Err(e) => {
            eprintln!("Error in chat_inner_async_wrapper: {}", e);
            llama_response = "Failed to get a response from the chat system".to_string();
        }
    };

    Ok(())
}
