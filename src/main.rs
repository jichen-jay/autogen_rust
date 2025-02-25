#![allow(warnings, deprecated)]

use anyhow::anyhow;
use anyhow::Result;
use async_openai::types::Role;
use autogen_rust::agent_runtime::{
    agent::{AgentActor, AgentState},
    router::{RouterActor, RouterState, RouterStatus},
    ActorContext, AgentId, MessageContext, RouterCommand, SpawnAgentResponse, TopicId,
};
use autogen_rust::llama::*;
use autogen_rust::FormatterWrapper;
use autogen_rust::{immutable_agent::*, llama::Content};
use env_logger;
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

    let router_actor = RouterActor::default();
    let (router_ref, _handle) = Actor::spawn(Some("router".to_string()), router_actor, ()).await?;

    router_ref.cast(RouterCommand::Ready)?;

    // let task_agent_id = spawn_agent(
    //     router_ref.clone(),
    //     "You're an AI assistant".to_string(),
    //     TopicId::from("chat"),
    //     None,
    // )
    // .await?;

    let task_agent_id = spawn_agent(
        router_ref.clone(),
        "<|im_start|>system You are a function calling AI model. You are provided with function signatures within <tools></tools> XML tags. You may call one or more functions to assist with the user query. Don't make assumptions about what values to plug into functions. Here are the available tools: <tools>".to_string(),
      None,
        TopicId::from("chat"),
        Some(json!([{
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
}]))
    )
    .await?;
    println!("Task agent id: {}", task_agent_id);

    time::sleep(std::time::Duration::from_secs(1)).await;
    let temp_agent_id = Uuid::new_v4();
    // let user_proxy_agent_id = spawn_agent(
    //     router_ref.clone(),
    //     "get_user_feedback".to_string(),
    //     TopicId::from("chat"),
    //     None,
    // )
    // .await?;

    // let user_proxy_agent_id = spawn_agent(
    //     router_ref.clone(),
    //     "You're a user proxy agent, sending tasks".to_string(),
    //     TopicId::from("chat"),
    //     Some(serde_json::json!([{"name": "get_user_feedback"}])),
    // )
    // .await?;

    // println!("User Proxy agent id: {}", user_proxy_agent_id);

    time::sleep(std::time::Duration::from_secs(1)).await;

    let task_message = Message::new(
        Content::Text(
            "<|im_start|>user Fetch the weather of New York in Celsius unit<|im_end|>".to_string(),
        ),
        None,
        Role::User,
    );

    let task_context = ActorContext::new()
        .with_sender(temp_agent_id)
        .with_topic("chat".to_string());
    router_ref.cast(RouterCommand::RouteMessage {
        topic: "chat".to_string(),
        message: task_message,
        context: task_context,
    })?;

    // time::sleep(std::time::Duration::from_secs(3)).await;

    // println!("Notifying UserProxy agent to initiate shutdown (its default_method will read terminal input).");

    time::sleep(Duration::from_secs(5)).await;

    router_ref.cast(RouterCommand::ShutdownAgent {
        agent_id: task_agent_id,
    })?;
    // router_ref.cast(RouterCommand::ShutdownAgent {
    //     agent_id: user_proxy_agent_id,
    // })?;
    router_ref.cast(RouterCommand::Off)?;

    time::sleep(Duration::from_secs(1)).await;

    Ok(())
}

async fn spawn_agent(
    router_ref: ActorRef<RouterCommand>,
    system_prompt: String,
    user_prompt_formatter: Option<FormatterWrapper>,
    topic: TopicId,
    tools_map_meta: Option<Value>,
) -> Result<AgentId> {
    let spawn_response = router_ref
        .call(
            |reply_to| RouterCommand::SpawnAgent {
                system_prompt,
                user_prompt_formatter,
                topic,
                tools_map_meta,
                reply_to,
            },
            None, // Optional timeout can be passed here if needed.
        )
        .await;

    match spawn_response {
        Ok(CallResult::Success(Ok(agent_id))) => {
            println!("Successfully spawned agent with id: {:?}", agent_id);
            Ok(agent_id)
        }
        Ok(CallResult::Success(Err(error_msg))) => {
            println!("Agent failed to spawn: {}", error_msg);
            Err(anyhow!("Agent failed to spawn: {}", error_msg))
        }
        Ok(CallResult::Timeout) => {
            println!("Agent spawn timed out");
            Err(anyhow!("Agent spawn timed out"))
        }
        Ok(CallResult::SenderError) => {
            println!("Sender error during agent spawn");
            Err(anyhow!("Sender error during agent spawn"))
        }
        Err(err) => {
            println!("Messaging error during agent spawn: {:?}", err);
            Err(anyhow!("Messaging error during agent spawn: {:?}", err))
        }
    }
}
