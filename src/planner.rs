#![allow(warnings, deprecated)]

use anyhow::anyhow;
use anyhow::Result;
use async_openai::types::Role;
use autogen_rust::agent_runtime::{
    agent::{AgentActor, AgentState},
    router::{RouterActor, RouterState, RouterStatus},
    ActorContext, AgentId, MessageContext, RouterCommand, TopicId,
};
use autogen_rust::llama::*;
use autogen_rust::FormatterWrapper;
use autogen_rust::{immutable_agent::*, llama::Content};
use autogen_rust::{
    TEMPLATE_SYSTEM_PROMPT_PLANNER, TEMPLATE_SYSTEM_PROMPT_TOOL_USE, TEMPLATE_USER_PROMPT_TASK_JSON,
};
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

    // let planner_agent_id = spawn_agent(
    //     router_ref.clone(),
    //     "You're an AI assistant".to_string(),
    //     TopicId::from("chat"),
    //     None,
    // )
    // .await?;

    let description = "planner agent".to_string();

    let planner_agent_id = spawn_agent(
        router_ref.clone(),
        TEMPLATE_SYSTEM_PROMPT_PLANNER.to_string(),
        Some(TEMPLATE_USER_PROMPT_TASK_JSON.clone()),
        TopicId::from("chat"),
        None,
        description,
    )
    .await?;
    println!("Task agent id: {}", planner_agent_id);

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
            "<|im_start|>how to build a 30W music amplifier<|im_end|>".to_string(),
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
        agent_id: planner_agent_id,
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
    description: String,
) -> Result<AgentId> {
    let spawn_response = router_ref
        .call(
            |reply_to| RouterCommand::SpawnAgent {
                system_prompt,
                user_prompt_formatter,
                topic,
                tools_map_meta,
                description,
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
