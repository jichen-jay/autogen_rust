#![allow(warnings, deprecated)]

use anyhow::anyhow;
use anyhow::Result;
use async_openai::types::Role;
use autogen_rust::actor::{
    agent::{AgentActor, AgentState},
    router::{RouterActor, RouterState, RouterStateData},
    ActorContext, AgentId, MessageContext, RouterCommand, SpawnAgentResponse, TopicId,
};
use autogen_rust::llama_structs::output_llama_response;
use autogen_rust::llm_utils::*;
use autogen_rust::{immutable_agent::*, llama_structs::Content};
use env_logger;
use ractor::{call_t, rpc::CallResult, spawn_named, Actor, ActorCell, ActorRef, RpcReplyPort};
use std::collections::HashMap;
use std::marker::PhantomData;
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

    let task_agent_id = spawn_agent(
        router_ref.clone(),
        "You're an AI assistant".to_string(),
        TopicId::from("chat"),
    )
    .await?;
    println!("Task agent id: {}", task_agent_id);

    // let judge_agent_id = spawn_agent(
    //     router_ref.clone(),
    //     "You're an AI judge, you'll be given a task/solution pair, reply yes if you think the solution is correct, reply no otherwise".to_string(),
    //     TopicId::from("judge"),
    // )
    // .await?;
    // println!("Judge agent id: {}", judge_agent_id);

    time::sleep(std::time::Duration::from_secs(1)).await;

    let task_message = Message::new(
        Content::Text("calculate 1 + 2".to_string()),
        None,
        Role::User,
    );
    router_ref.cast(RouterCommand::RouteMessage {
        topic: "chat".to_string(),
        message: task_message,
    })?;

    time::sleep(std::time::Duration::from_secs(3)).await;

    // let task_solution_message = Message::new(
    //     Content::Text("Task: calculate 1 + 2\nSolution: 3".to_string()),
    //     None,
    //     Role::Assistant,
    // );
    // router_ref.cast(RouterCommand::RouteMessage {
    //     topic: "judge".to_string(),
    //     message: task_solution_message,
    // })?;

    // time::sleep(std::time::Duration::from_secs(3)).await;
    // let judge_reply_message = Message::new(Content::Text("yes".to_string()), None, Role::Assistant);
    // router_ref.cast(RouterCommand::RouteMessage {
    //     topic: "judge".to_string(),
    //     message: judge_reply_message,
    // })?;

    // println!("Judge Agent confirmed the solution as correct. Shutting down Task and Judge Agents.");
    router_ref.cast(RouterCommand::ShutdownAgent {
        agent_id: task_agent_id,
    })?;
    // router_ref.cast(RouterCommand::ShutdownAgent {
    //     agent_id: judge_agent_id,
    // })?;

    router_ref.cast(RouterCommand::Off)?;
    
    time::sleep(std::time::Duration::from_secs(1)).await;

    Ok(())
}

async fn spawn_agent(
    router_ref: ActorRef<RouterCommand>,
    system_prompt: String,
    topic: TopicId,
) -> Result<AgentId> {
    let spawn_response = router_ref
        .call(
            |reply_to| RouterCommand::SpawnAgent {
                system_prompt,
                topic,
                reply_to,
            },
            None, // Optional timeout can be passed here if needed
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
