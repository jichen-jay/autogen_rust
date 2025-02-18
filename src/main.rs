#![allow(warnings, deprecated)]

use anyhow::Result;
use async_openai::types::Role;
use autogen_rust::actor::{
    agent::{AgentActor, AgentState},
    router::{RouterActor, RouterState, RouterStateData},
    ActorContext, AgentId, MessageContext, RouterCommand,
};
use autogen_rust::llama_structs::output_llama_response;
use autogen_rust::llm_utils::*;
use autogen_rust::{immutable_agent::*, llama_structs::Content};
use env_logger;
use ractor::{rpc::CallResult, spawn_named, Actor, ActorCell, ActorRef};
use std::collections::HashMap;
use std::marker::PhantomData;
use tokio_util::sync::CancellationToken;
use uuid::Uuid;

#[tokio::main]
async fn main() -> Result<()> {
    dotenv::dotenv().ok();
    env_logger::init();
    std::env::set_var("RUST_LOG", "debug");
    let router_actor = RouterActor::default();

    // Spawn the actor with the default reference
    let (router_ref, _handle) = Actor::spawn(
        Some("router".to_string()),
        router_actor,
        (),
    ).await?;

    router_ref.cast(RouterCommand::Ready)?;

    router_ref.cast(RouterCommand::SpawnAgent {
        system_prompt: "You're an AI assistant".to_string(),
        topic: "chat".to_string(),
    })?;

    tokio::time::sleep(std::time::Duration::from_secs(1)).await;

    let hello_msg = Message::new(
        Content::Text("calculate 1 + 2".to_string()),
        None,
        Role::User,
    );

    router_ref.cast(RouterCommand::RouteMessage {
        topic: "chat".to_string(),
        message: hello_msg,
    })?;

    tokio::time::sleep(std::time::Duration::from_secs(15)).await;

    router_ref.cast(RouterCommand::Off)?;

    tokio::time::sleep(std::time::Duration::from_secs(1)).await;

    Ok(())
}
