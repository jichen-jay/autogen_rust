#![allow(warnings, deprecated)]

use anyhow::Result;
use async_openai::types::Role;
use autogen_rust::actor::{
    ActorContext, AgentActor, AgentState, MessageContext, MessageEnvelope, RouterActor,
};
use autogen_rust::llama_structs::output_llama_response;
use autogen_rust::llm_utils::*;
use autogen_rust::{immutable_agent::*, llama_structs::Content};
use env_logger;
use ractor::{Actor, ActorRef};
use std::collections::HashMap;
use std::marker::PhantomData;
use tokio_util::sync::CancellationToken;
use uuid::Uuid;

#[tokio::main]
async fn main() -> Result<()> {
    dotenv::dotenv().ok();
    env_logger::init();

    let router_actor_obj = RouterActor {
        agents: HashMap::new(),
        topic_subscriptions: HashMap::new(),
        agent_subscriptions: HashMap::new(),
        agent_states: HashMap::new(),
        context: ActorContext::new(),
        llm: None,
    };

    let (router_actor, _router_handle) = Actor::spawn(None, router_actor_obj, ()).await?;

    let agent_id = router_actor
        .spawn_agent("You're an AI assistant", "chat")
        .await?;

    let hello_msg = Message::new(
        Content::Text("calculate 1 + 2".to_string()),
        None,
        Role::User,
    );

    router_actor.route_message("chat", hello_msg)?;

    tokio::time::sleep(std::time::Duration::from_secs(15)).await;

    router_actor.shutdown_agent(agent_id)?;

    Ok(())
}
