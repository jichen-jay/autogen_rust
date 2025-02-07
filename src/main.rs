#![allow(warnings, deprecated)]

use anyhow::Result;
use async_openai::types::Role;
use autogen_rust::actor::{
    ActorContext, AgentActor, AgentState, MessageContext, MessageEnvelope, RouterActor,
    RouterMessage,
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
        agent_tokens: HashMap::new(),
        context: ActorContext::new(),
        llm: None,
    };

    let (router_actor, router_handle) = Actor::spawn(None, router_actor_obj, ()).await?;

    let agent_id = Uuid::new_v4();

    let llm_agent = LlmAgent::new("You are a helpful assistant".to_string(), None);

    let agent_actor_obj = AgentActor {
        agent_id,
        router: router_actor.clone(),
        subscribed_topics: Vec::new(),
        state: AgentState::Ready,
        context: ActorContext::new(),
        llm: llm_agent.clone(),
    };

    let (agent_actor, agent_handle) = Actor::spawn_linked(
        None,
        agent_actor_obj,
        (agent_id, router_actor.clone(), llm_agent),
        router_actor.clone().into(),
    )
    .await?;

    let state_update = Box::new(move |router: &mut RouterActor| {
        router.subscribe_agent(agent_id, "chat".to_string());
    });

    let msg_envelope = MessageEnvelope::new(
        MessageContext::new().with_sender(agent_id),
        RouterMessage::UpdateState(state_update),
    );

    router_actor.send_message(msg_envelope)?;

    let hello_msg = Message::new(
        Content::Text("calculate 1 + 2".to_string()),
        None,
        Role::User,
    );

    let msg_envelope = MessageEnvelope::new(
        MessageContext::new()
            .with_sender(agent_id)
            .with_topic("chat".to_string()),
        RouterMessage::RouteMessage(hello_msg),
    );

    router_actor.send_message(msg_envelope)?;

    tokio::time::sleep(std::time::Duration::from_secs(15)).await;

    let shutdown_msg = MessageEnvelope {
        context: MessageContext::new().with_sender(agent_id),
        payload: RouterMessage::ShutdownAgent(agent_id),
    };
    router_actor.send_message(shutdown_msg)?;

    let _ = agent_handle.await;

    return Ok(());
}
