//this project is being transformed
//previous iteration used more nested Message structure for agent and router
//here, the mid layer wrapper is removed, both agent and router handle llm Message directly
//Router spawns, shutdown agents directly; when to do it is yet to be fleshed out

pub mod agent;
pub mod router;

use crate::immutable_agent::{LlmAgent, Message};
use ractor::ActorRef;
use std::collections::HashMap;
use std::error::Error;
use std::marker::PhantomData;
use std::time::SystemTime;
use uuid::Uuid;

pub type AgentId = Uuid;
pub type TopicId = String;

#[derive(Debug, Clone)]
pub struct Context<M> {
    pub sender: Option<AgentId>,
    pub topic_id: Option<TopicId>,
    pub timestamp: SystemTime,
    pub marker: PhantomData<M>,
}

impl<M> Context<M> {
    pub fn new() -> Self {
        Self {
            sender: None,
            topic_id: None,
            timestamp: SystemTime::now(),
            marker: PhantomData,
        }
    }

    pub fn with_sender(mut self, sender: AgentId) -> Self {
        self.sender = Some(sender);
        self
    }

    pub fn with_topic(mut self, topic: TopicId) -> Self {
        self.topic_id = Some(topic);
        self
    }
}

#[derive(Debug, Clone)]
pub struct ActorMarker;

#[derive(Debug, Clone)]
pub struct MessageMarker;

pub type ActorContext = Context<ActorMarker>;
pub type MessageContext = Context<MessageMarker>;

#[derive(Debug, Clone)]
pub struct MessageEnvelope<T> {
    pub context: MessageContext,
    pub payload: T,
}

impl<T> MessageEnvelope<T> {
    pub fn new(context: MessageContext, payload: T) -> Self {
        Self { context, payload }
    }
}

#[derive(Debug)]
pub enum AgentState {
    Ready,
    Processing,
    PendingShutdown,
}

pub struct AgentActor {
    pub agent_id: AgentId,
    pub router: ActorRef<MessageEnvelope<Message>>,
    pub subscribed_topics: Vec<TopicId>,
    pub state: AgentState,
    pub context: ActorContext,
    pub llm: LlmAgent,
}

//do we need a RouterState enum or struct also?
//are there additional data to carry other than the RouterActor struct?
pub struct RouterActor {
    pub agents: std::collections::HashMap<AgentId, ActorRef<MessageEnvelope<Message>>>,
    pub topic_subscriptions: std::collections::HashMap<TopicId, Vec<AgentId>>,
    pub agent_subscriptions: std::collections::HashMap<AgentId, Vec<TopicId>>,
    pub agent_states: std::collections::HashMap<AgentId, AgentState>,
    pub context: ActorContext,
    pub llm: Option<LlmAgent>,
}
