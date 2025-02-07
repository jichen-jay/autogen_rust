pub mod agent;
pub mod router;

use crate::immutable_agent::{LlmAgent, Message};
use ractor::ActorRef;
use std::collections::HashMap;
use std::error::Error;
use std::marker::PhantomData;
use std::time::SystemTime;
use tokio_util::sync::CancellationToken;
use uuid::Uuid;

pub type AgentId = Uuid;
pub type TopicId = String;

#[derive(Debug, Clone)]
pub struct Context<M> {
    pub sender: Option<AgentId>,
    pub topic_id: Option<TopicId>,
    pub timestamp: SystemTime,
    pub cancellation_token: CancellationToken, // CHANGED: Now a real token
    pub marker: PhantomData<M>,
}

impl<M> Context<M> {
    pub fn new() -> Self {
        Self {
            sender: None,
            topic_id: None,
            timestamp: SystemTime::now(),
            cancellation_token: CancellationToken::new(), // CHANGED
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
    Error(Box<dyn Error + Send + Sync>),
}

pub enum RouterMessage {
    RegisterAgent {
        agent: ActorRef<MessageEnvelope<AgentMessage>>,
        cancellation_token: CancellationToken, // CHANGED
    },
    RouteMessage(Message),
    InternalBroadcast(TopicId, Message),
    UpdateState(Box<dyn FnOnce(&mut RouterActor) + Send>),
    ShutdownAgent(AgentId), // CHANGED: New variant to shutdown an agent.
}

impl std::fmt::Debug for RouterMessage {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            RouterMessage::RegisterAgent { agent, .. } => {
                f.debug_tuple("RegisterAgent").field(agent).finish()
            }
            RouterMessage::RouteMessage(ref msg) => {
                f.debug_tuple("RouteMessage").field(msg).finish()
            }
            RouterMessage::InternalBroadcast(ref topic, ref msg) => f
                .debug_tuple("InternalBroadcast")
                .field(topic)
                .field(msg)
                .finish(),
            RouterMessage::UpdateState(_) => f.write_str("UpdateState(<closure>)"),
            RouterMessage::ShutdownAgent(agent_id) => {
                f.debug_tuple("ShutdownAgent").field(agent_id).finish() // CHANGED
            }
        }
    }
}

#[derive(Debug)]
pub enum AgentMessage {
    Process(Message),
    UpdateState(AgentState),
    Shutdown, // CHANGED: New variant to signal the agent to shutdown.
}

pub struct AgentActor {
    pub agent_id: AgentId,
    pub router: ActorRef<MessageEnvelope<RouterMessage>>,
    pub subscribed_topics: Vec<TopicId>,
    pub state: AgentState,
    pub context: ActorContext,
    pub llm: LlmAgent,
}

pub struct RouterActor {
    pub agents: std::collections::HashMap<AgentId, ActorRef<MessageEnvelope<AgentMessage>>>,
    pub topic_subscriptions: std::collections::HashMap<TopicId, Vec<AgentId>>,
    pub agent_subscriptions: std::collections::HashMap<AgentId, Vec<TopicId>>,
    pub agent_states: std::collections::HashMap<AgentId, AgentState>,
    pub agent_tokens: HashMap<AgentId, CancellationToken>, // CHANGED: Store tokens here.
    pub context: ActorContext,
    pub llm: Option<LlmAgent>,
}
