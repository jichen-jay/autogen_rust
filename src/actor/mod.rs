pub mod agent;
pub mod router;

use ractor::ActorRef;
use std::error::Error;
use std::marker::PhantomData;
use std::time::SystemTime;
use uuid::Uuid;
pub type AgentId = Uuid;
pub type TopicId = String;

use crate::immutable_agent::Message;

#[derive(Debug, Clone)]
pub struct ActorContext {
    pub sender: Option<AgentId>,
    pub topic_id: Option<TopicId>,
    pub cancellation_token: PhantomData<()>,
    pub timestamp: SystemTime,
}

impl ActorContext {
    pub fn new() -> Self {
        Self {
            sender: None,
            topic_id: None,
            cancellation_token: PhantomData,
            timestamp: SystemTime::now(),
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
pub struct MessageEnvelope<T> {
    pub context: ActorContext,
    pub payload: T,
}

impl<T> MessageEnvelope<T> {
    pub fn new(context: ActorContext, payload: T) -> Self {
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

#[derive(Debug)]
pub enum RouterMessage {
    RegisterAgent(ActorRef<MessageEnvelope<AgentMessage>>),
    RouteMessage(Message),
    InternalBroadcast(TopicId, Message),
}

#[derive(Debug)]
pub enum AgentMessage {
    Process(Message),
    UpdateState(AgentState),
}

pub struct AgentActor {
    pub agent_id: AgentId,
    pub router: ActorRef<MessageEnvelope<RouterMessage>>,
    pub subscribed_topics: Vec<TopicId>,
    pub state: AgentState,
    pub context: ActorContext,
}

pub struct RouterActor {
    pub agents: std::collections::HashMap<AgentId, ActorRef<MessageEnvelope<AgentMessage>>>,
    pub topic_subscriptions: std::collections::HashMap<TopicId, Vec<AgentId>>,
    pub agent_subscriptions: std::collections::HashMap<AgentId, Vec<TopicId>>,
    pub agent_states: std::collections::HashMap<AgentId, AgentState>,
    pub context: ActorContext,
}
