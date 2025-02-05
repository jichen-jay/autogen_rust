pub mod agent;
pub mod router;

use std::error::Error;
use uuid::Uuid;
use ractor::ActorRef;
use crate::immutable_agent::Message;

pub type AgentId = Uuid;
pub type TopicId = String;

#[derive(Debug, Clone)]
pub struct ActorContext {
    pub sender: AgentId,
    pub timestamp: std::time::SystemTime,
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
    RegisterAgent(ActorRef<AgentMessage>),
    RouteMessage(Message),
    AgentSubscribe(TopicId),
    AgentUnsubscribe(TopicId),
    InternalBroadcast(TopicId, Message),
}

#[derive(Debug)]
pub enum AgentMessage {
    Process(Message),
    UpdateState(AgentState),
    Subscribe(TopicId),
    Unsubscribe(TopicId),
}

pub struct AgentActor {
    pub agent_id: AgentId,
    pub router: ActorRef<RouterMessage>,
    pub subscribed_topics: Vec<TopicId>,
    pub state: AgentState,
}

pub struct RouterActor {
    pub agents: std::collections::HashMap<AgentId, ActorRef<AgentMessage>>,
    pub topic_subscriptions: std::collections::HashMap<TopicId, Vec<AgentId>>,
    pub agent_subscriptions: std::collections::HashMap<AgentId, Vec<TopicId>>,
    pub agent_states: std::collections::HashMap<AgentId, AgentState>,
}
