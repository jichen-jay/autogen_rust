pub mod agent;
pub mod router;

use crate::immutable_agent::{AgentId, Message, TopicId};
use ractor::ActorRef;

#[derive(Debug)]
pub enum RouterMessage {
    RegisterAgent(AgentId, ActorRef<AgentMessage>),
    RouteMessage(AgentId, Message),
    // AgentSubscribe(AgentId, TopicId),
}

#[derive(Debug)]
pub struct AgentMessage(Message);

pub struct AgentActor {
    pub id: AgentId,
    pub router: ActorRef<RouterMessage>,
}

pub struct RouterActor {
    pub agents: std::collections::HashMap<AgentId, ActorRef<AgentMessage>>,
    pub subscriptions: std::collections::HashMap<AgentId, ActorRef<AgentMessage>>,
}
