pub mod agent;
pub mod router;

use crate::immutable_agent::{LlmAgent, Message};
use ractor::{ActorRef, RpcReplyPort};
use serde_json::Value;
use std::marker::PhantomData;
use std::time::SystemTime;
use uuid::Uuid;

pub type AgentId = Uuid;
pub type TopicId = String;

#[derive(Debug, Clone)]
pub struct Context<M> {
    sender: Option<AgentId>,
    topic_id: Option<TopicId>,
    timestamp: SystemTime,
    marker: PhantomData<M>,
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

#[derive(Debug, Default)]
pub enum RouterCommand {
    #[default]
    Off,
    Ready,
    RouteMessage {
        topic: TopicId,
        message: Message,
        context: ActorContext,
    },
    ShutdownAgent {
        agent_id: AgentId,
    },
    SubscribeAgent {
        agent_id: AgentId,
        topic: TopicId,
    },
    UnsubscribeAgent {
        agent_id: AgentId,
        topic: TopicId,
    },

    SpawnAgent {
        system_prompt: String,
        topic: TopicId,
        reply_to: RpcReplyPort<SpawnAgentResponse>,
        tools_map_meta: Option<Value>,
    },
}

pub type SpawnAgentResponse = Result<AgentId, String>;
