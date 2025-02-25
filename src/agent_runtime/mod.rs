pub mod agent;
pub mod router;

use crate::immutable_agent::{LlmAgent, Message};
use crate::FormatterWrapper;
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

#[derive(Default)]
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
        user_prompt_formatter: Option<FormatterWrapper>,
        topic: TopicId,
        reply_to: RpcReplyPort<SpawnAgentResponse>,
        tools_map_meta: Option<Value>,
    },
}

pub type SpawnAgentResponse = Result<AgentId, String>;

impl std::fmt::Debug for RouterCommand {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            RouterCommand::Off => f.debug_tuple("Off").finish(),
            RouterCommand::Ready => f.debug_tuple("Ready").finish(),
            RouterCommand::RouteMessage {
                topic,
                message,
                context,
            } => f
                .debug_struct("RouteMessage")
                .field("topic", topic)
                .field("message", message)
                .field("context", context)
                .finish(),
            RouterCommand::ShutdownAgent { agent_id } => f
                .debug_struct("ShutdownAgent")
                .field("agent_id", agent_id)
                .finish(),
            RouterCommand::SubscribeAgent { agent_id, topic } => f
                .debug_struct("SubscribeAgent")
                .field("agent_id", agent_id)
                .field("topic", topic)
                .finish(),
            RouterCommand::UnsubscribeAgent { agent_id, topic } => f
                .debug_struct("UnsubscribeAgent")
                .field("agent_id", agent_id)
                .field("topic", topic)
                .finish(),
            RouterCommand::SpawnAgent {
                system_prompt,
                user_prompt_formatter: _,
                topic,
                reply_to,
                tools_map_meta,
            } => {
                f.debug_struct("SpawnAgent")
                    .field("system_prompt", system_prompt)
                    // Skip the formatter field entirely
                    .field("topic", topic)
                    .field("reply_to", reply_to)
                    .field("tools_map_meta", tools_map_meta)
                    .finish()
            }
        }
    }
}
