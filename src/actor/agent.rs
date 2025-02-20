use crate::actor::{ActorContext, AgentId, MessageContext, RouterCommand, TopicId};
use crate::immutable_agent::{LlmAgent, Message};
use crate::llama_structs::Content;
use async_openai::types::Role;
use ractor::{Actor, ActorProcessingErr, ActorRef};
use std::fmt;
use uuid::Uuid;

#[derive(Debug, Clone, PartialEq)]
pub enum AgentState {
    Ready,
    Processing,
    Off,
}

impl AgentState {
    pub fn new() -> Self {
        Self::Ready
    }
}

#[derive(Debug)]
struct ShutdownError(String);

impl fmt::Display for ShutdownError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl std::error::Error for ShutdownError {}

pub struct AgentActor {
    agent_id: AgentId,
    router: ActorRef<RouterCommand>,
    subscribed_topics: Vec<TopicId>,
    state: AgentState,
    context: ActorContext,
    llm: LlmAgent,
}

impl AgentActor {
    pub fn new(agent_id: AgentId, router: ActorRef<RouterCommand>, llm: LlmAgent) -> Self {
        Self {
            agent_id,
            router,
            subscribed_topics: Vec::new(),
            state: AgentState::Ready,
            context: ActorContext::new().with_sender(agent_id),
            llm,
        }
    }
}

// #[ractor::async_trait]
impl Actor for AgentActor {
    type Msg = RouterCommand;
    type State = AgentState;
    type Arguments = (AgentId, ActorRef<RouterCommand>, LlmAgent);

    async fn pre_start(
        &self,
        myself: ActorRef<Self::Msg>,
        args: Self::Arguments,
    ) -> Result<Self::State, ActorProcessingErr> {
        Ok(AgentState::new())
    }

    async fn handle(
        &self,
        myself: ActorRef<Self::Msg>,
        msg: Self::Msg,
        state: &mut Self::State,
    ) -> Result<(), ActorProcessingErr> {
        match (msg, &state) {
            (RouterCommand::Ready, AgentState::Off) => {
                *state = AgentState::Ready;
                Ok(())
            }
            (RouterCommand::Off, AgentState::Ready | AgentState::Processing) => {
                *state = AgentState::Off;
                Ok(())
            }
            (
                RouterCommand::RouteMessage {
                    message,
                    topic,
                    context,
                },
                AgentState::Ready,
            ) => {
                if context.sender == Some(self.agent_id) {
                    return Ok(());
                }

                *state = AgentState::Processing;
                let input = message.content.content_to_string();

                println!("Agent {} processing message: {:?}", self.agent_id, input);

                if let Ok(response) = self.llm.default_method(&input).await {
                    println!("LLM response: {:?}", response);

                    let route_msg = RouterCommand::RouteMessage {
                        topic: topic.clone(),
                        message: Message::new(
                            Content::Text(response.content.content_to_string()),
                            None,
                            Role::Assistant,
                        ),
                        context: self.context.clone(), // Use agent's context
                    };
                    self.router
                        .send_message(route_msg)
                        .map_err(|e| Box::new(e) as Box<dyn std::error::Error + Send + Sync>)?;

                    *state = AgentState::Ready;
                    Ok(())
                } else {
                    Err(Box::new(std::io::Error::new(
                        std::io::ErrorKind::Other,
                        "Failed to process LLM response",
                    ))) // ... error handling
                }
            }

            (RouterCommand::ShutdownAgent { agent_id }, _) if agent_id == self.agent_id => {
                *state = AgentState::Off;
                Ok(())
            }
            _ => Ok(()),
        }
    }
}
