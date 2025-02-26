use crate::agent_runtime::{ActorContext, AgentId, MessageContext, RouterCommand, TopicId};
use crate::immutable_agent::{LlmAgent, Message};
use crate::llama::Content;
use crate::llama::LlamaResponseMessage;
use async_openai::types::Role;
use ractor::{Actor, ActorProcessingErr, ActorRef, MessagingErr};
use std::fmt;
use thiserror::Error;
use uuid::Uuid;

#[derive(Debug, Clone, PartialEq)]
pub enum ProcessingState {
    Ready,
    Processing,
    Off,
}

#[derive(Debug, Clone)]
pub struct AgentState {
    processing_state: ProcessingState,
    subscribed_topics: Vec<TopicId>,
    context: ActorContext,
}

impl AgentState {
    pub fn new(agent_id: AgentId) -> Self {
        Self {
            processing_state: ProcessingState::Ready,
            subscribed_topics: Vec::new(),
            context: ActorContext::new().with_sender(agent_id),
        }
    }

    pub fn add_topic(&mut self, topic: TopicId) {
        self.subscribed_topics.push(topic.clone());
        self.context = self.context.clone().with_topic(topic);
    }

    pub fn remove_topic(&mut self, topic: &TopicId) {
        self.subscribed_topics.retain(|t| t != topic);
        if self.context.topic_id.as_ref() == Some(topic) {
            self.context = self
                .context
                .clone()
                .with_topic(self.subscribed_topics.last().cloned().unwrap_or_default());
        }
    }

    pub fn get_context(&self) -> ActorContext {
        self.context.clone()
    }
}

#[derive(Debug, Error)]
pub enum AgentActorError {
    #[error("Router communication failure: {0}")]
    RouterCommunication(#[from] MessagingErr<RouterCommand>),

    #[error("LLM processing error: {0}")]
    LlmProcessing(#[source] Box<dyn std::error::Error + Send + Sync>),

    #[error("Shutdown failed: {0}")]
    ShutdownFailure(String),
}

pub struct AgentActor {
    agent_id: AgentId,
    router: ActorRef<RouterCommand>,
    llm: LlmAgent,
}

impl AgentActor {
    pub fn new(agent_id: AgentId, router: ActorRef<RouterCommand>, llm: LlmAgent) -> Self {
        Self {
            agent_id,
            router,
            llm,
        }
    }
}

impl Actor for AgentActor {
    type Msg = RouterCommand;
    type State = AgentState;
    type Arguments = (AgentId, ActorRef<RouterCommand>, LlmAgent);

    async fn pre_start(
        &self,
        _myself: ActorRef<Self::Msg>,
        args: Self::Arguments,
    ) -> Result<Self::State, ActorProcessingErr> {
        Ok(AgentState::new(args.0))
    }

    async fn handle(
        &self,
        _myself: ActorRef<Self::Msg>,
        msg: Self::Msg,
        state: &mut Self::State,
    ) -> Result<(), ActorProcessingErr> {
        match (msg, &state.processing_state) {
            (RouterCommand::Ready, ProcessingState::Off) => {
                state.processing_state = ProcessingState::Ready;
                Ok(())
            }
            (RouterCommand::Off, ProcessingState::Ready | ProcessingState::Processing) => {
                state.processing_state = ProcessingState::Off;
                Ok(())
            }
            (
                RouterCommand::RouteMessage {
                    message,
                    topic,
                    context,
                },
                ProcessingState::Ready,
            ) => {
                if context.sender == Some(self.agent_id) {
                    return Ok(());
                }
                state.processing_state = ProcessingState::Processing;
                let input = message.content.content_to_string();

                println!("Agent {} processing message: {:?}", self.agent_id, input);

                match self.llm.default_method(&input).await {
                    Ok(llama_response) => {
                        println!("LLM response (Llama): {:?}", llama_response);
                        let route_msg = RouterCommand::RouteMessage {
                            topic: topic.clone(),
                            message: Message::new(
                                Content::Text(llama_response.content.content_to_string()),
                                None,
                                Role::Assistant,
                            ),
                            context: state.get_context(),
                        };
                        self.router
                            .send_message(route_msg)
                            .map_err(AgentActorError::from)?;

                        state.processing_state = ProcessingState::Ready;
                        Ok(())
                    }
                    Err(e) => Err(Box::new(AgentActorError::LlmProcessing(e.into()))),
                }
            }

            (RouterCommand::ShutdownAgent { agent_id }, _) => {
                if agent_id != self.agent_id {
                    return Err(Box::new(AgentActorError::ShutdownFailure(agent_id.into())));
                }

                match state.processing_state {
                    ProcessingState::Off => {
                        Err(Box::new(AgentActorError::ShutdownFailure(agent_id.into())))
                    }
                    _ => {
                        state.processing_state = ProcessingState::Off;
                        Ok(())
                    }
                }
            }

            (RouterCommand::SubscribeAgent { topic, .. }, _) => {
                state.add_topic(topic);
                Ok(())
            }

            (RouterCommand::UnsubscribeAgent { topic, .. }, _) => {
                state.remove_topic(&topic);
                Ok(())
            }

            _ => Ok(()),
        }
    }
}
