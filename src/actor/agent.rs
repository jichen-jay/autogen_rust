use crate::actor::{ActorContext, AgentId, MessageContext, RouterCommand, TopicId};
use crate::immutable_agent::{AgentResponse, LlmAgent, Message};
use crate::llama_structs::Content;
use async_openai::types::Role;
use ractor::{Actor, ActorProcessingErr, ActorRef};
use std::fmt;
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
                    Ok(agent_response) => {
                        match agent_response {
                            AgentResponse::Llama(llama_response) => {
                                //code entered this branch, but I intended that llm agent will check if this is a toolcall, 
                                //if yes, it'll go execute the function and obtain the result and then pass it on
                                //the said logic is in default_method in llm_agent.rs
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
                                self.router.send_message(route_msg).map_err(|e| {
                                    Box::new(e) as Box<dyn std::error::Error + Send + Sync>
                                })?;
                            }
                            AgentResponse::Proxy(proxy_str) => {
                                println!("LLM response (Proxy): {:?}", proxy_str);
                                let route_msg = RouterCommand::RouteMessage {
                                    topic: topic.clone(),
                                    message: Message::new(
                                        Content::Text(proxy_str),
                                        None,
                                        Role::Assistant,
                                    ),
                                    context: state.get_context(),
                                };
                                self.router.send_message(route_msg).map_err(|e| {
                                    Box::new(e) as Box<dyn std::error::Error + Send + Sync>
                                })?;
                            }
                        }
                        state.processing_state = ProcessingState::Ready;
                        Ok(())
                    }
                    Err(_e) => Err(Box::new(std::io::Error::new(
                        std::io::ErrorKind::Other,
                        "Failed to process LLM response",
                    ))),
                }
            }

            (RouterCommand::ShutdownAgent { agent_id }, _) if agent_id == self.agent_id => {
                state.processing_state = ProcessingState::Off;
                Ok(())
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
