use crate::actor::{
    ActorContext, AgentActor, AgentMessage, AgentState, MessageContext, MessageEnvelope,
    RouterMessage,
};
use crate::immutable_agent::{LlmAgent, Message};
use crate::llama_structs::Content;
use async_openai::types::Role;
use futures::TryFutureExt;
use log::debug;
use ractor::{Actor, ActorProcessingErr, ActorRef};
use std::fmt;
use std::time::SystemTime;
use uuid::Uuid;

#[derive(Debug)]
struct ShutdownError(String);

impl fmt::Display for ShutdownError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl std::error::Error for ShutdownError {}

impl Actor for AgentActor {
    type Msg = MessageEnvelope<AgentMessage>;
    type State = AgentActor;
    type Arguments = (Uuid, ActorRef<MessageEnvelope<RouterMessage>>, LlmAgent);

    async fn pre_start(
        &self,
        myself: ActorRef<Self::Msg>,
        args: Self::Arguments,
    ) -> Result<Self::State, ActorProcessingErr> {
        let (id, router, llm_agent) = args;

        let message_context = MessageContext::new().with_sender(id);
        let actor_context = ActorContext::new().with_sender(id);

        let register_msg = MessageEnvelope::new(
            message_context,
            RouterMessage::RegisterAgent {
                agent: myself.clone(),
                cancellation_token: actor_context.cancellation_token.clone(),
            },
        );
        router.send_message(register_msg)?;

        Ok(AgentActor {
            agent_id: id,
            router: router.clone(),
            subscribed_topics: Vec::new(),
            state: AgentState::Ready,
            context: actor_context,
            llm: llm_agent,
        })
    }

    async fn handle(
        &self,
        myself: ActorRef<Self::Msg>,
        envelope: Self::Msg,
        state: &mut Self::State,
    ) -> Result<(), ActorProcessingErr> {
        let mut context = envelope.context.clone();
        context.timestamp = SystemTime::now();

        match envelope.payload {
            AgentMessage::Process(content) => {
                state.state = AgentState::Processing;
                println!("Agent {} processing message: {:?}", self.agent_id, content);

                let input = content.content.content_to_string();

                let response = self.llm.default_method(&input).await?;
                println!("llm response: {:?}", response);

                let new_context = context.clone(); // Now new_context still has the topic from envelope
                let route_msg = MessageEnvelope::new(
                    new_context,
                    RouterMessage::RouteMessage(Message::new(
                        Content::Text(response),
                        None,
                        Role::Assistant,
                    )),
                );

                state.router.send_message(route_msg)?;

                state.state = AgentState::Ready;
            }

            AgentMessage::UpdateState(new_state) => {
                println!("Agent {} updating state: {:?}", self.agent_id, new_state);
                state.state = new_state;
            }

            AgentMessage::Shutdown => {
                println!(
                    "Agent {} received Shutdown. Cleaning up and shutting down...",
                    self.agent_id
                );
                state.state = AgentState::PendingShutdown;
                return Err(Box::new(ShutdownError(
                    "Agent shutdown requested".to_string(),
                )));
            }
        }

        Ok(())
    }
}
