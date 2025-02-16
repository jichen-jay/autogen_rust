use crate::actor::{ActorContext, AgentActor, AgentState, MessageContext, MessageEnvelope};
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
    type Msg = MessageEnvelope<Message>;
    type State = AgentActor;
    type Arguments = (Uuid, ActorRef<MessageEnvelope<Message>>, LlmAgent);

    async fn pre_start(
        &self,
        _myself: ActorRef<MessageEnvelope<Message>>,
        args: Self::Arguments,
    ) -> Result<Self, ActorProcessingErr> {
        let (id, router, llm_agent) = args;

        Ok(AgentActor {
            agent_id: id,
            router,
            subscribed_topics: Vec::new(),
            state: AgentState::Ready,
            context: ActorContext::new().with_sender(id),
            llm: llm_agent,
        })
    }

    async fn handle(
        &self,
        myself: ActorRef<MessageEnvelope<Message>>,
        envelope: Self::Msg,
        state: &mut Self::State,
    ) -> Result<(), ActorProcessingErr> {
        if let content = envelope.payload {
            state.state = AgentState::Processing;
            println!("Agent {} processing message: {:?}", self.agent_id, content);

            let input = content.content.content_to_string();

            // Check if the agent should be stopped.
            if input.trim().eq_ignore_ascii_case("shutdown")
                || input.trim().eq_ignore_ascii_case("stop")
            {
                println!("Agent {} received shutdown command.", self.agent_id);
                myself.stop(None);
                return Ok(());
            }

            // Process the input using the LLM.
            if let Ok(response) = self.llm.default_method(&input).await {
                println!("LLM response: {:?}", response);

                // Route the LLM's response back to the RouterActor.
                let route_msg = MessageEnvelope::new(
                    envelope.context.clone(),
                    Message::new(Content::Text(response), None, Role::Assistant),
                );
                state.router.send_message(route_msg)?;
                state.state = AgentState::Ready;

                Ok(())
            } else {
                Err("Failed to process LLM response".into())
            }
        } else {
            Err("Empty payload in message".into())
        }
    }
}
