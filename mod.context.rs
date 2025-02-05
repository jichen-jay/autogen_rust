// mod.rs
pub mod agent;
pub mod router;

use std::collections::HashMap;
use uuid::Uuid;
use std::time::SystemTime;

#[derive(Clone, Debug)]
pub struct ActorContext {
    pub sender: AgentId,
    pub timestamp: SystemTime,
    pub trace_id: Uuid,
    pub metadata: HashMap<String, String>,
}

impl ActorContext {
    pub fn new(sender: AgentId) -> Self {
        Self {
            sender,
            timestamp: SystemTime::now(),
            trace_id: Uuid::new_v4(),
            metadata: HashMap::new(),
        }
    }

    pub fn with_metadata(mut self, key: &str, value: &str) -> Self {
        self.metadata.insert(key.to_string(), value.to_string());
        self
    }
}

#[derive(Debug)]
pub struct MessageEnvelope<T> {
    pub context: ActorContext,
    pub payload: T,
}

impl<T> MessageEnvelope<T> {
    pub fn new(context: ActorContext, payload: T) -> Self {
        Self { context, payload }
    }
}

// Updated message types
pub type EnvelopedRouterMessage = MessageEnvelope<RouterMessage>;
pub type EnvelopedAgentMessage = MessageEnvelope<AgentMessage>;

// agent.rs
impl Actor for AgentActor {
    type Msg = EnvelopedAgentMessage;
    type State = AgentActor;
    type Arguments = (AgentId, ActorRef<EnvelopedRouterMessage>);

    async fn pre_start(
        &self,
        myself: ActorRef<Self::Msg>,
        args: Self::Arguments,
    ) -> Result<Self::State, ActorProcessingErr> {
        let (id, router) = args;
        let context = ActorContext::new(id);
        
        let register_msg = MessageEnvelope::new(
            context,
            RouterMessage::RegisterAgent(myself.clone())
        );
        router.send_message(register_msg)?;

        Ok(AgentActor {
            agent_id: id,
            router: router.clone(),
            subscribed_topics: Vec::new(),
            state: AgentState::Ready,
        })
    }

    async fn handle(
        &self,
        myself: ActorRef<Self::Msg>,
        envelope: Self::Msg,
        state: &mut Self::State,
    ) -> Result<(), ActorProcessingErr> {
        match envelope.payload {
            AgentMessage::Process(content) => {
                let new_envelope = MessageEnvelope::new(
                    envelope.context,
                    RouterMessage::RouteMessage(content)
                );
                state.router.send_message(new_envelope)?;
            },
            // ... other message handling
        }
        Ok(())
    }
}
