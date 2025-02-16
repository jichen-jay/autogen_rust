use crate::actor::{
    ActorContext, AgentActor, AgentId, AgentState, Message, MessageContext, MessageEnvelope,
    RouterActor, TopicId,
};
use crate::immutable_agent::LlmAgent;
use log::{debug, error, warn};
use ractor::{Actor, ActorProcessingErr, ActorRef};
use std::collections::HashMap;
use std::time::SystemTime;
use tokio_util::sync::CancellationToken; // CHANGED
use uuid::Uuid;

impl Actor for RouterActor {
    type Msg = MessageEnvelope<Message>;
    type State = RouterActor;
    type Arguments = ();

    async fn pre_start(
        &self,
        _myself: ActorRef<Self::Msg>,
        _: (),
    ) -> Result<Self::State, ActorProcessingErr> {
        Ok(RouterActor {
            agents: HashMap::new(),
            topic_subscriptions: HashMap::new(),
            agent_subscriptions: HashMap::new(),
            agent_states: HashMap::new(),
            context: ActorContext::new(),
            llm: None,
        })
    }

    async fn handle(
        &self,
        _myself: ActorRef<Self::Msg>,
        mut envelope: Self::Msg,
        state: &mut Self::State,
    ) -> Result<(), ActorProcessingErr> {
        state.context.timestamp = SystemTime::now();

        if let Some(ref topic) = envelope.context.topic_id {
            self.route_message(topic.clone(), envelope.payload.clone())?;
        } else {
            warn!(
                "Received message without a topic in context: {:?}",
                envelope
            );
        }
        Ok(())
    }
}

impl RouterActor {
    pub async fn spawn_agent(
        &mut self,
        parent_ref: ActorRef<MessageEnvelope<Message>>, // pass in the parent's actor ref
        system_prompt: &str,
        topic: TopicId,
    ) -> Result<AgentId, ActorProcessingErr> {
        let new_agent_id = Uuid::new_v4();
        let new_llm_agent = LlmAgent::new(system_prompt.to_string(), None);

        let agent_actor_obj = AgentActor {
            agent_id: new_agent_id,
            router: parent_ref.clone().into(),
            subscribed_topics: Vec::new(),
            state: AgentState::Ready,
            context: ActorContext::new().with_sender(new_agent_id),
            llm: new_llm_agent.clone(),
        };

        let (agent_actor, _) = Actor::spawn_linked(
            None,
            agent_actor_obj,
            (new_agent_id, parent_ref.clone(), new_llm_agent),
            parent_ref.clone().into(), // use the passed-in parent's actor ref here
        )
        .await?;

        self.agents.insert(new_agent_id, agent_actor);
        self.agent_states.insert(new_agent_id, AgentState::Ready);
        self.agent_subscriptions.insert(new_agent_id, Vec::new());
        self.subscribe_agent(new_agent_id, topic);

        Ok(new_agent_id)
    }

    pub fn shutdown_agent(&mut self, agent_id: AgentId) -> Result<(), ActorProcessingErr> {
        let agent = match self.agents.get(&agent_id) {
            Some(a) => a.clone(),
            None => return Err("Agent not found during shutdown".into()),
        };

        // Mutate state fields.
        self.agent_states
            .insert(agent_id, AgentState::PendingShutdown);

        // Clone the topics vector to drop the immutable borrow from self.agent_subscriptions.
        if let Some(topics) = self.agent_subscriptions.get(&agent_id).cloned() {
            for topic in topics {
                self.unsubscribe_agent(agent_id, &topic);
            }
        }

        // Now release the immutable borrow before calling mutable methods.
        agent.stop(None);

        self.agents.remove(&agent_id);
        self.agent_states.remove(&agent_id);
        self.agent_subscriptions.remove(&agent_id);

        Ok(())
    }

    pub fn route_message(
        &self,
        topic_id: TopicId,
        message: Message,
    ) -> Result<(), ActorProcessingErr> {
        if let Some(subscribed_agents) = self.topic_subscriptions.get(&topic_id) {
            for agent_id in subscribed_agents {
                if let Some(agent_ref) = self.agents.get(agent_id) {
                    let message_context = MessageContext::new()
                        .with_sender(agent_id.clone())
                        .with_topic(topic_id.clone());
                    let agent_msg = MessageEnvelope::new(message_context, message.clone());
                    agent_ref.send_message(agent_msg)?;
                }
            }
        }
        Ok(())
    }

    pub fn subscribe_agent(&mut self, agent_id: AgentId, topic_id: TopicId) {
        self.topic_subscriptions
            .entry(topic_id.clone())
            .or_insert_with(Vec::new)
            .push(agent_id);
        self.agent_subscriptions
            .entry(agent_id)
            .or_insert_with(Vec::new)
            .push(topic_id);
    }

    pub fn unsubscribe_agent(&mut self, agent_id: AgentId, topic_id: &TopicId) {
        if let Some(subscribed_agents) = self.topic_subscriptions.get_mut(topic_id) {
            subscribed_agents.retain(|id| *id != agent_id);
        }
        if let Some(agent_topics) = self.agent_subscriptions.get_mut(&agent_id) {
            agent_topics.retain(|t| t != topic_id);
        }
    }
}
