use crate::actor::{
    ActorContext, AgentId, AgentMessage, AgentState, MessageContext, MessageEnvelope, RouterActor,
    RouterMessage, TopicId,
};
use log::debug;
use ractor::{Actor, ActorProcessingErr, ActorRef};
use std::collections::HashMap;
use std::time::SystemTime;

impl Actor for RouterActor {
    type Msg = MessageEnvelope<RouterMessage>;
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
        myself: ActorRef<Self::Msg>,
        envelope: Self::Msg,
        state: &mut Self::State,
    ) -> Result<(), ActorProcessingErr> {
        state.context.timestamp = SystemTime::now();
        let sender_id = envelope.context.sender;
        match envelope.payload {
            RouterMessage::UpdateState(state_fn) => {
                println!(
                    "RouterActor received update_state message: {:?}",
                    state.context
                );

                state_fn(state);
            }
            RouterMessage::RegisterAgent(actor_ref) => {
                if let Some(agent_id) = sender_id {
                    println!(
                        "RouterActor received register message from agent: {:?}",
                        agent_id
                    );

                    state.agents.insert(agent_id, actor_ref.clone());
                    state.agent_states.insert(agent_id, AgentState::Ready);
                    state.agent_subscriptions.insert(agent_id, Vec::new());
                }
            }

            RouterMessage::RouteMessage(msg) => {
                if let Some(topic_id) = envelope.context.topic_id {
                    println!("RouterActor received routeMessage on topic: {:?}", topic_id);

                    if let Some(subscribed_agents) = state.topic_subscriptions.get(&topic_id) {
                        println!(
                            "RouterActor received routeMessage addressed to : {:?}",
                            subscribed_agents.clone()
                        );
                        for agent_id in subscribed_agents {
                            if let Some(agent_ref) = state.agents.get(agent_id) {
                                let context = MessageContext::new()
                                    .with_sender(*agent_id)
                                    .with_topic(topic_id.clone());
                                let agent_msg = MessageEnvelope::new(
                                    context,
                                    AgentMessage::Process(msg.clone()),
                                );
                                agent_ref.send_message(agent_msg)?;
                            }
                        }
                    }
                }
            }

            RouterMessage::InternalBroadcast(topic_id, msg) => {
                if let Some(subscribed_agents) = state.topic_subscriptions.get(&topic_id) {
                    for agent_id in subscribed_agents {
                        if let Some(agent_ref) = state.agents.get(agent_id) {
                            let context = MessageContext::new()
                                .with_sender(*agent_id)
                                .with_topic(topic_id.clone());

                            let agent_msg =
                                MessageEnvelope::new(context, AgentMessage::Process(msg.clone()));
                            agent_ref.send_message(agent_msg)?;
                        }
                    }
                }
            }
        }

        Ok(())
    }
}

impl RouterActor {
    pub fn subscribe_agent(&mut self, agent_id: AgentId, topic_id: TopicId) {
        self.topic_subscriptions
            .entry(topic_id.clone())
            .or_insert_with(Vec::new)
            .push(agent_id);

        if let Some(agent_topics) = self.agent_subscriptions.get_mut(&agent_id) {
            agent_topics.push(topic_id);
        }
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
