use crate::actor::{
    AgentMessage, RouterActor, RouterMessage, AgentState, 
    ActorContext, Message, AgentId, TopicId
};
use ractor::{Actor, ActorProcessingErr, ActorRef};
use std::collections::HashMap;
use std::time::SystemTime;

impl Actor for RouterActor {
    type Msg = RouterMessage;
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
        })
    }

    async fn handle(
        &self,
        myself: ActorRef<Self::Msg>,
        message: Self::Msg,
        state: &mut Self::State,
    ) -> Result<(), ActorProcessingErr> {
        match message {
            RouterMessage::RegisterAgent(actor_ref) => {
                let agent_id = uuid::Uuid::new_v4();
                state.agents.insert(agent_id, actor_ref.clone());
                state.agent_states.insert(agent_id, AgentState::Ready);
                state.agent_subscriptions.insert(agent_id, Vec::new());
            }
            
            RouterMessage::RouteMessage(msg) => {
                if let Some(subscribed_agents) = state.topic_subscriptions.get(&msg.topic) {
                    for agent_id in subscribed_agents {
                        if let Some(agent_ref) = state.agents.get(agent_id) {
                            if let Some(AgentState::Ready) = state.agent_states.get(agent_id) {
                                agent_ref.send_message(AgentMessage::Process(msg.clone()))?;
                            }
                        }
                    }
                }
            }
            
            RouterMessage::AgentSubscribe(topic_id) => {
                let context = ActorContext {
                    sender: state.agents.keys().next().cloned().unwrap_or_default(),
                    timestamp: SystemTime::now(),
                };
                
                state.topic_subscriptions
                    .entry(topic_id.clone())
                    .or_insert_with(Vec::new)
                    .push(context.sender);
                
                if let Some(agent_topics) = state.agent_subscriptions.get_mut(&context.sender) {
                    agent_topics.push(topic_id);
                }
            }
            
            RouterMessage::AgentUnsubscribe(topic_id) => {
                let context = ActorContext {
                    sender: state.agents.keys().next().cloned().unwrap_or_default(),
                    timestamp: SystemTime::now(),
                };
                
                if let Some(subscribed_agents) = state.topic_subscriptions.get_mut(&topic_id) {
                    subscribed_agents.retain(|id| *id != context.sender);
                }
                
                if let Some(agent_topics) = state.agent_subscriptions.get_mut(&context.sender) {
                    agent_topics.retain(|t| t != &topic_id);
                }
            }
            
            RouterMessage::InternalBroadcast(topic_id, msg) => {
                if let Some(subscribed_agents) = state.topic_subscriptions.get(&topic_id) {
                    for agent_id in subscribed_agents {
                        if let Some(agent_ref) = state.agents.get(agent_id) {
                            agent_ref.send_message(AgentMessage::Process(msg.clone()))?;
                        }
                    }
                }
            }
        }
        Ok(())
    }
}
