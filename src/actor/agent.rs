use crate::actor::{AgentActor, AgentMessage, RouterMessage, AgentState, ActorContext, Message};
use ractor::{Actor, ActorProcessingErr, ActorRef};
use uuid::Uuid;
use std::time::SystemTime;

impl Actor for AgentActor {
    type Msg = AgentMessage;
    type State = AgentActor;
    type Arguments = (Uuid, ActorRef<RouterMessage>);

    async fn pre_start(
        &self,
        myself: ActorRef<Self::Msg>,
        args: Self::Arguments,
    ) -> Result<Self::State, ActorProcessingErr> {
        let (id, router) = args;
        let context = ActorContext {
            sender: id,
            timestamp: SystemTime::now(),
        };
        
        router.send_message(RouterMessage::RegisterAgent(myself.clone()))?;

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
        message: Self::Msg,
        state: &mut Self::State,
    ) -> Result<(), ActorProcessingErr> {
        let context = ActorContext {
            sender: state.agent_id,
            timestamp: SystemTime::now(),
        };

        match message {
            AgentMessage::Process(content) => {
                state.state = AgentState::Processing;
                state.router.send_message(RouterMessage::RouteMessage(content))?;
                state.state = AgentState::Ready;
            },
            AgentMessage::UpdateState(new_state) => {
                state.state = new_state;
            },
            AgentMessage::Subscribe(topic) => {
                state.subscribed_topics.push(topic.clone());
                state.router.send_message(RouterMessage::AgentSubscribe(topic))?;
            },
            AgentMessage::Unsubscribe(topic) => {
                state.subscribed_topics.retain(|t| t != &topic);
                state.router.send_message(RouterMessage::AgentUnsubscribe(topic))?;
            }
        }
        Ok(())
    }
}
