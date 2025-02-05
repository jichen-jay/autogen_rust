use crate::actor::{AgentMessage, RouterActor, RouterMessage};
use ractor::{Actor, ActorProcessingErr, ActorRef};
use std::collections::HashMap;

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
            subscriptions: HashMap::new(),
        })
    }

    async fn handle(
        &self,
        _myself: ActorRef<Self::Msg>,
        message: Self::Msg,
        state: &mut Self::State,
    ) -> Result<(), ActorProcessingErr> {
        match message {
            RouterMessage::RegisterAgent(id, actor_ref) => {
                state.agents.insert(id, actor_ref);
            }
            RouterMessage::RouteMessage(id, msg) => {
                if let Some(agent_ref) = state.subscriptions.get(&id) {
                    agent_ref.send_message(AgentMessage(msg))?;
                }
            }
        }
        Ok(())
    }
}
