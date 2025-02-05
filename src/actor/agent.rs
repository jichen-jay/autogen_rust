use crate::actor::{AgentActor, AgentMessage, RouterMessage};
use ractor::{Actor, ActorProcessingErr, ActorRef};
use uuid::Uuid;

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
        router.send_message(RouterMessage::RegisterAgent(id, myself.clone()))?;

        Ok(AgentActor {
            id,
            router: router.clone(),
        })
    }

    async fn handle(
        &self,
        _myself: ActorRef<Self::Msg>,
        message: Self::Msg,
        state: &mut Self::State,
    ) -> Result<(), ActorProcessingErr> {
        state
            .router
            .send_message(RouterMessage::RouteMessage(state.id, message.0))?;
        Ok(())
    }
}
