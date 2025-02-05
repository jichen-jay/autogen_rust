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
            agent_id: Uuid::new_v4(),
            router: router.clone(),
            subscribed_topics: Vec::new(),
        })
    }

    //being the media to hold comms data, Msg, State shares responsibilities
    //want to find a reasonable task split between them
    //Msg is supposed to be the main vehicle of agent's raw input output
    //State is supposed to show the condition the Actor is in, want it to hold the subscription
    //status of agents, the agent's status - ready to work, pending-shutdown, propose reasonable ways
    //to distribute the job
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
