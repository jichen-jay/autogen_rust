use crate::actor::{
    ActorContext, AgentActor, AgentMessage, AgentState, MessageEnvelope, RouterMessage,
};
use ractor::{Actor, ActorProcessingErr, ActorRef};
use std::time::SystemTime;
use uuid::Uuid;

impl Actor for AgentActor {
    type Msg = MessageEnvelope<AgentMessage>;
    type State = AgentActor;
    type Arguments = (Uuid, ActorRef<MessageEnvelope<RouterMessage>>);

    async fn pre_start(
        &self,
        myself: ActorRef<Self::Msg>,
        args: Self::Arguments,
    ) -> Result<Self::State, ActorProcessingErr> {
        let (id, router) = args;

        let context = ActorContext::new().with_sender(id);

        let register_msg = MessageEnvelope::new(
            context.clone(),
            RouterMessage::RegisterAgent(myself.clone()),
        );
        router.send_message(register_msg)?;

        Ok(AgentActor {
            agent_id: id,
            router: router.clone(),
            subscribed_topics: Vec::new(),
            state: AgentState::Ready,
            context,
        })
    }

    async fn handle(
        &self,
        myself: ActorRef<Self::Msg>,
        envelope: Self::Msg,
        state: &mut Self::State,
    ) -> Result<(), ActorProcessingErr> {
        let mut context = state.context.clone();
        context.timestamp = SystemTime::now();

        match envelope.payload {
            AgentMessage::Process(content) => {
                state.state = AgentState::Processing;

                let route_msg =
                    MessageEnvelope::new(context.clone(), RouterMessage::RouteMessage(content));
                state.router.send_message(route_msg)?;

                state.state = AgentState::Ready;
            }

            AgentMessage::UpdateState(new_state) => {
                state.state = new_state;
            }
        }


        Ok(())
    }
}
