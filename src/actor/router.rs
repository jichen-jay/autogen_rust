use crate::actor::{AgentMessage, RouterActor, RouterMessage};
use ractor::{Actor, ActorProcessingErr, ActorRef};
use std::collections::HashMap;
use uuid::Uuid;

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
            receiver: Uuid::new_v4(),
            sender: Uuid::new_v4(),
        })
    }
    //need to find a way to distrubute data between Msg and State
    //Msg is supposed to be the payload of agent's work
    // State is supposed to be meta data of the agent's work
    // my challenge is that I don't want to make agent struct mutable,
    //so I need to distrute some changeable data to State
    //for LLM agent which I'm dealing with, Msg is the data that agent ingest and emits
    //LLM agent changes its comms state by subscribing to topics, unsubscribe to topics
    //agent also has state of ready, pending-shutdown,
    async fn handle(
        &self,
        _myself: ActorRef<Self::Msg>,
        message: Self::Msg,
        state: &mut Self::State,
    ) -> Result<(), ActorProcessingErr> {
        match message {
            RouterMessage::AgentSubscribe(agent_id, topic_id) => {
                state.agents.insert(id, actor_ref);
            }
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
