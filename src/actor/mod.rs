pub mod agent;
pub mod router;

use crate::immutable_agent::{AgentId, Message, TopicId};
use ractor::ActorRef;

// #[derive(Serialize, Deserialize, Debug, Clone)]
// pub struct Message {
//     pub content: Content,
//     pub name: Option<String>,
//     pub role: Role,
// }

//RouterMessage is a super structure of Msg, it wraps agent's payload
//it also represent other conditions that agent is in
//agent needs to register itself to Router, to subscribe to topics, 
//what else do you see is needed for an LLM agent
#[derive(Debug)]
pub enum RouterMessage {
    RegisterAgent(AgentId, ActorRef<AgentMessage>),
    RouteMessage(AgentId, Message),
    AgentSubscribe(AgentId, TopicId),
}

//this is a simplified way of wrapping Message, which is the raw input and output of LLM agent
//you can propose better structure to wrap and encapsulate other needed data as well
//need to distribute tasks between AgentMessage and RouterMessage, suggest better ways to distribute features
#[derive(Debug)]
pub struct AgentMessage(Message);

//I think agent just needs to know from whom to receive Msg, after processing
//it sends Msg to Router, Router decides what to do next with Msg
pub struct AgentActor {
    pub agent_id: AgentId,
    pub router: ActorRef<RouterMessage>,
    pub subscribed_topics: Vec<TopicId>,

}

//Router receives Msg from agents, it distributes to agents according to their subscriptions
//
pub struct RouterActor {
    pub agents: std::collections::HashMap<AgentId, ActorRef<AgentMessage>>,
    pub subscriptions: std::collections::HashMap<AgentId, ActorRef<AgentMessage>>,
    pub receiver: AgentId,
    pub sender: AgentId,

}
