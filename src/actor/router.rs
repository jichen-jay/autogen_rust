use crate::actor::{
    agent::{AgentActor, AgentState},
    ActorContext, AgentId, RouterCommand, SpawnAgentResponse, TopicId,
};
use crate::immutable_agent::{LlmAgent, Message};
use ractor::{Actor, ActorCell, ActorProcessingErr, ActorRef};
use std::collections::HashMap;

#[derive(Debug, Default, Clone, PartialEq)]
pub enum RouterState {
    Ready,
    #[default]
    Off,
}

#[derive(Debug)]
pub enum RouterError {
    NotReady,
    AgentNotFound(AgentId),
    SpawnFailed(String),
    TopicNotFound(TopicId),
    InvalidState(String),
}

impl From<RouterError> for ActorProcessingErr {
    fn from(error: RouterError) -> Self {
        Box::new(std::io::Error::new(
            std::io::ErrorKind::Other,
            format!("{:?}", error),
        ))
    }
}

#[derive(Clone)]
pub struct RouterStateData {
    agents: HashMap<AgentId, ActorRef<RouterCommand>>,
    topic_subscriptions: HashMap<TopicId, Vec<AgentId>>,
    agent_subscriptions: HashMap<AgentId, Vec<TopicId>>,
    agent_states: HashMap<AgentId, AgentState>,
    state: RouterState,
    router: Option<ActorRef<RouterCommand>>, // Make this Optional
}

impl Default for RouterStateData {
    fn default() -> Self {
        Self {
            agents: HashMap::new(),
            topic_subscriptions: HashMap::new(),
            agent_subscriptions: HashMap::new(),
            agent_states: HashMap::new(),
            state: RouterState::default(),
            router: None, // Start with None
        }
    }
}

impl RouterStateData {
    pub fn set_router(&mut self, router: ActorRef<RouterCommand>) {
        self.router = Some(router);
    }

    pub fn new() -> Self {
        Self {
            agents: HashMap::new(),
            topic_subscriptions: HashMap::new(),
            agent_subscriptions: HashMap::new(),
            agent_states: HashMap::new(),
            state: RouterState::Off,
            router: None,
        }
    }

    pub fn is_ready(&self) -> bool {
        self.state == RouterState::Ready
    }

    fn ensure_ready(&self) -> Result<(), RouterError> {
        if !self.is_ready() {
            Err(RouterError::NotReady)
        } else {
            Ok(())
        }
    }

    pub fn subscribe_agent(
        &mut self,
        agent_id: AgentId,
        topic: TopicId,
    ) -> Result<(), RouterError> {
        self.ensure_ready()?;

        if !self.agents.contains_key(&agent_id) {
            return Err(RouterError::AgentNotFound(agent_id));
        }

        self.topic_subscriptions
            .entry(topic.clone())
            .or_insert_with(Vec::new)
            .push(agent_id);

        self.agent_subscriptions
            .entry(agent_id)
            .or_insert_with(Vec::new)
            .push(topic);

        Ok(())
    }

    pub fn unsubscribe_agent(
        &mut self,
        agent_id: AgentId,
        topic: &TopicId,
    ) -> Result<(), RouterError> {
        self.ensure_ready()?;

        if !self.agents.contains_key(&agent_id) {
            return Err(RouterError::AgentNotFound(agent_id));
        }

        if let Some(subscribers) = self.topic_subscriptions.get_mut(topic) {
            subscribers.retain(|id| *id != agent_id);
        }

        if let Some(topics) = self.agent_subscriptions.get_mut(&agent_id) {
            topics.retain(|t| t != topic);
        }

        Ok(())
    }

    pub fn get_agent_topics(&self, agent_id: &AgentId) -> Result<Vec<TopicId>, RouterError> {
        self.ensure_ready()?;
        self.agent_subscriptions
            .get(agent_id)
            .cloned()
            .ok_or(RouterError::AgentNotFound(*agent_id))
    }

    pub fn get_topic_subscribers(&self, topic: &TopicId) -> Result<Vec<AgentId>, RouterError> {
        self.ensure_ready()?;
        self.topic_subscriptions
            .get(topic)
            .cloned()
            .ok_or(RouterError::TopicNotFound(topic.clone()))
    }

    async fn spawn_agent(
        &mut self,
        system_prompt: &str,
        topic: TopicId,
    ) -> Result<AgentId, RouterError> {
        self.ensure_ready()?;

        let new_agent_id = AgentId::new_v4();
        let llm_agent = LlmAgent::new(system_prompt.to_string(), None);

        let agent_actor = AgentActor::new(
            new_agent_id,
            self.router
                .as_ref()
                .expect("Router not initialized")
                .clone(),
            llm_agent.clone(),
        );

        let (agent_ref, _) = Actor::spawn_linked(
            None,
            agent_actor,
            (
                new_agent_id,
                self.router
                    .as_ref()
                    .expect("Router not initialized")
                    .clone(),
                llm_agent,
            ),
            self.router
                .as_ref()
                .expect("Router not initialized")
                .clone()
                .into(),
        )
        .await
        .map_err(|e| RouterError::SpawnFailed(e.to_string()))?;

        self.agents.insert(new_agent_id, agent_ref);
        self.agent_states.insert(new_agent_id, AgentState::Ready);
        self.agent_subscriptions.insert(new_agent_id, Vec::new());
        self.subscribe_agent(new_agent_id, topic)?;

        Ok(new_agent_id)
    }

    fn shutdown_agent(&mut self, agent_id: AgentId) -> Result<(), RouterError> {
        let agent_ref = self
            .agents
            .remove(&agent_id)
            .ok_or(RouterError::AgentNotFound(agent_id))?;

        if let Some(topics) = self.agent_subscriptions.remove(&agent_id) {
            for topic in topics {
                if let Some(subscribers) = self.topic_subscriptions.get_mut(&topic) {
                    subscribers.retain(|id| *id != agent_id);
                }
            }
        }

        agent_ref.stop(None);
        self.agent_states.remove(&agent_id);

        Ok(())
    }

    fn route_message(
        &mut self,
        topic: TopicId,
        message: Message,
        context: ActorContext,
    ) -> Result<(), RouterError> {
        self.ensure_ready()?;

        let agent_ids = self
            .topic_subscriptions
            .get(&topic)
            .ok_or(RouterError::TopicNotFound(topic.clone()))?;

        if agent_ids.is_empty() {
            return Err(RouterError::TopicNotFound(topic));
        }

        for agent_id in agent_ids {
            // Don't route message back to sender using context
            if context.sender == Some(*agent_id) {
                continue;
            }

            if let Some(agent_ref) = self.agents.get(agent_id) {
                if let Err(e) = agent_ref.cast(RouterCommand::RouteMessage {
                    topic: topic.clone(),
                    message: message.clone(),
                    context: context.clone(),
                }) {
                    log::warn!("Failed to route message to agent {}: {:?}", agent_id, e);
                }
            }
        }

        Ok(())
    }
}

pub struct RouterActor;

impl Default for RouterActor {
    fn default() -> Self {
        RouterActor // Simple initialization
    }
}

// #[ractor::async_trait]
impl Actor for RouterActor {
    type Msg = RouterCommand;
    type State = RouterStateData; // Use RouterStateData as the state type
    type Arguments = ();

    async fn pre_start(
        &self,
        myself: ActorRef<Self::Msg>,
        args: Self::Arguments,
    ) -> Result<Self::State, ActorProcessingErr> {
        // Initialize state here
        Ok(RouterStateData {
            agents: HashMap::new(),
            topic_subscriptions: HashMap::new(),
            agent_subscriptions: HashMap::new(),
            agent_states: HashMap::new(),
            state: RouterState::Off,
            router: Some(myself), // Store the actor's own reference
        })
    }
    async fn handle(
        &self,
        myself: ActorRef<Self::Msg>,
        msg: Self::Msg,
        state: &mut Self::State, // Use state parameter for mutations
    ) -> Result<(), ActorProcessingErr> {
        match msg {
            RouterCommand::SpawnAgent {
                system_prompt,
                topic,
                reply_to,
            } => match state.spawn_agent(&system_prompt, topic.clone()).await {
                Ok(agent_id) => {
                    let response = SpawnAgentResponse::Ok(agent_id);
                    if !reply_to.is_closed() {
                        let _ = reply_to.send(response);
                    }
                }
                Err(e) => {
                    let response =
                        SpawnAgentResponse::Err(format!("spawn agent on topic: {} failed", topic));
                    if !reply_to.is_closed() {
                        let _ = reply_to.send(response);
                    }
                }
            },

            RouterCommand::RouteMessage {
                topic,
                message,
                context,
            } => {
                state.route_message(topic, message, context)?;
            }

            RouterCommand::ShutdownAgent { agent_id } => {
                state.shutdown_agent(agent_id)?;
            }
            RouterCommand::Off => {
                state.state = RouterState::Off;
            }
            RouterCommand::Ready => {
                state.state = RouterState::Ready;
            }
            RouterCommand::SubscribeAgent { agent_id, topic } => {
                state.subscribe_agent(agent_id, topic)?;
            }
            RouterCommand::UnsubscribeAgent { agent_id, topic } => {
                state.unsubscribe_agent(agent_id, &topic)?;
            }
        }
        Ok(())
    }
}
