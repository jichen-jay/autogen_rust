use crate::agent_runtime::{
    agent::{AgentActor, AgentState},
    ActorContext, AgentId, RouterCommand, SpawnAgentResponse, TopicId,
};
use crate::immutable_agent::{LlmAgent, Message};
use crate::FormatterWrapper;
use ractor::{Actor, ActorCell, ActorProcessingErr, ActorRef};
use serde_json::Value;
use std::collections::HashMap;
use std::result::Result as StdResult;
use thiserror::Error;

#[derive(Debug, Default, Clone, PartialEq)]
pub enum RouterStatus {
    Ready,
    #[default]
    Off,
}

#[derive(Debug, Error, Clone)]
pub enum RouterError {
    #[error("Router is not in ready state")]
    NotReady,

    #[error("Agent {0} not found")]
    AgentNotFound(AgentId),

    #[error("Agent {0} build failed")]
    AgentBuildFailed(AgentId),

    #[error("Failed to spawn agent: {0}")]
    SpawnFailed(String),

    #[error("Topic {0} not found")]
    TopicNotFound(TopicId),

    #[error("Invalid state: {0}")]
    InvalidState(String),
    // #[error("Agent actor failure: {0}")]
    // ActorFailure(#[from] ActorProcessingErr),
}

#[derive(Clone)]
pub struct RouterState {
    agents: HashMap<AgentId, ActorRef<RouterCommand>>,
    topic_subscriptions: HashMap<TopicId, Vec<AgentId>>,
    agent_subscriptions: HashMap<AgentId, Vec<TopicId>>,
    agent_states: HashMap<AgentId, AgentState>,
    state: RouterStatus,
    router: Option<ActorRef<RouterCommand>>,
}

impl Default for RouterState {
    fn default() -> Self {
        Self {
            agents: HashMap::new(),
            topic_subscriptions: HashMap::new(),
            agent_subscriptions: HashMap::new(),
            agent_states: HashMap::new(),
            state: RouterStatus::default(),
            router: None,
        }
    }
}

impl RouterState {
    pub fn set_router(&mut self, router: ActorRef<RouterCommand>) {
        self.router = Some(router);
    }

    pub fn is_ready(&self) -> bool {
        self.state == RouterStatus::Ready
    }

    fn ensure_ready(&self) -> StdResult<(), RouterError> {
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
    ) -> StdResult<(), RouterError> {
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
    ) -> StdResult<(), RouterError> {
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

    pub fn get_agent_topics(&self, agent_id: &AgentId) -> StdResult<Vec<TopicId>, RouterError> {
        self.ensure_ready()?;
        self.agent_subscriptions
            .get(agent_id)
            .cloned()
            .ok_or(RouterError::AgentNotFound(*agent_id))
    }

    pub fn get_topic_subscribers(&self, topic: &TopicId) -> StdResult<Vec<AgentId>, RouterError> {
        self.ensure_ready()?;
        self.topic_subscriptions
            .get(topic)
            .cloned()
            .ok_or(RouterError::TopicNotFound(topic.clone()))
    }

    async fn spawn_agent_w_actor(
        &mut self,
        system_prompt: &str,
        user_prompt_formatter: Option<FormatterWrapper>,
        topic: TopicId,
        tools_map_meta: Option<Value>,
        description: String,
    ) -> StdResult<AgentId, RouterError> {
        self.ensure_ready()?;

        let new_agent_id = AgentId::new_v4();
        match LlmAgent::build(
            system_prompt.to_string(),
            user_prompt_formatter,
            None,
            tools_map_meta,
            description,
        ) {
            Ok(llm_agent) => {
                let agent_actor = AgentActor::new(
                    new_agent_id,
                    self.router
                        .as_ref()
                        .ok_or(RouterError::InvalidState("Router reference missing".into()))?
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
                            .ok_or(RouterError::InvalidState("Router reference missing".into()))?
                            .clone(),
                        llm_agent,
                    ),
                    self.router
                        .as_ref()
                        .ok_or(RouterError::InvalidState("Router reference missing".into()))?
                        .clone()
                        .into(),
                )
                .await
                .map_err(|e| RouterError::SpawnFailed(e.to_string()))?;

                self.agents.insert(new_agent_id, agent_ref);
                self.agent_states
                    .insert(new_agent_id, AgentState::new(new_agent_id));
                self.agent_subscriptions.insert(new_agent_id, Vec::new());

                self.subscribe_agent(new_agent_id, topic)?;

                Ok(new_agent_id)
            }
            Err(e) => Err(RouterError::AgentBuildFailed(new_agent_id)),
        }
    }

    fn shutdown_agent(&mut self, agent_id: AgentId) -> StdResult<(), RouterError> {
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
    ) -> StdResult<(), RouterError> {
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
    type State = RouterState; // Use RouterState as the state type
    type Arguments = ();

    async fn pre_start(
        &self,
        myself: ActorRef<Self::Msg>,
        args: Self::Arguments,
    ) -> StdResult<Self::State, ActorProcessingErr> {
        // Initialize state here
        Ok(RouterState {
            agents: HashMap::new(),
            topic_subscriptions: HashMap::new(),
            agent_subscriptions: HashMap::new(),
            agent_states: HashMap::new(),
            state: RouterStatus::Off,
            router: Some(myself), // Store the actor's own reference
        })
    }
    async fn handle(
        &self,
        myself: ActorRef<Self::Msg>,
        msg: Self::Msg,
        state: &mut Self::State, // Use state parameter for mutations
    ) -> StdResult<(), ActorProcessingErr> {
        match msg {
            RouterCommand::SpawnAgent {
                system_prompt,
                user_prompt_formatter,
                topic,
                tools_map_meta,
                description,
                reply_to,
            } => match state
                .spawn_agent_w_actor(
                    &system_prompt,
                    user_prompt_formatter,
                    topic.clone(),
                    tools_map_meta,
                    description,
                )
                .await
            {
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
                state.state = RouterStatus::Off;
            }
            RouterCommand::Ready => {
                state.state = RouterStatus::Ready;
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
