use crate::llama_structs::*;
use crate::llm_utils::*;
use crate::task_ledger::*;
use crate::utils::*;
// use crate::webscraper_hook::{get_webpage_text, search_with_bing};
use crate::{IS_TERMINATION_SYSTEM_PROMPT, ITERATE_NEXT_STEP_TEMPLATE, TOGETHER_CONFIG};
use anyhow::Result;
use async_openai::types::Role;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::HashMap;
use std::error::Error;
use std::sync::{Arc, RwLock};
use tokio::io::{self, AsyncBufReadExt};
use tokio::time::{timeout, Duration};
use uuid::Uuid;

const NEXT_STEP_PLANNING: &'static str = r#"
<|im_start|>system You are a helpful AI assistant. Your task is to decompose complex tasks into clear, manageable sub-tasks and provide high-level guidance.

Define Objective:
Clearly state the main goal.
Break Down Tasks:
Divide the main goal into logical, high-level sub-tasks without delving into excessive detail.
Summarize Findings:

DO NOT further break down the sub-tasks.

Think aloud and write down your thoughts in the following template:
{
 "top_of_mind_reply": "how I would instinctively answer this question",
 "task_needs_break_down_or_not": "is the task difficult, I need multiple steps to solve? if not, no need to create sub_tasks, just repeat the task requirement in task_summary section",
 "sub_tasks": [
        "sub_task one",
        "...",
       "sub_task N”
    ],
”task_summary”: "summary”
”solution_found”: "the solution you may have found in single shot, and its content”
}
"#;

const NEXT_STEP_BY_TOOLCALL: &'static str = r#"
<|im_start|>system You are a function-calling AI model. You are provided with function signatures within <tools></tools> XML tags. You will call one function and ONLY ONE to assist with the user query. Do not make assumptions about what values to plug into functions.

<tools>
1. **use_intrinsic_knowledge**: 
Description: Solves tasks using capabilities and knowledge obtained at training time.
Special Note: You can handle many fuzzy tasks this way because you have great writing skills, you may provide a common sense solution despite you might not know the exact details. 
Example Call:
<tool_call>
{"arguments": {"task": "tell a joke"}, 
"name": "use_intrinsic_knowledge"}
</tool_call>

2. **search_with_bing**: 
Description: Conducts an internet search using Bing search engine and returns relevant results based on the query provided by the user.
Special Note 1: This function helps narrow down potential sources of information before extracting specific content.
Special Note 2: Using this as an initial step can make subsequent tasks more targeted by providing exact links that can then be scraped using get_webpage_text.
Example Call:
<tool_call>
{"arguments": {"query": "latest AI research trends"}, 
"name": "search_with_bing"}
</tool_call>

3. **code_with_python**: 
Description: Generates clean, executable Python code for various tasks based on user input.
Special Note: When task requires precise mathematical operations; processing, analyzing and creating complex data types, where AI models can not efficiently represent and manipulate in natural language terms, this is the way out.
Example Call:
<tool_call>
{"arguments": {"key_points": "Create a Python script that reads a CSV file and plots a graph"}, 
"name": "code_with_python"}
</tool_call>
</tools>

Function Definitions

use_intrinsic_knowledge
Description: Solves tasks using built-in capabilities obtained at training time.
Parameters: "task" The task you receive (type:string)
Required Parameters: ["task"]

search_with_bing
Description: Conducts an internet search using Bing search engine and returns relevant results based on the query provided by the user.
Parameters: "query" The search query to be executed on Bing (type:string)
Required Parameters: ["query"]

code_with_python
Description: Generates clean executable Python code for various tasks based on key points describing what needs to be solved with code.
Parameters: "key_points" Key points describing what kind of problem needs to be solved with Python code (type:string)
Required Parameters: ["key_points"]

Remember that you are a dispatcher; you DO NOT work on tasks yourself, especially when you see specific coding suggestions, don't write any code, just dispatch.

For each function call, return a JSON object with function name and arguments within <tool_call></tool_call> XML tags as follows:

<tool_call>  
{"arguments": <args-dict>,   
"name": "<function_name>"}
</tool_call>
"#;

const ITERATE_NEXT_STEP: &'static str = r#"
<|im_start|>system You are a task solving expert. You follow steps to solve complex problems. For much of the time, you're working iteratively on the sub_tasks, you are given the result from a previous step, you execute on the instruction you receive for your current step.
"#;

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct Message {
    pub content: Content,
    pub name: Option<String>,
    pub role: Role,
}

impl Default for Message {
    fn default() -> Self {
        Message {
            content: Content::Text("placeholder".to_string()),
            name: None,
            role: Role::User,
        }
    }
}

impl Message {
    pub fn new(content: Content, name: Option<String>, role: Role) -> Self {
        Message {
            content,
            name,
            role, // Set default role to Assistant if None is provided
        }
    }
}

pub type AgentId = Uuid;
pub type Subscription = HashMap<AgentId, Vec<TopicId>>;
pub type TopicId = String;

pub struct LlmAgent {
    pub agent_id: AgentId,
    pub system_prompt: String,
    pub llm_config: Option<Value>,
    pub tools_map_meta: String,
    pub description: String,
    subscriptions: Arc<RwLock<Subscription>>,
}

impl LlmAgent {
    pub fn simple(system_prompt: &str) -> Self {
        LlmAgent {
            agent_id: Uuid::new_v4(),
            system_prompt: system_prompt.to_string(),
            llm_config: None,
            tools_map_meta: String::from(""),
            description: String::from(""),
            subscriptions: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    pub fn add_subscription(&self, topic_id: TopicId) -> Result<(), Box<dyn Error>> {
        let mut subs = self.subscriptions.write().unwrap();
        subs.entry(self.agent_id)
            .or_insert_with(Vec::new)
            .push(topic_id);
        Ok(())
    }

    pub fn get_subscriptions(&self) -> Option<Vec<TopicId>> {
        let subs = self.subscriptions.read().unwrap();
        subs.get(&self.agent_id).cloned()
    }

    pub async fn next_step_by_toolcall(
        &self,
        carry_over: Option<String>,
        input: &str,
    ) -> anyhow::Result<String> {
        let max_token = 1000u16;
        let output: LlamaResponseMessage =
            chat_inner_async_wrapper(&TOGETHER_CONFIG, NEXT_STEP_BY_TOOLCALL, input, max_token)
                .await
                .expect("Failed to generate reply");
        match &output.content {
            Content::Text(unexpected_result) => {
                return Ok(format!(
                    "attempt to run tool_call failed, returning text result: {} ",
                    unexpected_result
                ));
            }
            Content::ToolCall(call) => {
                let args = call.clone().arguments.unwrap_or_default();
                let res = match call.name.as_str() {
                    "use_intrinsic_knowledge" => match args.get("task") {
                        Some(t) => {
                            println!("entered intrinsic arm: {:?}", t.clone());
                            //why program doesn't continue to execute the next async func?
                            match self.iterate_next_step(carry_over, t).await {
                                Ok(res) => {
                                    println!("intrinsic result: {:?}", res.clone());
                                    res
                                }

                                Err(_) => String::from("failed in use_intrinsic_knowledge"),
                            }
                        }
                        None => String::from("failed in use_intrinsic_knowledge"),
                    },

                    _ => {
                        panic!();
                    }
                };
                Ok(res)
            }
        }
    }
    pub async fn iterate_next_step(
        &self,
        carry_over: Option<String>,
        input: &str,
    ) -> anyhow::Result<String> {
        let max_token = 1000u16;

        let formatter = ITERATE_NEXT_STEP_TEMPLATE.lock().unwrap();
        let user_prompt = match &carry_over {
            Some(c) => formatter(&[&c, input]),
            None => input.to_string(),
        };

        let system_prompt = match &carry_over {
            Some(_) => ITERATE_NEXT_STEP,
            None => "<|im_start|>system You are a task solving expert.",
        };
        let output: LlamaResponseMessage =
            chat_inner_async_wrapper(&TOGETHER_CONFIG, ITERATE_NEXT_STEP, &user_prompt, max_token)
                .await
                .expect("Failed to generate reply");
        match &output.content {
            Content::Text(res) => Ok(res.to_string()),
            Content::ToolCall(call) => Err(anyhow::Error::msg("entered tool_call arm incorrectly")),
        }
    }

    pub async fn planning(&self, input: &str) -> (TaskLedger, Option<String>) {
        let max_token = 500u16;
        let output: LlamaResponseMessage =
            chat_inner_async_wrapper(&TOGETHER_CONFIG, NEXT_STEP_PLANNING, input, max_token)
                .await
                .expect("Failed to generate reply");

        match &output.content {
            Content::Text(_out) => {
                let (task_list, task_summary, solution_found) = parse_planning_sub_tasks(_out);
                println!(
                    "sub_tasks: {:?}\n task_summary: {:?}",
                    task_list.clone(),
                    task_summary.clone()
                );

                (
                    TaskLedger::new(task_list, Some(task_summary)),
                    Some(solution_found),
                )
            }
            _ => unreachable!(),
        }
    }

    pub async fn simple_reply(&self, input: &str) -> anyhow::Result<bool> {
        let user_prompt = format!("Here is the task for you: {:?}", input);

        let max_token = 1000u16;
        let output: LlamaResponseMessage = chat_inner_async_wrapper(
            &TOGETHER_CONFIG,
            &self.system_prompt,
            &user_prompt,
            max_token,
        )
        .await
        .expect("Failed to generate reply");

        match &output.content {
            Content::Text(_out) => {
                let (terminate_or_not, key_points) =
                    self._is_termination(&_out, &user_prompt).await;

                println!(
                    "terminate?: {:?}, points: {:?}\n",
                    terminate_or_not.clone(),
                    key_points.clone()
                );
                if terminate_or_not {
                    let _ = get_user_feedback().await;
                }
                return Ok(terminate_or_not);
            }
            _ => unreachable!(),
        }
    }

    pub async fn _is_termination(
        &self,
        current_text_result: &str,
        instruction: &str,
    ) -> (bool, String) {
        let user_prompt = format!(
            "Given the task: {:?}, examine current result: {}, please decide whether the task is done or not",
            instruction,
            current_text_result
        );

        println!("{:?}", user_prompt.clone());

        let raw_reply = chat_inner_async_wrapper(
            &TOGETHER_CONFIG,
            &IS_TERMINATION_SYSTEM_PROMPT,
            &user_prompt,
            300,
        )
        .await
        .expect("llm generation failure");

        println!(
            "_is_termination raw_reply: {:?}",
            raw_reply.content_to_string()
        );

        let (terminate_or_not, _, key_points) =
            parse_next_move_and_(&raw_reply.content_to_string(), None);

        (terminate_or_not, key_points.join(","))
    }
}

pub async fn get_user_feedback() -> Result<String> {
    // print!("User input: ");

    let mut input = String::new();
    let mut reader = io::BufReader::new(io::stdin());

    match timeout(Duration::from_secs(10), async {
        reader
            .read_line(&mut input)
            .await
            .expect("Failed to read line");
        input
    })
    .await
    {
        Ok(mut input) => {
            if let Some('\n') = input.clone().chars().next_back() {
                input.pop();
            }
            if let Some('\r') = input.chars().next_back() {
                input.pop();
            }

            if input == "stop" {
                std::process::exit(0);
            }
            if input == "back" {
                return Err(anyhow::Error::msg("back to main"));
            }
            Ok(input)
        }
        Err(e) => {
            eprintln!("{}", e);
            std::process::exit(0);
        }
    }
}
