use crate::actor::RouterMessage;
use crate::llama_structs::*;
use crate::llm_utils::*;
use crate::task_ledger::*;
use crate::utils::*;
// use crate::webscraper_hook::{get_webpage_text, search_with_bing};
use crate::actor::{AgentActor, AgentId, TopicId};
use crate::actor::{AgentMessage, MessageEnvelope};
use crate::{IS_TERMINATION_SYSTEM_PROMPT, ITERATE_NEXT_STEP_TEMPLATE, TOGETHER_CONFIG};
use anyhow::Result;
use async_openai::types::Role;
use log::debug;
use ractor::ActorRef;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::HashMap;
use std::error::Error;
use std::sync::{Arc, RwLock};
use tokio::io::{stdin, AsyncBufReadExt, BufReader};
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
            role,
        }
    }
}

#[derive(Clone)]
pub struct LlmAgent {
    pub system_prompt: String,
    pub llm_config: Option<Value>,
    pub tools_map_meta: String,
    pub description: String,
}

impl LlmAgent {
    pub fn new(system_prompt: String, llm_config: Option<Value>) -> Self {
        Self {
            system_prompt,
            llm_config,
            tools_map_meta: String::new(),
            description: String::new(),
        }
    }

    pub async fn default_method(&self, input: &str) -> anyhow::Result<String> {
        debug!("default_method: received input: {:?}", input);

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

        debug!("default_method: generated output: {:?}", output);

        match &output.content {
            Content::Text(_out) => Ok(_out.to_string()),
            _ => unreachable!(),
        }
    }
}

pub async fn get_user_feedback() -> Result<String> {
    // print!("User input: ");

    let mut input = String::new();

    let mut reader = BufReader::new(stdin());

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
