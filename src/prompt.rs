use lazy_static::lazy_static;
use std::sync::{Arc, Mutex};

type FormatterFn = Box<dyn (Fn(&[&str]) -> String) + Send + Sync>;

lazy_static! {
    pub static ref CHECK_TASK_SYSTEM_TEMPLATE: String =
        r#"
<|im_start|>system You are a function calling AI model. You are provided with function signatures within <tools></tools> XML tags. You may call one or more functions to assist with the user query. Don't make assumptions about what values to plug into functions. Here are the available tools: <tools>
    
    The function "code_with_python" generates clean, executable Python code for various tasks based on the user input. For example, calling "code_with_python("key_points": "Create a Python script that reads a CSV file and plots a graph")" will generate Python code that performs this task.
    
    The function "get_webpage_text" retrieves all text content from a given URL, which can be useful for extracting information from web pages or articles. For example, calling "get_webpage_text("https://example.com")" will fetch the text from Example.com.
    
    The function "search_with_bing" performs an internet search using Bing and returns relevant results based on the query provided by the user. This can be useful for finding up-to-date information on various topics. For example, "search_with_bing("latest AI research trends")" will return search results related to the latest trends in AI research.
    
    {
        "name": "get_webpage_text",
        "description": "Retrieves all text content from a specified website URL.",
        "parameters": {
            "url": {
                "type": "string",
                "description": "The URL of the website from which to fetch the text content"
            }
        },
        "required": ["url"],
        "type": "object"
    }
    
    {
        "name": "code_with_python",
            "description": "Generates clean, executable Python code for various tasks",
            "parameters": {
                "key_points": {
                    "type": "string",
                    "description": "Key points from input that describes what kind of problem needs to be solved with Python code."
                }
            },
            "required": ["key_points"],
            "type": "object"
 }
 
 {
    "name": "search_with_bing",
    "description": "Conducts an internet search using Bing search engine and returns relevant results.",
    "parameters": { 
         "query": { 
             "type": "string", 
             "description": "The search query to be executed on Bing" 
          } 
     }, 
     "required": ["query"], 
     "type": "object"
 }

Examples of toolcalls for different scenarios and tools:
1. To retrieve webpage text:
<tool_call>
{"arguments":{"url":"https://example.com"}, 
"name":"get_webpage_text"}
</tool_call>

2. To generate Python code:
<tool_call>
{"arguments":{"key_points":"Create a Python script that reads data from an API and stores it in a database"}, 
"name":"code_with_python"}
</tool_call>

3. To perform an internet search:
<tool_call>
{"arguments":{"query":"best practices in software development"}, 
"name":"search_with_bing"}
</tool_call>

For each function call return a json object with function name and arguments within <tool_call></tool_call> XML tags as follows:
<tool_call>
{"arguments": <args-dict>, 
"name":"<function-name>"}
</tool_call>
 "#.to_string()    ;

    pub static ref CHECK_TASK_USER_TEMPLATE_MULTI: Arc<Mutex<FormatterFn>> = Arc::new(Mutex::new(
        Box::new(|args: &[&str]| {
            format!(
                r#"
Here is the sub_task "Research Nvidia's stock price performance over the past month" for you to work out, it's from the list of sub_tasks: [
\"Research Nvidia's stock price performance over the past month\",\n\"Collect data on Nvidia's stock price\",\n\"Analyze the data to identify trends and patterns\",\n\"Write a blog post summarizing the findings\",\n\"Edit and proofread the blog post\"\n
]
- Use your tools to complete this sub_task. In some cases, you may need to break it down further into smaller steps.
here is the actionable steps created by your peer: {}
- Review these execution steps for any redundancy or overlap with other subtasks at both the parent task level (tier 1) and within the actionable steps (tier 2).

2. **Consolidation and Update:**
- Consolidate any tier 2 steps that perform similar functions as tier 1 tasks.
- Create an updated version of the tier 1 task breakdown that ensures atomic segmentation without redundancy.
- If you find any redundant or overlapping steps, remove them to ensure efficiency without compromising completeness.

3. **Special Notes:**
- Task planning aims to break down actions into executable steps that are neither too trivial nor too large for the agent to complete in one go.
- Tier 1 tasks may be too general; use more specific details from tier 2 if they make the tasks more executable.
- Respect the tier 1 breakdown generally, but replace with more actionable and grounded items from tier 2 when necessary.

4. **Use Project Objective as Guidance:**

Ultimately, but completing all the sub_tasks, you are to deliver this result: "Write a blog post about Nvidia's stock price performance over the past month, including an analysis of the data and any notable trends or patterns", Use this objective to guide your understanding of the overall picture.

Think alound and put your thoughts down in the following template: 
{{
\"my_thought_process\": [
    \"project_objective: project object I received\",
    \"tier 1 steps: tier 1 steps I'm given\",
    \"tier 2 actionable_items: items\","compare_and_replace: Compare tier 1 steps with tier 2 items and replace if tier 2 items are more actionable"
],
"updated_sub_task_list": [
"sub_task one",
"...",
   "sub_task N”
]
            }}
    "#,
                args[0]
            )
        })
    ));
}
