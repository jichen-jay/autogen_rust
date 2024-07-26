use anyhow::Result;
use autogen_rust::exec_python::run_python_func_react;
use autogen_rust::webscraper_hook::*;
use autogen_rust::{immutable_agent::*, task_ledger};
use tokio;

#[tokio::main]
async fn main() -> Result<()> {
    dotenv::dotenv().ok();

    let user_proxy = ImmutableAgent::simple("user_proxy", "");

    // match run_python_func_react("/Users/jichen/Projects/autogen_rust/src/test.py").await {
    //     Ok(res) => println!("solution: {:?}\n\n ", res),
    //     Err(res) => println!("code run error: {:?}\n\n ", res),
    // };
    //
    // return Ok(());

    // let guide =
    //     "today is 2024-06-18, find the stock price info of Nvidia in the past month".to_string();
    //
    // let ve = search_with_bing(
    //     "today is 2024-06-18, go get data: https://ca.finance.yahoo.com/quote/NVDA/history",
    // )
    // .await?;
    //
    // for (url, _) in ve {
    //     let res = get_webpage_guided(
    //         "https://ca.finance.yahoo.com/quote/NVDA/history".to_string(),
    //         &guide,
    //     )
    //     .await;
    //
    // }

    // let code = user_proxy
    //     .code_with_python("create a 5x5 tick tac toe game in python")
    //     .await?;

    let (mut task_ledger, solution) = user_proxy
        // .planning("tell me a joke")
        .planning("Today is 2024-03-18. Write code to find the stock price performance of Nvidia in the past month")
        .await;

    if task_ledger.task_list.is_empty() && solution.is_some() {
        println!("solution: {:?} ", solution);
        std::process::exit(0);
    }

    loop {
        let task_summary = task_ledger
            .clone()
            .parent_task
            .unwrap_or("no task".to_string());
        let task = task_ledger.current_task().unwrap_or(task_summary);

        // let _ = user_proxy
        //     .code_with_python("Write python code to show the stock price performance of Nvidia in the past month")
        //     .await;
        let carry_over = match task_ledger.solution_list.last() {
            Some(c) => Some(c.clone()),
            None => None,
        };
        let res_alt = user_proxy
            .next_step_by_toolcall(carry_over, &task)
            .await
            .unwrap_or("no result generated".to_string());
        println!("{:?}", res_alt.clone());

        tokio::time::sleep(std::time::Duration::from_secs(60)).await;

        if !task_ledger.record_solution(res_alt) {
            break;
        }
    }

    println!(
        "{:?}",
        &task_ledger
            .solution_list
            .last()
            .unwrap_or(&"no final result".to_string())
    );

    Ok(())
}
