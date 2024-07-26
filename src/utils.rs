use regex::Regex;

pub fn parse_next_move_and_(
    input: &str,
    next_marker: Option<&str>,
) -> (bool, Option<String>, Vec<String>) {
    let json_regex = Regex::new(r"\{[^}]*\}").unwrap();
    let json_str = json_regex
        .captures(input)
        .and_then(|cap| cap.get(0))
        .map_or(String::new(), |m| m.as_str().to_string());

    let continue_or_terminate_regex =
        Regex::new(r#""continue_or_terminate":\s*"([^"]*)""#).unwrap();
    let continue_or_terminate = continue_or_terminate_regex
        .captures(&json_str)
        .and_then(|cap| cap.get(1))
        .map_or(String::new(), |m| m.as_str().to_string());

    let next_move = match next_marker {
        Some(marker) => {
            let next_marker_regex = Regex::new(&format!(r#""{}":\s*"([^"]*)""#, marker)).unwrap();
            Some(
                next_marker_regex
                    .captures(&json_str)
                    .and_then(|cap| cap.get(1))
                    .map_or(String::new(), |m| m.as_str().to_string()),
            )
        }
        None => None,
    };

    let key_points_array_regex = Regex::new(r#""key_points":\s*\[(.*?)\]"#).unwrap();

    let key_points_array = key_points_array_regex
        .captures(&json_str)
        .and_then(|cap| cap.get(1))
        .map_or(String::new(), |m| m.as_str().to_string());

    let key_points: Vec<String> = if !key_points_array.is_empty() {
        key_points_array
            .split(',')
            .map(|s| s.trim().trim_matches('"').to_string())
            .collect()
    } else {
        vec![]
    };

    (&continue_or_terminate == "TERMINATE", next_move, key_points)
}

pub fn parse_planning_sub_tasks(input: &str) -> (Vec<String>, String, String) {
    let sub_tasks_regex = Regex::new(r#""sub_tasks":\s*(\[[^\]]*\])"#).unwrap();
    let sub_tasks_str = sub_tasks_regex
        .captures(input)
        .and_then(|cap| cap.get(1))
        .map_or(String::new(), |m| m.as_str().to_string());

    let solution_found_regex = Regex::new(r#""solution_found":\s*"([^"]*)""#).unwrap();
    let solution_found = solution_found_regex
        .captures(input)
        .and_then(|cap| cap.get(1))
        .map_or(String::new(), |m| m.as_str().to_string());

    if sub_tasks_str.is_empty() {
        eprintln!("Failed to extract 'sub_tasks' from input.");
        return (vec![], input.to_string(), solution_found);
    }
    let task_summary_regex = Regex::new(r#""task_summary":\s*"([^"]*)""#).unwrap();
    let task_summary = task_summary_regex
        .captures(input)
        .and_then(|cap| cap.get(1))
        .map_or(String::new(), |m| m.as_str().to_string());

    let parsed_sub_tasks: Vec<String> = match serde_json::from_str(&sub_tasks_str) {
        Ok(val) => val,
        Err(_) => {
            eprintln!("Failed to parse extracted 'sub_tasks' as JSON.");
            return (vec![], task_summary, solution_found);
        }
    };

    (parsed_sub_tasks, task_summary, solution_found)
}
