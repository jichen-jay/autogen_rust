#[derive(Clone, Debug)]
pub struct TaskLedger {
    current_idx: usize,
    pub task_list: Vec<String>,
    pub solution_list: Vec<String>,
    pub parent_task: Option<String>,
    pub task_done: bool,
}

impl TaskLedger {
    pub fn new(task_list: Vec<String>, parent_task: Option<String>) -> Self {
        TaskLedger {
            current_idx: 0,
            task_list,
            solution_list: Vec::new(),
            parent_task,
            task_done: false,
        }
    }

    pub fn record_solution(&mut self, task_solution: String) -> bool {
        if self.current_idx > self.task_list.len() - 1 {
            return false;
        }
        self.solution_list.push(task_solution);
        self.current_idx += 1;
        if self.current_idx == self.task_list.len() - 1 {
            self.task_done = true;
        }
        true
    }

    pub fn current_task(&self) -> Option<String> {
        let idx = self.current_idx;
        match self.task_list.get(idx) {
            Some(ct) => Some(ct.to_string()),
            None => None,
        }
    }
}
