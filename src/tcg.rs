use std::collections::{hash_map::Entry, HashMap};

use serde::Serialize;

#[derive(Clone, Debug, Default, Serialize)]
pub struct Node {
    pub children: HashMap<String, Node>,
    pub sandbox_id: Option<String>,
    pub value: Option<String>,
    pub runtime: i16,
}

#[derive(Clone, Debug, Default, Serialize)]
pub struct ToolCallGraph {
    pub root: Node,
    pub number_of_nodes: i16,
    pub number_of_envs: i16,
}

impl ToolCallGraph {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn insert(
        &mut self,
        path: Vec<String>,
        value: Option<String>,
        sandbox_id: Option<String>,
        runtime: i16,
    ) -> (bool, bool) {
        if path.is_empty() {
            return (false, false);
        }

        let last_idx = path.len() - 1;
        let mut curr: &mut Node = &mut self.root;

        for (idx, p) in path.into_iter().enumerate() {
            if idx < last_idx {
                if let Some(next) = curr.children.get_mut(&p) {
                    curr = next;
                } else {
                    return (false, false);
                }
            } else {
                let new_node = match curr.children.entry(p) {
                    Entry::Occupied(entry) => entry.into_mut(),
                    Entry::Vacant(entry) => {
                        self.number_of_nodes += 1;
                        entry.insert(Node::default())
                    }
                };

                new_node.value = value;
                new_node.runtime = runtime;

                if sandbox_id.is_some() && new_node.sandbox_id.is_none() {
                    new_node.sandbox_id = sandbox_id;
                    self.number_of_envs += 1;
                    return (true, true);
                }

                return (true, false);
            }
        }

        (false, false)
    }

    pub fn find(
        &self,
        path: Vec<String>,
    ) -> (Option<String>, (Option<Vec<String>>, Option<String>)) {
        // this function tries to find the exact path in the tcg and return the value of the last node or return the longest prefix match with a sandbox id

        let mut curr: &Node = &self.root;
        let mut curr_path: Vec<String> = Vec::new();
        let mut sandbox_path: Option<Vec<String>> = None;
        let mut sandbox_id: Option<String> = None;
        let mut value_of_path: Option<String> = None;
        let path_len = path.len();

        if let Some(id) = &curr.sandbox_id {
            sandbox_id = Some(id.clone());
        }

        for p in path {
            if let Some(next) = curr.children.get(&p) {
                curr_path.push(p);
                curr = next;

                if let Some(id) = &curr.sandbox_id {
                    sandbox_id = Some(id.clone());
                    sandbox_path = Some(curr_path.clone());
                }

                if curr_path.len() == path_len {
                    value_of_path = curr.value.clone();
                }
            } else {
                // The path has diverged from the tcg, so return what you have seen so far, if value is found, return it
                break;
            }
        }

        return (value_of_path, (sandbox_path, sandbox_id));
    }
}
