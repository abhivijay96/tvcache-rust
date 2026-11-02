use serde::Serialize;
use std::collections::HashMap;
use std::mem::{size_of, size_of_val};

#[derive(Clone, Debug, Default, Serialize)]
pub struct ToolCallEntry {
    pub value: Option<String>,
    pub runtime: u32, // milliseconds
    pub embedding: Option<Vec<f32>>,
}

#[derive(Clone, Debug, Default, Serialize)]
pub struct Node {
    pub children: Vec<Node>,
    pub sandbox_id: Option<String>,
    pub tool_calls: HashMap<String, ToolCallEntry>,
    pub settled: bool, // if true, this node has been compared with every other settled sibling during contraction
    pub ttl_present: bool,
    pub ttl: u32, // seconds
    pub inserted_at_seconds: u64,
    pub number_of_cache_hits: u64,
    pub number_of_lpm_hits: u64,
    pub size_in_bytes: u32, // A single entry can't be more than 4GB, need to update this if we remove sandbox id later.
}

impl Node {
    fn child_index(&self, call: &str) -> Option<usize> {
        self.children
            .iter()
            .position(|child| child.tool_calls.contains_key(call))
    }

    pub fn child(&self, call: &str) -> Option<&Node> {
        Some(&self.children[self.child_index(call)?])
    }

    pub fn child_mut(&mut self, call: &str) -> Option<&mut Node> {
        let index = self.child_index(call)?;
        Some(&mut self.children[index])
    }

    /// Returns the child and whether it was newly created.
    pub fn get_or_create_child(&mut self, call: String) -> (&mut Node, bool) {
        if let Some(index) = self.child_index(&call) {
            return (&mut self.children[index], false);
        }
        let mut node = Node::default();
        node.tool_calls.insert(call, ToolCallEntry::default());
        self.children.push(node);
        (self.children.last_mut().unwrap(), true)
    }

    pub fn remove_child(&mut self, call: &str) -> Option<Node> {
        let index = self.child_index(call)?;
        Some(self.children.remove(index))
    }

    pub fn retain_children<F>(&mut self, keep: F)
    where
        F: FnMut(&mut Node) -> bool,
    {
        self.children.retain_mut(keep);
    }

    /// Yields one representative tool call and node per child.
    pub fn iter_children(&self) -> impl Iterator<Item = (&String, &Node)> {
        self.children.iter().map(|child| {
            let call = child.tool_calls.keys().next();
            (call.expect("Child without tool calls"), child)
        })
    }

    pub fn walk_subtcg<T, F>(&self, initial: T, f: &mut F) -> T
    where
        F: FnMut(T, &Node) -> T,
    {
        let mut acc = f(initial, self);

        for (_, child) in self.iter_children() {
            acc = child.walk_subtcg(acc, f);
        }

        return acc;
    }

    pub fn update_size_in_bytes(&mut self) {
        let entries_size_in_bytes: u32 = self
            .tool_calls
            .iter()
            .map(|(call, entry)| {
                call.len() as u32
                    + entry.value.as_ref().map_or(0, |value| value.len() as u32)
                    + size_of_val(&entry.runtime) as u32
                    + entry
                        .embedding
                        .as_ref()
                        .map_or(0, |embedding| (embedding.len() * size_of::<f32>()) as u32)
            })
            .sum();

        let sandbox_id_size_in_bytes = self
            .sandbox_id
            .as_ref()
            .map_or(0, |sandbox_id| sandbox_id.len() as u32);

        self.size_in_bytes = entries_size_in_bytes
            + sandbox_id_size_in_bytes
            + size_of_val(&self.settled) as u32
            + size_of_val(&self.ttl_present) as u32
            + size_of_val(&self.ttl) as u32
            + size_of_val(&self.inserted_at_seconds) as u32
            + size_of_val(&self.number_of_cache_hits) as u32
            + size_of_val(&self.number_of_lpm_hits) as u32;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn children_resolve_through_any_tool_call() {
        let mut root = Node::default();
        let (child, is_new) = root.get_or_create_child("a".into());
        assert!(is_new);
        child
            .tool_calls
            .insert("alias".into(), ToolCallEntry::default());
        let (child, is_new) = root.get_or_create_child("b".into());
        assert!(is_new && child.tool_calls.len() == 1 && child.tool_calls.contains_key("b"));

        let (child, is_new) = root.get_or_create_child("alias".into());
        assert!(!is_new && child.tool_calls.contains_key("a"));
        assert!(root.child("a").is_some() && root.child("missing").is_none());
        assert_eq!(root.iter_children().count(), 2);
        for (call, child) in root.iter_children() {
            assert!(child.tool_calls.contains_key(call));
        }

        assert!(root
            .remove_child("alias")
            .unwrap()
            .tool_calls
            .contains_key("a"));
        assert!(root.child("a").is_none() && root.child_mut("b").is_some());
        root.retain_children(|child| !child.tool_calls.contains_key("b"));
        assert!(root.children.is_empty());
    }
}
