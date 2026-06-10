use serde::Serialize;
use std::collections::HashMap;
use std::mem::size_of_val;

#[derive(Clone, Debug, Default, Serialize)]
pub struct Node {
    pub children: HashMap<String, Node>,
    pub sandbox_id: Option<String>,
    pub value: Option<String>,
    pub runtime: u32, // milliseconds
    pub ttl_present: bool,
    pub ttl: u32, // seconds
    pub inserted_at_seconds: u64,
    pub number_of_cache_hits: u64,
    pub number_of_lpm_hits: u64,
    pub size_in_bytes: u32, // A single entry can't be more than 4GB, need to update this if we remove sandbox id later.
}

impl Node {
    pub fn walk_subtcg<T, F>(&self, initial: T, f: &mut F) -> T
    where
        F: FnMut(T, &Node) -> T,
    {
        let mut acc = f(initial, self);

        for child in self.children.values() {
            acc = child.walk_subtcg(acc, f);
        }

        return acc;
    }

    pub fn update_size_in_bytes(&mut self, path_entry_size_in_bytes: u32) {
        let value_size_in_bytes = self.value.as_ref().map_or(0, |value| value.len() as u32);

        let sandbox_id_size_in_bytes = self
            .sandbox_id
            .as_ref()
            .map_or(0, |sandbox_id| sandbox_id.len() as u32);

        self.size_in_bytes = path_entry_size_in_bytes
            + value_size_in_bytes
            + sandbox_id_size_in_bytes
            + size_of_val(&self.runtime) as u32
            + size_of_val(&self.ttl_present) as u32
            + size_of_val(&self.ttl) as u32
            + size_of_val(&self.inserted_at_seconds) as u32
            + size_of_val(&self.number_of_cache_hits) as u32
            + size_of_val(&self.number_of_lpm_hits) as u32;
    }
}
