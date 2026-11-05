//! Adapter for tvcache-synthetic-v1 nodes.jsonl + events.jsonl.
use super::*;
use std::collections::HashMap;
use std::io::{self, BufRead};

#[derive(Debug, Deserialize)]
pub struct NodeRecord {
    pub task_id: String,
    pub node_id: String,
    pub parent_id: Option<String>,
    pub key: Vec<String>,
    pub depth: u64,
    pub first_seen_event: u64,
    pub cache_entry_bytes: u64,
    pub value_bytes: u64,
    pub runtime_ms: f64,
    pub tool_cost_usd: Option<f64>,
    pub snapshot_candidate: Option<SnapshotCost>,
    pub tool_call: ToolCall,
}

#[derive(Debug, Deserialize)]
pub struct ToolCall {
    pub tool: String,
    pub mutates_state: bool,
}

#[derive(Debug, Deserialize)]
pub struct AccessRecord {
    pub event: String,
    pub event_index: u64,
    pub time_ms: f64,
    pub task_id: String,
    pub node_id: String,
    pub rollout_id: String,
    pub features: AccessFeatures,
}

/// Metadata catalog is not a resident cache and must not be exposed to policies.
pub struct TraceCatalog(HashMap<EntryId, NodeRecord>);

fn invalid(message: impl Into<String>) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, message.into())
}

pub fn read_jsonl<T: serde::de::DeserializeOwned>(reader: impl BufRead) -> io::Result<Vec<T>> {
    reader
        .lines()
        .enumerate()
        .map(|(i, line)| {
            let line = line?;
            serde_json::from_str(&line).map_err(|e| invalid(format!("line {}: {e}", i + 1)))
        })
        .collect()
}

impl TraceCatalog {
    pub fn read(reader: impl BufRead) -> io::Result<Self> {
        let mut nodes = HashMap::new();
        for node in read_jsonl::<NodeRecord>(reader)? {
            let id = EntryId {
                task_id: node.task_id.clone(),
                node_id: node.node_id.clone(),
            };
            if nodes.insert(id.clone(), node).is_some() {
                return Err(invalid(format!("duplicate node: {id:?}")));
            }
        }
        Ok(Self(nodes))
    }

    /// Resolve only an observed request, avoiding exposure of future graph nodes.
    /// Cost fields are fixture estimates; live replay must supply past estimates.
    pub fn resolve(&self, event: &AccessRecord) -> io::Result<(EntryMeta, AccessContext)> {
        if event.event != "access" || !event.time_ms.is_finite() || event.time_ms < 0.0 {
            return Err(invalid(
                "expected access event with finite nonnegative time",
            ));
        }
        let id = EntryId {
            task_id: event.task_id.clone(),
            node_id: event.node_id.clone(),
        };
        let node = self
            .0
            .get(&id)
            .ok_or_else(|| invalid(format!("unknown node: {id:?}")))?;
        if node.first_seen_event > event.event_index {
            return Err(invalid("access precedes node first_seen_event"));
        }
        if !node.runtime_ms.is_finite()
            || node.runtime_ms < 0.0
            || node
                .tool_cost_usd
                .is_some_and(|v| !v.is_finite() || v < 0.0)
            || node.snapshot_candidate.as_ref().is_some_and(|s| {
                !s.create_ms.is_finite()
                    || s.create_ms < 0.0
                    || !s.restore_ms.is_finite()
                    || s.restore_ms < 0.0
            })
        {
            return Err(invalid("cost estimates must be finite and nonnegative"));
        }
        Ok((
            EntryMeta {
                parent: node.parent_id.as_ref().map(|p| EntryId {
                    task_id: id.task_id.clone(),
                    node_id: p.clone(),
                }),
                id,
                path: node.key.clone(),
                depth: node.depth,
                tool: node.tool_call.tool.clone(),
                mutates_state: node.tool_call.mutates_state,
                cache_entry_bytes: node.cache_entry_bytes,
                value_bytes: node.value_bytes,
                runtime_ms: Some(node.runtime_ms),
                tool_cost_usd: node.tool_cost_usd,
                snapshot: node.snapshot_candidate.clone(),
            },
            AccessContext {
                event_index: event.event_index,
                time_ms: event.time_ms,
            },
        ))
    }
}
