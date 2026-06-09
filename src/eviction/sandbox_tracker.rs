use std::collections::HashMap;

#[derive(Clone, Debug, Default)]
pub struct SandboxTracker {
    pub evicted_sandbox_ids: Vec<String>,
    pub references: HashMap<String, u32>,
}

impl SandboxTracker {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn record(&mut self, sandbox_id: &str) {
        self.evicted_sandbox_ids.push(sandbox_id.to_string());
    }

    pub fn record_many(&mut self, sandbox_ids: &[String]) {
        for sandbox_id in sandbox_ids {
            self.record(sandbox_id);
        }
    }

    pub fn get_reference_count(&self, sandbox_id: &str) -> u32 {
        *self.references.get(sandbox_id).unwrap_or(&0)
    }

    pub fn increment_reference(&mut self, sandbox_id: &str) -> u32 {
        let reference_count = self.references.entry(sandbox_id.to_string()).or_insert(0);
        *reference_count += 1;
        *reference_count
    }

    pub fn decrement_reference(&mut self, sandbox_id: &str) -> u32 {
        let Some(reference_count) = self.references.get_mut(sandbox_id) else {
            return 0;
        };

        *reference_count -= 1;

        if *reference_count == 0 {
            self.references.remove(sandbox_id);
            return 0;
        }

        *reference_count
    }
}
