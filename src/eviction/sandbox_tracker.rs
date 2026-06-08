#[derive(Clone, Debug, Default)]
pub struct SandboxTracker {
    pub evicted_sandbox_ids: Vec<String>,
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
}
