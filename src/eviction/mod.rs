pub mod sandbox_tracker;
pub mod ttl_and_watermark;

#[derive(Clone, Debug, Default)]
pub enum EvictionPolicyName {
    #[default]
    None,
    TtlAndWatermark,
}

#[derive(Clone, Debug)]
pub struct EvictionPolicyConfig {
    pub policy_name: EvictionPolicyName,
    pub high_watermark_mb: u32,
    pub low_watermark_mb: u32,
}

impl EvictionPolicyConfig {
    pub fn new(
        policy_name: EvictionPolicyName,
        high_watermark_mb: u32,
        low_watermark_mb: u32,
    ) -> Self {
        Self {
            policy_name,
            high_watermark_mb,
            low_watermark_mb,
        }
    }
}

impl Default for EvictionPolicyConfig {
    fn default() -> Self {
        Self {
            policy_name: EvictionPolicyName::None,
            high_watermark_mb: 1024,
            low_watermark_mb: 768,
        }
    }
}

#[derive(Clone, Debug, Default)]
pub struct EvictionResult {
    pub nodes_removed: u32,
    pub sandboxes_removed: u32,
    pub size_removed_in_bytes: u64,
    pub sandbox_ids_removed: Vec<String>,
}

impl EvictionResult {
    pub fn update(&mut self, other: EvictionResult) {
        self.nodes_removed += other.nodes_removed;
        self.sandboxes_removed += other.sandboxes_removed;
        self.size_removed_in_bytes += other.size_removed_in_bytes;
        self.sandbox_ids_removed.extend(other.sandbox_ids_removed);
    }
}
