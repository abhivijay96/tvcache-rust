pub mod ttl_and_watermark;

#[derive(Clone, Debug, Default)]
pub enum EvictionPolicyConfig {
    #[default]
    None,
    TtlAndWatermark(TtlAndWatermarkEvictionConfig),
}

#[derive(Clone, Debug)]
pub struct TtlAndWatermarkEvictionConfig {
    pub high_watermark_mb: u32,
    pub low_watermark_mb: u32,
}

#[derive(Clone, Debug, Default)]
pub struct EvictionResult {
    pub nodes_removed: u32,
    pub sandboxes_removed: u32,
}

impl EvictionResult {
    pub fn update(&mut self, other: EvictionResult) {
        self.nodes_removed += other.nodes_removed;
        self.sandboxes_removed += other.sandboxes_removed;
    }
}
