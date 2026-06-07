use std::time::{SystemTime, UNIX_EPOCH};

pub fn current_utc_seconds() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("system clock is before Unix epoch")
        .as_secs()
}
