use std::io;

use tokio::net::TcpStream;
use tracing::error;

use crate::{
    protocol_primitives,
    tcg::contraction::{class_call_count, SiblingGroup},
    tvcache::tcg_entries_to_json,
    types::{
        CacheSharedObject, ReclaimableSandbox, Request, Response, SandboxCandidate,
        SandboxEvictionRecord,
    },
};

pub struct RequestHandler {
    cache: CacheSharedObject,
}

impl RequestHandler {
    pub fn new(cache: CacheSharedObject) -> Self {
        return Self { cache: cache };
    }

    pub async fn handle_stream(&self, mut stream: TcpStream) {
        loop {
            match self.parse_request(&mut stream).await {
                Ok(request) => {
                    let response = self.execute_request(request).await;
                    if let Err(err) = self.write_response(&mut stream, response).await {
                        error!(error = %err, "error while writing response");
                    }
                }
                Err(_err) => {
                    break;
                }
            }
        }
    }

    async fn parse_request(&self, stream: &mut TcpStream) -> Result<Request, io::Error> {
        let request_type = protocol_primitives::read_u8(stream).await?;

        match request_type {
            1 => {
                let task_id = protocol_primitives::read_string_u16(stream).await?;
                let path = protocol_primitives::read_path(stream).await?;
                let mut value = None;
                let mut sandbox_id = None;
                let mut ttl: Option<u32> = None;
                let mut embedding = None;

                let runtime = protocol_primitives::read_u32(stream).await?;

                let value_exists = protocol_primitives::read_u8(stream).await?;

                if value_exists == 1 {
                    value = Some(protocol_primitives::read_string_u32(stream).await?);
                } else if value_exists != 0 {
                    return Err(io::Error::new(
                        io::ErrorKind::InvalidData,
                        format!("invalid value exists flag: {value_exists}"),
                    ));
                }

                // getting sandbox id
                let sandbox_id_exists = protocol_primitives::read_u8(stream).await?;

                if sandbox_id_exists == 1 {
                    sandbox_id = Some(protocol_primitives::read_string_u16(stream).await?);
                } else if sandbox_id_exists != 0 {
                    return Err(io::Error::new(
                        io::ErrorKind::InvalidData,
                        format!("invalid sandbox id exists flag: {sandbox_id_exists}"),
                    ));
                }

                let ttl_exists = protocol_primitives::read_u8(stream).await?;

                if ttl_exists == 1 {
                    ttl = Some(protocol_primitives::read_u32(stream).await?);
                } else if ttl_exists != 0 {
                    return Err(io::Error::new(
                        io::ErrorKind::InvalidData,
                        format!("invalid ttl exists flag: {ttl_exists}"),
                    ));
                }

                let embedding_exists = protocol_primitives::read_u8(stream).await?;

                if embedding_exists == 1 {
                    embedding = Some(protocol_primitives::read_embedding(stream).await?);
                } else if embedding_exists != 0 {
                    return Err(io::Error::new(
                        io::ErrorKind::InvalidData,
                        format!("invalid embedding exists flag: {embedding_exists}"),
                    ));
                }

                Ok(Request::Insert {
                    task_id,
                    path,
                    value,
                    sandbox_id,
                    runtime,
                    ttl,
                    ttl_present: ttl_exists == 1,
                    embedding,
                })
            }

            2 => {
                let task_id = protocol_primitives::read_string_u16(stream).await?;
                let path = protocol_primitives::read_path(stream).await?;

                Ok(Request::Find { task_id, path })
            }

            3 => Ok(Request::DumpAll),

            4 => {
                let task_id = protocol_primitives::read_string_u16(stream).await?;
                let sandbox_id = protocol_primitives::read_string_u16(stream).await?;

                Ok(Request::UnrefSandbox {
                    task_id,
                    sandbox_id,
                })
            }

            5 => {
                let sandbox_eviction_records = self.read_sandbox_eviction_records(stream).await?;

                Ok(Request::MarkSandboxesEvicted {
                    sandbox_eviction_records,
                })
            }

            6 => {
                let limit = protocol_primitives::read_u16(stream).await?;

                Ok(Request::ListSandboxCandidates { limit })
            }

            7 => Ok(Request::DrainReclaimableSandboxes),

            8 => {
                let limit = protocol_primitives::read_u16(stream).await?;

                Ok(Request::ListHotSandboxes { limit })
            }

            9 => {
                let task_id = protocol_primitives::read_string_u16(stream).await?;
                let fork_cost_ms = protocol_primitives::read_u32(stream).await?;

                Ok(Request::ReportForkCost {
                    task_id,
                    fork_cost_ms,
                })
            }

            10 => {
                let task_id = protocol_primitives::read_string_u16(stream).await?;

                Ok(Request::GetForkCost { task_id })
            }

            11 => {
                let task_id = protocol_primitives::read_string_u16(stream).await?;
                let path = protocol_primitives::read_path(stream).await?;

                Ok(Request::LeaseSandbox { task_id, path })
            }

            12 => {
                let task_id = protocol_primitives::read_string_u16(stream).await?;
                let depth = protocol_primitives::read_u16(stream).await?;
                let threshold = protocol_primitives::read_f32(stream).await?;

                Ok(Request::ListSiblingGroups {
                    task_id,
                    depth,
                    threshold,
                })
            }

            13 => {
                let task_id = protocol_primitives::read_string_u16(stream).await?;
                let parent_path = protocol_primitives::read_path(stream).await?;
                let class_count = protocol_primitives::read_u16(stream).await?;
                let mut classes = Vec::with_capacity(usize::from(class_count));
                for _ in 0..class_count {
                    classes.push(protocol_primitives::read_path(stream).await?);
                }
                let settled = protocol_primitives::read_path(stream).await?;

                Ok(Request::ApplyContraction {
                    task_id,
                    parent_path,
                    classes,
                    settled,
                })
            }

            _ => Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("unknown request type: {request_type}"),
            )),
        }
    }

    async fn execute_request(&self, request: Request) -> Response {
        match request {
            Request::Insert {
                task_id,
                path,
                value,
                sandbox_id,
                runtime,
                ttl_present,
                ttl,
                embedding,
            } => {
                let tcg = {
                    let mut cache = self.cache.lock().await;
                    cache.get_tcg(task_id)
                };

                let (value_inserted, sandbox_inserted) = {
                    let mut tcg = tcg.lock().await;
                    tcg.insert(path, value, sandbox_id, runtime, ttl, ttl_present, embedding)
                };

                Response::InsertResult {
                    value_inserted,
                    sandbox_inserted,
                }
            }

            Request::Find { task_id, path } => {
                let tcg = {
                    let mut cache = self.cache.lock().await;
                    cache.get_tcg(task_id)
                };

                {
                    let mut tcg = tcg.lock().await;
                    let (value, (sandbox_path, sandbox_id)) = { tcg.find(path) };

                    // The returned sandbox may now be in use by the rollout, so count
                    // the reference under the same per-TCG lock (no cross-TCG lock).
                    if let Some(id) = &sandbox_id {
                        tcg.increment_sandbox_reference(id);
                    }

                    Response::FindResult {
                        value,
                        sandbox_path,
                        sandbox_id,
                    }
                }
            }

            Request::DumpAll => {
                let entries = {
                    let cache = self.cache.lock().await;
                    cache.entries_snapshot()
                };

                let json_result = tcg_entries_to_json(entries).await;

                match json_result {
                    Ok(json) => Response::Json { json },
                    Err(error) => Response::Error {
                        message: error.to_string(),
                    },
                }
            }

            Request::UnrefSandbox {
                task_id,
                sandbox_id,
            } => {
                let tcg = {
                    let cache = self.cache.lock().await;
                    cache.peek_tcg(&task_id)
                };

                let remaining_references = match tcg {
                    Some(tcg) => {
                        let mut tcg = tcg.lock().await;
                        tcg.decrement_sandbox_reference(&sandbox_id)
                    }
                    None => 0,
                };

                Response::UnrefSandboxResult {
                    remaining_references,
                }
            }

            Request::MarkSandboxesEvicted {
                sandbox_eviction_records,
            } => {
                let sandbox_ids_recorded = sandbox_eviction_records.len() as u32;

                // Make a list of tuples, (sandbox eviction record, tcg). Using this list, the next loop will lock the tcg and mark the sandbox as evicted. I am using the idiomatic rust syntax to learn
                // if we would use traditional style, we would use the code below.

                // let mut records_with_tcgs = Vec::<(TcgSharedObject, SandboxEvictionRecord)>::new();

                // let mut cache = self.cache.lock().await;

                // for record in sandbox_eviction_records {
                //     records_with_tcgs.push((cache.get_tcg(record.task_id.clone()), record));
                // }

                let records_with_tcgs = {
                    let mut cache = self.cache.lock().await;

                    sandbox_eviction_records
                        .into_iter()
                        .map(|record| (cache.get_tcg(record.task_id.clone()), record))
                        .collect::<Vec<_>>()
                };

                for (tcg, record) in records_with_tcgs {
                    let mut tcg = tcg.lock().await;
                    tcg.mark_sandbox_evicted(&record.sandbox_path, &record.sandbox_id);
                }

                Response::MarkSandboxesEvictedResult {
                    sandbox_ids_recorded,
                }
            }

            Request::ListSandboxCandidates { limit } => {
                // Ascending lpm-hit score: the least useful sandboxes first, for eviction.
                let candidates = self.collect_sorted_candidates(limit, false).await;

                Response::SandboxCandidates { candidates }
            }

            Request::ListHotSandboxes { limit } => {
                // Descending lpm-hit score: the most useful sandboxes first, for
                // proactive forking before a training iteration.
                let candidates = self.collect_sorted_candidates(limit, true).await;

                Response::SandboxCandidates { candidates }
            }

            Request::DrainReclaimableSandboxes => {
                let entries = {
                    let cache = self.cache.lock().await;
                    cache.entries_snapshot()
                };

                let mut sandboxes = Vec::new();

                for (task_id, tcg) in entries {
                    let mut tcg = tcg.lock().await;
                    let drained = tcg.drain_reclaimable_sandboxes();

                    sandboxes.extend(drained.into_iter().map(|sandbox_id| ReclaimableSandbox {
                        task_id: task_id.clone(),
                        sandbox_id,
                    }));
                }

                Response::ReclaimableSandboxes { sandboxes }
            }

            Request::ReportForkCost {
                task_id,
                fork_cost_ms,
            } => {
                // get_tcg (creating) so the first samples for a task (e.g. fork
                // bank warm-up) are kept even before any insert.
                let tcg = {
                    let mut cache = self.cache.lock().await;
                    cache.get_tcg(task_id)
                };

                let updated_ema = {
                    let mut tcg = tcg.lock().await;
                    tcg.record_fork_cost(fork_cost_ms)
                };

                Response::ForkCost {
                    fork_cost_ms: updated_ema,
                }
            }

            Request::GetForkCost { task_id } => {
                let tcg = {
                    let cache = self.cache.lock().await;
                    cache.peek_tcg(&task_id)
                };

                let fork_cost_ms = match tcg {
                    Some(tcg) => {
                        let tcg = tcg.lock().await;
                        tcg.fork_cost_ema_ms
                    }
                    None => 0,
                };

                Response::ForkCost { fork_cost_ms }
            }

            Request::LeaseSandbox { task_id, path } => {
                let tcg = {
                    let cache = self.cache.lock().await;
                    cache.peek_tcg(&task_id)
                };

                let (sandbox_path, sandbox_id) = match tcg {
                    Some(tcg) => {
                        let mut tcg = tcg.lock().await;
                        tcg.lease_sandbox(&path)
                    }
                    None => (None, None),
                };

                Response::LeaseResult {
                    sandbox_path,
                    sandbox_id,
                }
            }

            Request::ListSiblingGroups {
                task_id,
                depth,
                threshold,
            } => {
                let tcg = {
                    let cache = self.cache.lock().await;
                    cache.peek_tcg(&task_id)
                };

                let (deeper_nodes_exist, groups) = match tcg {
                    Some(tcg) => {
                        let tcg = tcg.lock().await;
                        tcg.sibling_groups(usize::from(depth), threshold)
                    }
                    None => (false, Vec::new()),
                };

                if sibling_groups_fit(&groups) {
                    Response::SiblingGroups {
                        deeper_nodes_exist,
                        groups,
                    }
                } else {
                    Response::Error {
                        message: "sibling groups exceed protocol limits".to_string(),
                    }
                }
            }

            Request::ApplyContraction {
                task_id,
                parent_path,
                classes,
                settled,
            } => {
                let tcg = {
                    let cache = self.cache.lock().await;
                    cache.peek_tcg(&task_id)
                };

                // An unknown task skips every call, like a missing parent.
                let (merged_classes, skipped_calls) = match tcg {
                    Some(tcg) => {
                        let mut tcg = tcg.lock().await;
                        tcg.apply_contraction(&parent_path, &classes, &settled)
                    }
                    None => (0, class_call_count(&classes)),
                };

                Response::ContractionResult {
                    merged_classes,
                    skipped_calls,
                }
            }
        }
    }

    async fn write_response(
        &self,
        stream: &mut TcpStream,
        response: Response,
    ) -> Result<(), io::Error> {
        match response {
            Response::InsertResult {
                value_inserted,
                sandbox_inserted,
            } => {
                protocol_primitives::write_u8(stream, 1).await?;
                protocol_primitives::write_bool(stream, value_inserted).await?;
                protocol_primitives::write_bool(stream, sandbox_inserted).await?;
            }

            Response::FindResult {
                value,
                sandbox_path,
                sandbox_id,
            } => {
                protocol_primitives::write_u8(stream, 2).await?;

                if let Some(value) = value {
                    protocol_primitives::write_bool(stream, true).await?;
                    protocol_primitives::write_string_u32(stream, &value).await?;
                } else {
                    protocol_primitives::write_bool(stream, false).await?;
                }

                self.write_sandbox_location(stream, sandbox_path, sandbox_id)
                    .await?;
            }

            Response::UnrefSandboxResult {
                remaining_references,
            } => {
                protocol_primitives::write_u8(stream, 5).await?;
                protocol_primitives::write_u32(stream, remaining_references).await?;
            }

            Response::MarkSandboxesEvictedResult {
                sandbox_ids_recorded,
            } => {
                protocol_primitives::write_u8(stream, 6).await?;
                protocol_primitives::write_u32(stream, sandbox_ids_recorded).await?;
            }

            Response::SandboxCandidates { candidates } => {
                protocol_primitives::write_u8(stream, 7).await?;
                self.write_sandbox_candidates(stream, &candidates).await?;
            }

            Response::ReclaimableSandboxes { sandboxes } => {
                protocol_primitives::write_u8(stream, 8).await?;

                // u32 length: a large TTL sweep can produce more than u16::MAX victims.
                let sandbox_count = u32::try_from(sandboxes.len()).map_err(|error| {
                    io::Error::new(
                        io::ErrorKind::InvalidInput,
                        format!("too many reclaimable sandboxes for u32 length: {error}"),
                    )
                })?;

                protocol_primitives::write_u32(stream, sandbox_count).await?;

                for sandbox in &sandboxes {
                    protocol_primitives::write_string_u16(stream, &sandbox.task_id).await?;
                    protocol_primitives::write_string_u16(stream, &sandbox.sandbox_id).await?;
                }
            }

            Response::ForkCost { fork_cost_ms } => {
                protocol_primitives::write_u8(stream, 9).await?;
                protocol_primitives::write_u32(stream, fork_cost_ms).await?;
            }

            Response::LeaseResult {
                sandbox_path,
                sandbox_id,
            } => {
                protocol_primitives::write_u8(stream, 10).await?;
                self.write_sandbox_location(stream, sandbox_path, sandbox_id)
                    .await?;
            }

            Response::SiblingGroups {
                deeper_nodes_exist,
                groups,
            } => {
                protocol_primitives::write_u8(stream, 11).await?;
                protocol_primitives::write_bool(stream, deeper_nodes_exist).await?;
                self.write_sibling_groups(stream, &groups).await?;
            }

            Response::ContractionResult {
                merged_classes,
                skipped_calls,
            } => {
                protocol_primitives::write_u8(stream, 12).await?;
                protocol_primitives::write_u32(stream, merged_classes).await?;
                protocol_primitives::write_u32(stream, skipped_calls).await?;
            }

            Response::Error { message } => {
                protocol_primitives::write_u8(stream, 3).await?;
                protocol_primitives::write_string_u32(stream, &message).await?;
            }

            Response::Json { json } => {
                protocol_primitives::write_u8(stream, 4).await?;
                protocol_primitives::write_string_u32(stream, &json).await?;
            }
        }

        Ok(())
    }

    async fn collect_sorted_candidates(
        &self,
        limit: u16,
        descending: bool,
    ) -> Vec<SandboxCandidate> {
        let limit = if limit == 0 { 10 } else { limit } as usize;

        let mut candidates = Vec::<SandboxCandidate>::new();
        let entries = {
            let cache = self.cache.lock().await;
            cache.entries_snapshot()
        };

        for (task_id, tcg) in entries {
            let tcg = tcg.lock().await;
            candidates.extend(tcg.collect_sandbox_candidates(&task_id));
        }

        candidates.sort_by_key(|candidate| candidate.score);
        if descending {
            candidates.reverse();
        }
        candidates.truncate(limit);

        candidates
    }

    async fn write_sandbox_location(
        &self,
        stream: &mut TcpStream,
        sandbox_path: Option<Vec<String>>,
        sandbox_id: Option<String>,
    ) -> Result<(), io::Error> {
        if let Some(sandbox_path) = sandbox_path {
            protocol_primitives::write_bool(stream, true).await?;
            protocol_primitives::write_path(stream, &sandbox_path).await?;
        } else {
            protocol_primitives::write_bool(stream, false).await?;
        }

        if let Some(sandbox_id) = sandbox_id {
            protocol_primitives::write_bool(stream, true).await?;
            protocol_primitives::write_string_u16(stream, &sandbox_id).await?;
        } else {
            protocol_primitives::write_bool(stream, false).await?;
        }

        Ok(())
    }

    // Counts and indexes were checked by sibling_groups_fit.
    async fn write_sibling_groups(
        &self,
        stream: &mut TcpStream,
        groups: &[SiblingGroup],
    ) -> Result<(), io::Error> {
        protocol_primitives::write_u16(stream, groups.len() as u16).await?;

        for group in groups {
            protocol_primitives::write_path(stream, &group.parent_path).await?;
            protocol_primitives::write_u16(stream, group.children.len() as u16).await?;

            for sibling in &group.children {
                protocol_primitives::write_path(stream, &sibling.calls).await?;
                protocol_primitives::write_bool(stream, sibling.settled).await?;
                protocol_primitives::write_bool(stream, sibling.has_snapshot).await?;
            }

            protocol_primitives::write_u32(stream, group.pairs.len() as u32).await?;

            for &(i, j, score) in &group.pairs {
                protocol_primitives::write_u16(stream, i as u16).await?;
                protocol_primitives::write_u16(stream, j as u16).await?;
                protocol_primitives::write_f32(stream, score).await?;
            }
        }

        Ok(())
    }

    async fn read_sandbox_eviction_records(
        &self,
        stream: &mut TcpStream,
    ) -> Result<Vec<SandboxEvictionRecord>, io::Error> {
        let record_count = protocol_primitives::read_u16(stream).await? as usize;
        let mut records = Vec::with_capacity(record_count);

        for _ in 0..record_count {
            let task_id = protocol_primitives::read_string_u16(stream).await?;
            let sandbox_path = protocol_primitives::read_path(stream).await?;
            let sandbox_id = protocol_primitives::read_string_u16(stream).await?;

            records.push(SandboxEvictionRecord {
                task_id,
                sandbox_path,
                sandbox_id,
            });
        }

        Ok(records)
    }

    async fn write_sandbox_candidates(
        &self,
        stream: &mut TcpStream,
        candidates: &[SandboxCandidate],
    ) -> Result<(), io::Error> {
        let candidate_count = u16::try_from(candidates.len()).map_err(|error| {
            io::Error::new(
                io::ErrorKind::InvalidInput,
                format!("too many sandbox candidates for u16 length: {error}"),
            )
        })?;

        protocol_primitives::write_u16(stream, candidate_count).await?;

        for candidate in candidates {
            protocol_primitives::write_string_u16(stream, &candidate.task_id).await?;
            protocol_primitives::write_path(stream, &candidate.sandbox_path).await?;
            protocol_primitives::write_string_u16(stream, &candidate.sandbox_id).await?;
            protocol_primitives::write_u64(stream, candidate.score).await?;
        }

        Ok(())
    }
}

/// Returns whether every count and index fits its protocol field.
fn sibling_groups_fit(groups: &[SiblingGroup]) -> bool {
    let fits_u16 = |count: usize| u16::try_from(count).is_ok();
    fits_u16(groups.len())
        && groups.iter().all(|group| {
            fits_u16(group.parent_path.len())
                && fits_u16(group.children.len())
                && group
                    .children
                    .iter()
                    .all(|sibling| fits_u16(sibling.calls.len()))
                && u32::try_from(group.pairs.len()).is_ok()
        })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tcg::contraction::Sibling;

    #[test]
    fn sibling_groups_fit_checks_protocol_limits() {
        let group = |siblings: usize| SiblingGroup {
            parent_path: Vec::new(),
            children: (0..siblings)
                .map(|_| Sibling {
                    calls: vec!["a".into()],
                    settled: false,
                    has_snapshot: false,
                })
                .collect(),
            pairs: vec![(0, 1, 1.0)],
        };
        let limit = usize::from(u16::MAX);
        assert!(sibling_groups_fit(&[group(limit)]));
        assert!(!sibling_groups_fit(&[group(limit + 1)]));
    }
}
