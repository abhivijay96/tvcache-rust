use std::io;

use tokio::net::TcpStream;
use tracing::error;

use crate::{
    protocol_primitives,
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

                Ok(Request::Insert {
                    task_id,
                    path,
                    value,
                    sandbox_id,
                    runtime,
                    ttl,
                    ttl_present: ttl_exists == 1,
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
            } => {
                let tcg = {
                    let mut cache = self.cache.lock().await;
                    cache.get_tcg(task_id)
                };

                let (value_inserted, sandbox_inserted) = {
                    let mut tcg = tcg.lock().await;
                    tcg.insert(path, value, sandbox_id, runtime, ttl, ttl_present)
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

                // Resolve all TCG arcs in one cache lock, then lock TCGs one at a time.
                // get_tcg (creating) is intentional: a record for an unknown task_id still
                // lands in a victim list so the id flows back out through drain.
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
