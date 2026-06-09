use std::io;

use tokio::net::TcpStream;
use tracing::error;

use crate::{
    protocol_primitives,
    tvcache::tcg_entries_to_json,
    types::{
        CacheSharedObject, Request, Response, SandboxCandidate, SandboxEvictionRecord,
        SandboxTrackerSharedObject,
    },
};

pub struct RequestHandler {
    cache: CacheSharedObject,
    sandbox_tracker: SandboxTrackerSharedObject,
}

impl RequestHandler {
    pub fn new(cache: CacheSharedObject, sandbox_tracker: SandboxTrackerSharedObject) -> Self {
        return Self {
            cache: cache,
            sandbox_tracker: sandbox_tracker,
        };
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

                let runtime = protocol_primitives::read_u16(stream).await?;

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
                let sandbox_id = protocol_primitives::read_string_u16(stream).await?;

                Ok(Request::UnrefSandbox { sandbox_id })
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

                    if let Some(id) = &sandbox_id {
                        let mut sandbox_tracker = self.sandbox_tracker.lock().await;
                        sandbox_tracker.increment_reference(id);
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
            Request::UnrefSandbox { sandbox_id } => {
                let remaining_references = {
                    let mut sandbox_tracker = self.sandbox_tracker.lock().await;
                    sandbox_tracker.decrement_reference(&sandbox_id)
                };

                Response::UnrefSandboxResult {
                    remaining_references,
                }
            }
            Request::MarkSandboxesEvicted {
                sandbox_eviction_records,
            } => {
                let sandbox_ids_recorded = sandbox_eviction_records.len() as u32;

                {
                    let mut sandbox_tracker = self.sandbox_tracker.lock().await;
                    let sandbox_ids = sandbox_eviction_records
                        .iter()
                        .map(|record| {
                            let _ = (&record.task_id, &record.sandbox_path);
                            record.sandbox_id.clone()
                        })
                        .collect::<Vec<String>>();
                    sandbox_tracker.record_many(&sandbox_ids);
                }

                Response::MarkSandboxesEvictedResult {
                    sandbox_ids_recorded,
                }
            }
            Request::ListSandboxCandidates { limit } => {
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
                candidates.truncate(limit);

                Response::SandboxCandidates { candidates }
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
