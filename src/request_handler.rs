#![allow(dead_code)]

use std::{
    io::{self, Read, Write},
    net::TcpStream,
};

use crate::types::{CacheSharedObject, Request, Response};

pub struct RequestHandler {
    cache: CacheSharedObject,
}

impl RequestHandler {
    pub fn new(cache: CacheSharedObject) -> Self {
        return Self { cache: cache };
    }

    pub fn handle_stream(&mut self, mut stream: TcpStream) {
        match self.parse_request(&mut stream) {
            Ok(request) => {
                let response = self.handle_request(request);
                if let Err(err) = self.write_response(&mut stream, response) {
                    println!("Error while writing response: {err}");
                }
            }
            Err(err) => {
                println!("Error while handling stream: {err}");
            }
        }
    }

    fn parse_request(&mut self, stream: &mut TcpStream) -> Result<Request, io::Error> {
        let request_type = self.read_u8(stream)?;

        match request_type {
            1 => {
                let task_id = self.read_string_u16(stream)?;
                let path = self.read_path(stream)?;
                let mut value = None;
                let mut sandbox_id = None;

                let runtime = self.read_i16(stream)?;

                let value_exists = self.read_u8(stream)?;

                if value_exists == 1 {
                    value = Some(self.read_string_u32(stream)?);
                } else if value_exists != 0 {
                    return Err(io::Error::new(
                        io::ErrorKind::InvalidData,
                        format!("invalid value exists flag: {value_exists}"),
                    ));
                }

                // getting sandbox id
                let sandbox_id_exists = self.read_u8(stream)?;

                if sandbox_id_exists == 1 {
                    sandbox_id = Some(self.read_string_u16(stream)?);
                } else if sandbox_id_exists != 0 {
                    return Err(io::Error::new(
                        io::ErrorKind::InvalidData,
                        format!("invalid sandbox id exists flag: {sandbox_id_exists}"),
                    ));
                }

                Ok(Request::Insert {
                    task_id,
                    path,
                    value,
                    sandbox_id,
                    runtime,
                })
            }

            2 => {
                let task_id = self.read_string_u16(stream)?;
                let path = self.read_path(stream)?;

                Ok(Request::Find { task_id, path })
            }

            3 => Ok(Request::DumpAll),

            _ => Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("unknown request type: {request_type}"),
            )),
        }
    }

    fn handle_request(&mut self, request: Request) -> Response {
        match request {
            Request::Insert {
                task_id,
                path,
                value,
                sandbox_id,
                runtime,
            } => {
                let tcg = {
                    let mut cache = self.cache.lock().unwrap();
                    cache.get_tcg(task_id)
                };

                let (value_inserted, sandbox_inserted) = {
                    let mut tcg = tcg.lock().unwrap();
                    tcg.insert(path, value, sandbox_id, runtime)
                };

                Response::InsertResult {
                    value_inserted,
                    sandbox_inserted,
                }
            }
            Request::Find { task_id, path } => {
                let tcg = {
                    let mut cache = self.cache.lock().unwrap();
                    cache.get_tcg(task_id)
                };

                let (value, (sandbox_path, sandbox_id)) = {
                    let tcg = tcg.lock().unwrap();
                    tcg.find(path)
                };

                Response::FindResult {
                    value,
                    sandbox_path,
                    sandbox_id,
                }
            }
            Request::DumpAll => {
                let json_result = {
                    let cache = self.cache.lock().unwrap();
                    cache.to_json()
                };

                match json_result {
                    Ok(json) => Response::Json { json },
                    Err(error) => Response::Error {
                        message: error.to_string(),
                    },
                }
            }
        }
    }

    fn write_response(
        &mut self,
        stream: &mut TcpStream,
        response: Response,
    ) -> Result<(), io::Error> {
        match response {
            Response::InsertResult {
                value_inserted,
                sandbox_inserted,
            } => {
                self.write_u8(stream, 1)?;
                self.write_bool(stream, value_inserted)?;
                self.write_bool(stream, sandbox_inserted)?;
            }
            Response::FindResult {
                value,
                sandbox_path,
                sandbox_id,
            } => {
                self.write_u8(stream, 2)?;

                if let Some(value) = value {
                    self.write_bool(stream, true)?;
                    self.write_string_u32(stream, &value)?;
                } else {
                    self.write_bool(stream, false)?;
                }

                if let Some(sandbox_path) = sandbox_path {
                    self.write_bool(stream, true)?;
                    self.write_path(stream, &sandbox_path)?;
                } else {
                    self.write_bool(stream, false)?;
                }

                if let Some(sandbox_id) = sandbox_id {
                    self.write_bool(stream, true)?;
                    self.write_string_u16(stream, &sandbox_id)?;
                } else {
                    self.write_bool(stream, false)?;
                }
            }
            Response::Error { message } => {
                self.write_u8(stream, 3)?;
                self.write_string_u32(stream, &message)?;
            }
            Response::Json { json } => {
                self.write_u8(stream, 4)?;
                self.write_string_u32(stream, &json)?;
            }
        }

        Ok(())
    }

    fn read_u8(&mut self, stream: &mut TcpStream) -> Result<u8, io::Error> {
        let mut buffer = [0; 1];
        stream.read_exact(&mut buffer)?;
        Ok(buffer[0])
    }

    fn read_u16(&mut self, stream: &mut TcpStream) -> Result<u16, io::Error> {
        let mut buffer = [0; 2];
        stream.read_exact(&mut buffer)?;
        Ok(u16::from_be_bytes(buffer))
    }

    fn read_i16(&mut self, stream: &mut TcpStream) -> Result<i16, io::Error> {
        let mut buffer = [0; 2];
        stream.read_exact(&mut buffer)?;
        Ok(i16::from_be_bytes(buffer))
    }

    fn read_u32(&mut self, stream: &mut TcpStream) -> Result<u32, io::Error> {
        let mut buffer = [0; 4];
        stream.read_exact(&mut buffer)?;
        Ok(u32::from_be_bytes(buffer))
    }

    fn read_string_u16(&mut self, stream: &mut TcpStream) -> Result<String, io::Error> {
        let len = self.read_u16(stream)? as usize;
        self.read_string_bytes(stream, len)
    }

    fn read_string_u32(&mut self, stream: &mut TcpStream) -> Result<String, io::Error> {
        let len = self.read_u32(stream)? as usize;
        self.read_string_bytes(stream, len)
    }

    fn read_string_bytes(
        &mut self,
        stream: &mut TcpStream,
        len: usize,
    ) -> Result<String, io::Error> {
        let mut buffer = vec![0; len];
        stream.read_exact(&mut buffer)?;

        String::from_utf8(buffer).map_err(|error| io::Error::new(io::ErrorKind::InvalidData, error))
    }

    fn read_path(&mut self, stream: &mut TcpStream) -> Result<Vec<String>, io::Error> {
        let path_count = self.read_u16(stream)? as usize;
        let mut path = Vec::with_capacity(path_count);

        for _ in 0..path_count {
            path.push(self.read_string_u16(stream)?);
        }

        Ok(path)
    }

    fn write_u8(&mut self, stream: &mut TcpStream, value: u8) -> Result<(), io::Error> {
        stream.write_all(&[value])
    }

    fn write_bool(&mut self, stream: &mut TcpStream, value: bool) -> Result<(), io::Error> {
        self.write_u8(stream, if value { 1 } else { 0 })
    }

    fn write_u16(&mut self, stream: &mut TcpStream, value: u16) -> Result<(), io::Error> {
        stream.write_all(&value.to_be_bytes())
    }

    fn write_u32(&mut self, stream: &mut TcpStream, value: u32) -> Result<(), io::Error> {
        stream.write_all(&value.to_be_bytes())
    }

    fn write_string_u16(&mut self, stream: &mut TcpStream, value: &str) -> Result<(), io::Error> {
        let bytes = value.as_bytes();
        let len = u16::try_from(bytes.len()).map_err(|error| {
            io::Error::new(
                io::ErrorKind::InvalidInput,
                format!("string is too large for u16 length: {error}"),
            )
        })?;

        self.write_u16(stream, len)?;
        stream.write_all(bytes)
    }

    fn write_string_u32(&mut self, stream: &mut TcpStream, value: &str) -> Result<(), io::Error> {
        let bytes = value.as_bytes();
        let len = u32::try_from(bytes.len()).map_err(|error| {
            io::Error::new(
                io::ErrorKind::InvalidInput,
                format!("string is too large for u32 length: {error}"),
            )
        })?;

        self.write_u32(stream, len)?;
        stream.write_all(bytes)
    }

    fn write_path(&mut self, stream: &mut TcpStream, path: &[String]) -> Result<(), io::Error> {
        let path_count = u16::try_from(path.len()).map_err(|error| {
            io::Error::new(
                io::ErrorKind::InvalidInput,
                format!("path has too many entries for u16 length: {error}"),
            )
        })?;

        self.write_u16(stream, path_count)?;

        for entry in path {
            self.write_string_u16(stream, entry)?;
        }

        Ok(())
    }
}
