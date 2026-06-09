use std::io;

use tokio::{
    io::{AsyncReadExt, AsyncWriteExt},
    net::TcpStream,
};

pub async fn read_u8(stream: &mut TcpStream) -> Result<u8, io::Error> {
    let mut buffer = [0; 1];
    stream.read_exact(&mut buffer).await?;
    Ok(buffer[0])
}

pub async fn read_u16(stream: &mut TcpStream) -> Result<u16, io::Error> {
    let mut buffer = [0; 2];
    stream.read_exact(&mut buffer).await?;
    Ok(u16::from_be_bytes(buffer))
}

pub async fn read_u32(stream: &mut TcpStream) -> Result<u32, io::Error> {
    let mut buffer = [0; 4];
    stream.read_exact(&mut buffer).await?;
    Ok(u32::from_be_bytes(buffer))
}

pub async fn read_string_u16(stream: &mut TcpStream) -> Result<String, io::Error> {
    let len = read_u16(stream).await? as usize;
    read_string_bytes(stream, len).await
}

pub async fn read_string_u32(stream: &mut TcpStream) -> Result<String, io::Error> {
    let len = read_u32(stream).await? as usize;
    read_string_bytes(stream, len).await
}

async fn read_string_bytes(stream: &mut TcpStream, len: usize) -> Result<String, io::Error> {
    let mut buffer = vec![0; len];
    stream.read_exact(&mut buffer).await?;

    String::from_utf8(buffer).map_err(|error| io::Error::new(io::ErrorKind::InvalidData, error))
}

pub async fn read_path(stream: &mut TcpStream) -> Result<Vec<String>, io::Error> {
    let path_count = read_u16(stream).await? as usize;
    let mut path = Vec::with_capacity(path_count);

    for _ in 0..path_count {
        path.push(read_string_u16(stream).await?);
    }

    Ok(path)
}

pub async fn write_u8(stream: &mut TcpStream, value: u8) -> Result<(), io::Error> {
    stream.write_all(&[value]).await
}

pub async fn write_bool(stream: &mut TcpStream, value: bool) -> Result<(), io::Error> {
    write_u8(stream, if value { 1 } else { 0 }).await
}

pub async fn write_u16(stream: &mut TcpStream, value: u16) -> Result<(), io::Error> {
    stream.write_all(&value.to_be_bytes()).await
}

pub async fn write_u32(stream: &mut TcpStream, value: u32) -> Result<(), io::Error> {
    stream.write_all(&value.to_be_bytes()).await
}

pub async fn write_u64(stream: &mut TcpStream, value: u64) -> Result<(), io::Error> {
    stream.write_all(&value.to_be_bytes()).await
}

pub async fn write_string_u16(stream: &mut TcpStream, value: &str) -> Result<(), io::Error> {
    let bytes = value.as_bytes();
    let len = u16::try_from(bytes.len()).map_err(|error| {
        io::Error::new(
            io::ErrorKind::InvalidInput,
            format!("string is too large for u16 length: {error}"),
        )
    })?;

    write_u16(stream, len).await?;
    stream.write_all(bytes).await
}

pub async fn write_string_u32(stream: &mut TcpStream, value: &str) -> Result<(), io::Error> {
    let bytes = value.as_bytes();
    let len = u32::try_from(bytes.len()).map_err(|error| {
        io::Error::new(
            io::ErrorKind::InvalidInput,
            format!("string is too large for u32 length: {error}"),
        )
    })?;

    write_u32(stream, len).await?;
    stream.write_all(bytes).await
}

pub async fn write_path(stream: &mut TcpStream, path: &[String]) -> Result<(), io::Error> {
    let path_count = u16::try_from(path.len()).map_err(|error| {
        io::Error::new(
            io::ErrorKind::InvalidInput,
            format!("path has too many entries for u16 length: {error}"),
        )
    })?;

    write_u16(stream, path_count).await?;

    for entry in path {
        write_string_u16(stream, entry).await?;
    }

    Ok(())
}
