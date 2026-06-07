#!/usr/bin/env python3

import asyncio
import json
import socket
import struct
import time
from dataclasses import dataclass
from typing import Any


INSERT_REQUEST = 1
FIND_REQUEST = 2
DUMP_ALL_REQUEST = 3

INSERT_RESULT_RESPONSE = 1
FIND_RESULT_RESPONSE = 2
ERROR_RESPONSE = 3
JSON_RESPONSE = 4


def write_u8(value: int) -> bytes:
    return struct.pack(">B", value)


def write_bool(value: bool) -> bytes:
    return write_u8(1 if value else 0)


def write_u16(value: int) -> bytes:
    return struct.pack(">H", value)


def write_u32(value: int) -> bytes:
    return struct.pack(">I", value)


def write_string_u16(value: str) -> bytes:
    encoded = value.encode("utf-8")
    return write_u16(len(encoded)) + encoded


def write_string_u32(value: str) -> bytes:
    encoded = value.encode("utf-8")
    return write_u32(len(encoded)) + encoded


def write_path(path: list[str]) -> bytes:
    encoded = write_u16(len(path))
    for entry in path:
        encoded += write_string_u16(entry)
    return encoded


def encode_insert(
    task_id: str,
    path: list[str],
    value: str | None,
    sandbox_id: str | None,
    runtime: int,
    ttl: int | None = None,
) -> bytes:
    encoded = write_u8(INSERT_REQUEST)
    encoded += write_string_u16(task_id)
    encoded += write_path(path)
    encoded += write_u16(runtime)

    encoded += write_bool(value is not None)
    if value is not None:
        encoded += write_string_u32(value)

    encoded += write_bool(sandbox_id is not None)
    if sandbox_id is not None:
        encoded += write_string_u16(sandbox_id)

    encoded += write_bool(ttl is not None)
    if ttl is not None:
        encoded += write_u32(ttl)

    return encoded


def encode_find(task_id: str, path: list[str]) -> bytes:
    return write_u8(FIND_REQUEST) + write_string_u16(task_id) + write_path(path)


def encode_dump_all() -> bytes:
    return write_u8(DUMP_ALL_REQUEST)


def read_exact(sock: socket.socket, size: int) -> bytes:
    chunks = bytearray()
    while len(chunks) < size:
        chunk = sock.recv(size - len(chunks))
        if not chunk:
            raise EOFError(f"expected {size} bytes, got {len(chunks)} bytes")
        chunks.extend(chunk)
    return bytes(chunks)


def read_u8(sock: socket.socket) -> int:
    return struct.unpack(">B", read_exact(sock, 1))[0]


def read_bool(sock: socket.socket) -> bool:
    value = read_u8(sock)
    if value == 0:
        return False
    if value == 1:
        return True
    raise ValueError(f"invalid bool byte: {value}")


def read_u16(sock: socket.socket) -> int:
    return struct.unpack(">H", read_exact(sock, 2))[0]


def read_u32(sock: socket.socket) -> int:
    return struct.unpack(">I", read_exact(sock, 4))[0]


def read_string_u16(sock: socket.socket) -> str:
    return read_exact(sock, read_u16(sock)).decode("utf-8")


def read_string_u32(sock: socket.socket) -> str:
    return read_exact(sock, read_u32(sock)).decode("utf-8")


def read_path(sock: socket.socket) -> list[str]:
    count = read_u16(sock)
    return [read_string_u16(sock) for _ in range(count)]


def read_optional_string_u16(sock: socket.socket) -> str | None:
    if read_bool(sock):
        return read_string_u16(sock)
    return None


def read_optional_string_u32(sock: socket.socket) -> str | None:
    if read_bool(sock):
        return read_string_u32(sock)
    return None


def read_optional_path(sock: socket.socket) -> list[str] | None:
    if read_bool(sock):
        return read_path(sock)
    return None


def decode_response(sock: socket.socket) -> dict[str, Any]:
    response_type = read_u8(sock)

    if response_type == INSERT_RESULT_RESPONSE:
        return {
            "type": "insert_result",
            "value_inserted": read_bool(sock),
            "sandbox_inserted": read_bool(sock),
        }

    if response_type == FIND_RESULT_RESPONSE:
        return {
            "type": "find_result",
            "value": read_optional_string_u32(sock),
            "sandbox_path": read_optional_path(sock),
            "sandbox_id": read_optional_string_u16(sock),
        }

    if response_type == ERROR_RESPONSE:
        return {
            "type": "error",
            "message": read_string_u32(sock),
        }

    if response_type == JSON_RESPONSE:
        return {
            "type": "json",
            "json": read_string_u32(sock),
        }

    raise ValueError(f"unknown response type: {response_type}")


@dataclass(frozen=True)
class CacheClient:
    host: str = "127.0.0.1"
    port: int = 7878
    timeout: float = 5
    retries: int = 3
    retry_backoff: float = 0.05

    def send_request(self, request: bytes) -> dict[str, Any]:
        last_error: OSError | EOFError | ValueError | None = None

        for attempt in range(self.retries + 1):
            try:
                with socket.create_connection(
                    (self.host, self.port), timeout=self.timeout
                ) as sock:
                    sock.settimeout(self.timeout)
                    sock.sendall(request)
                    sock.shutdown(socket.SHUT_WR)
                    return decode_response(sock)
            except (OSError, EOFError, ValueError) as error:
                last_error = error
                if attempt == self.retries:
                    break
                time.sleep(self.retry_backoff * (attempt + 1))

        assert last_error is not None
        raise last_error

    def insert(
        self,
        task_id: str,
        path: list[str],
        value: str | None,
        sandbox_id: str | None,
        runtime: int,
        ttl: int | None = None,
    ) -> dict[str, Any]:
        return self.send_request(encode_insert(task_id, path, value, sandbox_id, runtime, ttl))

    def find(self, task_id: str, path: list[str]) -> dict[str, Any]:
        return self.send_request(encode_find(task_id, path))

    def dump_all_response(self) -> dict[str, Any]:
        return self.send_request(encode_dump_all())

    def dump_all(self) -> object:
        response = self.dump_all_response()
        if response["type"] == "error":
            raise RuntimeError(response["message"])
        if response["type"] != "json":
            raise RuntimeError(f"expected json response, got {response['type']}")
        return json.loads(response["json"])


@dataclass
class PersistentCacheClient:
    host: str = "127.0.0.1"
    port: int = 7878
    timeout: float = 5
    retries: int = 3
    retry_backoff: float = 0.05
    sock: socket.socket | None = None

    def __enter__(self) -> "PersistentCacheClient":
        self.connect()
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        self.close()

    def connect(self) -> None:
        last_error: OSError | None = None

        for attempt in range(self.retries + 1):
            try:
                self.sock = socket.create_connection(
                    (self.host, self.port), timeout=self.timeout
                )
                self.sock.settimeout(self.timeout)
                return
            except OSError as error:
                last_error = error
                if attempt == self.retries:
                    break
                time.sleep(self.retry_backoff * (attempt + 1))

        assert last_error is not None
        raise last_error

    def close(self) -> None:
        if self.sock is not None:
            self.sock.close()
            self.sock = None

    def send_request(self, request: bytes) -> dict[str, Any]:
        if self.sock is None:
            raise RuntimeError("persistent cache client is not connected")

        self.sock.sendall(request)
        return decode_response(self.sock)

    def insert(
        self,
        task_id: str,
        path: list[str],
        value: str | None,
        sandbox_id: str | None,
        runtime: int,
        ttl: int | None = None,
    ) -> dict[str, Any]:
        return self.send_request(encode_insert(task_id, path, value, sandbox_id, runtime, ttl))

    def find(self, task_id: str, path: list[str]) -> dict[str, Any]:
        return self.send_request(encode_find(task_id, path))

    def dump_all_response(self) -> dict[str, Any]:
        return self.send_request(encode_dump_all())


async def async_read_u8(reader: asyncio.StreamReader) -> int:
    return struct.unpack(">B", await reader.readexactly(1))[0]


async def async_read_bool(reader: asyncio.StreamReader) -> bool:
    value = await async_read_u8(reader)
    if value == 0:
        return False
    if value == 1:
        return True
    raise ValueError(f"invalid bool byte: {value}")


async def async_read_u16(reader: asyncio.StreamReader) -> int:
    return struct.unpack(">H", await reader.readexactly(2))[0]


async def async_read_u32(reader: asyncio.StreamReader) -> int:
    return struct.unpack(">I", await reader.readexactly(4))[0]


async def async_read_string_u16(reader: asyncio.StreamReader) -> str:
    return (await reader.readexactly(await async_read_u16(reader))).decode("utf-8")


async def async_read_string_u32(reader: asyncio.StreamReader) -> str:
    return (await reader.readexactly(await async_read_u32(reader))).decode("utf-8")


async def async_read_path(reader: asyncio.StreamReader) -> list[str]:
    count = await async_read_u16(reader)
    return [await async_read_string_u16(reader) for _ in range(count)]


async def async_read_optional_string_u16(
    reader: asyncio.StreamReader,
) -> str | None:
    if await async_read_bool(reader):
        return await async_read_string_u16(reader)
    return None


async def async_read_optional_string_u32(
    reader: asyncio.StreamReader,
) -> str | None:
    if await async_read_bool(reader):
        return await async_read_string_u32(reader)
    return None


async def async_read_optional_path(
    reader: asyncio.StreamReader,
) -> list[str] | None:
    if await async_read_bool(reader):
        return await async_read_path(reader)
    return None


async def async_decode_response(reader: asyncio.StreamReader) -> dict[str, Any]:
    response_type = await async_read_u8(reader)

    if response_type == INSERT_RESULT_RESPONSE:
        return {
            "type": "insert_result",
            "value_inserted": await async_read_bool(reader),
            "sandbox_inserted": await async_read_bool(reader),
        }

    if response_type == FIND_RESULT_RESPONSE:
        return {
            "type": "find_result",
            "value": await async_read_optional_string_u32(reader),
            "sandbox_path": await async_read_optional_path(reader),
            "sandbox_id": await async_read_optional_string_u16(reader),
        }

    if response_type == ERROR_RESPONSE:
        return {
            "type": "error",
            "message": await async_read_string_u32(reader),
        }

    if response_type == JSON_RESPONSE:
        return {
            "type": "json",
            "json": await async_read_string_u32(reader),
        }

    raise ValueError(f"unknown response type: {response_type}")


@dataclass
class AsyncPersistentCacheClient:
    host: str = "127.0.0.1"
    port: int = 7878
    timeout: float = 5
    retries: int = 3
    retry_backoff: float = 0.05
    reader: asyncio.StreamReader | None = None
    writer: asyncio.StreamWriter | None = None

    async def __aenter__(self) -> "AsyncPersistentCacheClient":
        await self.connect()
        return self

    async def __aexit__(self, exc_type: object, exc: object, traceback: object) -> None:
        await self.close()

    async def connect(self) -> None:
        last_error: OSError | TimeoutError | None = None

        for attempt in range(self.retries + 1):
            try:
                self.reader, self.writer = await asyncio.wait_for(
                    asyncio.open_connection(self.host, self.port),
                    timeout=self.timeout,
                )
                return
            except (OSError, TimeoutError) as error:
                last_error = error
                if attempt == self.retries:
                    break
                await asyncio.sleep(self.retry_backoff * (attempt + 1))

        assert last_error is not None
        raise last_error

    async def close(self) -> None:
        if self.writer is not None:
            self.writer.close()
            await self.writer.wait_closed()
            self.reader = None
            self.writer = None

    async def send_request(self, request: bytes) -> dict[str, Any]:
        if self.reader is None or self.writer is None:
            raise RuntimeError("async persistent cache client is not connected")

        self.writer.write(request)
        await asyncio.wait_for(self.writer.drain(), timeout=self.timeout)
        return await asyncio.wait_for(
            async_decode_response(self.reader),
            timeout=self.timeout,
        )

    async def insert(
        self,
        task_id: str,
        path: list[str],
        value: str | None,
        sandbox_id: str | None,
        runtime: int,
        ttl: int | None = None,
    ) -> dict[str, Any]:
        return await self.send_request(
            encode_insert(task_id, path, value, sandbox_id, runtime, ttl)
        )

    async def find(self, task_id: str, path: list[str]) -> dict[str, Any]:
        return await self.send_request(encode_find(task_id, path))

    async def dump_all_response(self) -> dict[str, Any]:
        return await self.send_request(encode_dump_all())
