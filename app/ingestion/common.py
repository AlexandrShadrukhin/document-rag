from __future__ import annotations

import hashlib
import uuid
from pathlib import Path

CHUNK_NAMESPACE = uuid.UUID("81376f7a-17d7-4d53-9321-cb57c273ad6d")


def file_sha256(path: Path, block_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(block_size):
            digest.update(block)
    return digest.hexdigest()


def normalized_source(path: Path) -> str:
    return str(path.expanduser().resolve())


def document_id_for(file_hash: str) -> str:
    return file_hash


def deterministic_chunk_id(
    file_hash: str, page_number: int | None, chunk_index: int
) -> str:
    key = f"{file_hash}:{page_number or 0}:{chunk_index}"
    return str(uuid.uuid5(CHUNK_NAMESPACE, key))
