from __future__ import annotations

import json
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def new_doc_id() -> str:
    return uuid.uuid4().hex


def get_storage_root() -> Path:
    return Path(os.getenv("STORAGE_DIR", "storage")).resolve()


def pdf_path(doc_id: str) -> Path:
    root = get_storage_root()
    ensure_dir(root / "pdfs")
    return root / "pdfs" / f"{doc_id}.pdf"


def _record_path(doc_id: str) -> Path:
    root = get_storage_root()
    ensure_dir(root / "pdfs")
    return root / "pdfs" / f"{doc_id}.json"


def save_document_record(doc_id: str, filename: str) -> None:
    record = {
        "doc_id": doc_id,
        "filename": filename,
        "uploaded_at": datetime.now(timezone.utc).isoformat(),
        "indexed": False,
    }
    _record_path(doc_id).write_text(json.dumps(record))


def mark_doc_indexed(doc_id: str, index_time_s: float | None = None) -> None:
    path = _record_path(doc_id)
    if not path.exists():
        return
    record = json.loads(path.read_text())
    record["indexed"] = True
    if index_time_s is not None:
        record["index_time_s"] = round(index_time_s, 1)
    path.write_text(json.dumps(record))


def delete_document(doc_id: str) -> None:
    pdf = pdf_path(doc_id)
    if pdf.exists():
        pdf.unlink()
    record = _record_path(doc_id)
    if record.exists():
        record.unlink()


def list_docs() -> list[dict]:
    root = get_storage_root() / "pdfs"
    if not root.exists():
        return []
    docs = []
    for record_file in sorted(root.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True):
        try:
            docs.append(json.loads(record_file.read_text()))
        except Exception:
            pass
    return docs
