from typing import List, Dict, Any, Optional
from .repository import Repository
from .utils import *


class FileRepository(Repository):
    """File-based repository implementation using local JSON/JSONL files"""
    
    def __init__(self):
        ensure_dirs()
        self._cached_ids = None
        self._cached_qa = None
        self._cached_tau = None
    
    def get_tau(self) -> float:
        if self._cached_tau is None:
            self._cached_tau = float(read_text(TAU_PATH, "0.30"))
        return self._cached_tau
    
    def set_tau(self, tau: float) -> None:
        write_text(TAU_PATH, str(tau))
        self._cached_tau = tau
    
    def get_image_ids(self) -> List[str]:
        if self._cached_ids is None:
            self._cached_ids = read_json(IDS_PATH, [])
        return self._cached_ids[:]  # Return copy to prevent modification
    
    def add_image_id(self, img_id: str) -> None:
        ids = self.get_image_ids()
        ids.append(img_id)
        write_json(IDS_PATH, ids)
        self._cached_ids = ids
    
    def get_all_qa(self) -> List[Dict[str, Any]]:
        if self._cached_qa is None:
            self._cached_qa = list(read_jsonl(QA_PATH))
        return self._cached_qa[:]  # Return copy to prevent modification
    
    def get_qa_by_id(self, img_id: str) -> Optional[Dict[str, Any]]:
        qa_records = self.get_all_qa()
        for record in qa_records:
            if record.get("id") == img_id:
                return record.copy()
        return None
    
    def upsert_qa(self, img_id: str, answer: str, quality: int, path: str = None, tags: List[str] = None) -> None:
        if path is None:
            path = f"data/images/{img_id}"
        if tags is None:
            tags = []
            
        # Read all records
        rows = list(read_jsonl(QA_PATH))
        found = False
        
        # Update existing record
        for r in rows:
            if r.get("id") == img_id:
                r["answer"] = answer
                r["quality"] = quality
                if path:
                    r["path"] = path
                if tags:
                    r["tags"] = tags
                found = True
                break
        
        # Rewrite entire file
        QA_PATH.unlink(missing_ok=True)
        for r in rows:
            append_jsonl(QA_PATH, r)
            
        # Append new record if not found
        if not found:
            append_jsonl(QA_PATH, {
                "id": img_id,
                "type": "image", 
                "path": path,
                "answer": answer,
                "quality": quality,
                "tags": tags
            })
        
        # Invalidate cache
        self._cached_qa = None
    
    def refresh(self) -> None:
        """Clear all caches to force reload from files"""
        self._cached_ids = None
        self._cached_qa = None
        self._cached_tau = None