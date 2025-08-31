from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional


class Repository(ABC):
    """Abstract repository for CommonsAI data operations"""
    
    @abstractmethod
    def get_tau(self) -> float:
        """Get similarity threshold value"""
        pass
    
    @abstractmethod
    def set_tau(self, tau: float) -> None:
        """Set similarity threshold value"""
        pass
    
    @abstractmethod
    def get_image_ids(self) -> List[str]:
        """Get all image IDs"""
        pass
    
    @abstractmethod
    def add_image_id(self, img_id: str) -> None:
        """Add new image ID"""
        pass
    
    @abstractmethod
    def get_all_qa(self) -> List[Dict[str, Any]]:
        """Get all Q&A records"""
        pass
    
    @abstractmethod
    def get_qa_by_id(self, img_id: str) -> Optional[Dict[str, Any]]:
        """Get Q&A record by image ID"""
        pass
    
    @abstractmethod
    def upsert_qa(self, img_id: str, answer: str, quality: int, path: str = None, tags: List[str] = None) -> None:
        """Update existing or insert new Q&A record"""
        pass
    
    @abstractmethod
    def refresh(self) -> None:
        """Refresh cached data from storage"""
        pass