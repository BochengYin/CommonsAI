"""
BM25 lexical search module for OCR text retrieval.
Provides keyword-based search to complement semantic similarity.
"""
import pickle
import logging
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from .utils import DATA

logger = logging.getLogger(__name__)

class BM25SearchEngine:
    """BM25-based lexical search for OCR text."""
    
    def __init__(self, index_path: Optional[Path] = None):
        self.index_path = index_path or (DATA / "bm25.pkl")
        self.bm25 = None
        self.image_indices = None  # Maps BM25 doc index to image index
        self.image_ids = None      # Image IDs corresponding to documents
        self._loaded = False
        
    def load_index(self) -> bool:
        """Load BM25 index from disk."""
        if not self.index_path.exists():
            logger.warning(f"BM25 index not found: {self.index_path}")
            return False
            
        try:
            with open(self.index_path, 'rb') as f:
                data = pickle.load(f)
                
            self.bm25 = data["bm25"]
            self.image_indices = data["image_indices"]
            self.image_ids = data["image_ids"]
            self._loaded = True
            
            logger.info(f"BM25 index loaded: {len(self.image_ids)} documents")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load BM25 index: {e}")
            return False
    
    def _ensure_loaded(self):
        """Ensure BM25 index is loaded."""
        if not self._loaded:
            if not self.load_index():
                raise RuntimeError("BM25 index not available")
    
    def search(self, query: str, k: int = 10) -> List[Tuple[str, float]]:
        """
        Search for images using BM25 text matching.
        
        Args:
            query: Search query text
            k: Number of results to return
            
        Returns:
            List of (image_id, score) tuples sorted by relevance
        """
        self._ensure_loaded()
        
        if not query.strip():
            return []
        
        # Tokenize query (simple whitespace splitting)
        query_tokens = query.lower().split()
        
        # Get BM25 scores
        try:
            scores = self.bm25.get_scores(query_tokens)
        except Exception as e:
            logger.error(f"BM25 search failed: {e}")
            return []
        
        # Create (image_id, score) pairs and sort by score
        results = []
        for i, score in enumerate(scores):
            if score > 0:  # Only include documents with positive scores
                image_id = self.image_ids[i]
                results.append((image_id, float(score)))
        
        # Sort by score descending and return top-k
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:k]
    
    def search_with_metadata(self, query: str, k: int = 10) -> List[Dict]:
        """
        Search with additional metadata about matches.
        
        Args:
            query: Search query text
            k: Number of results to return
            
        Returns:
            List of result dictionaries with metadata
        """
        self._ensure_loaded()
        
        basic_results = self.search(query, k)
        
        # Add metadata to results
        enriched_results = []
        query_tokens = set(query.lower().split())
        
        for image_id, score in basic_results:
            # Find the document index for this image
            doc_idx = None
            for i, img_id in enumerate(self.image_ids):
                if img_id == image_id:
                    doc_idx = i
                    break
            
            if doc_idx is not None:
                # Get document tokens for highlighting
                doc_tokens = self.bm25.corpus[doc_idx]
                matched_tokens = [token for token in doc_tokens if token in query_tokens]
                
                enriched_results.append({
                    "image_id": image_id,
                    "bm25_score": score,
                    "matched_tokens": matched_tokens,
                    "match_count": len(matched_tokens),
                    "doc_length": len(doc_tokens)
                })
        
        return enriched_results
    
    def get_document_text(self, image_id: str) -> Optional[str]:
        """Get the original text document for an image ID."""
        self._ensure_loaded()
        
        # Find document index for this image ID
        doc_idx = None
        for i, img_id in enumerate(self.image_ids):
            if img_id == image_id:
                doc_idx = i
                break
        
        if doc_idx is not None:
            # Reconstruct text from tokens
            return " ".join(self.bm25.corpus[doc_idx])
        
        return None
    
    def get_stats(self) -> Dict:
        """Get BM25 index statistics."""
        if not self._loaded:
            return {"loaded": False}
        
        return {
            "loaded": True,
            "num_documents": len(self.image_ids),
            "avg_doc_length": sum(len(doc) for doc in self.bm25.corpus) / len(self.bm25.corpus),
            "vocabulary_size": len(set(token for doc in self.bm25.corpus for token in doc)),
            "index_file": str(self.index_path)
        }

# Global BM25 search engine instance
_bm25_engine = None

def get_bm25_engine() -> BM25SearchEngine:
    """Get or create global BM25 search engine."""
    global _bm25_engine
    if _bm25_engine is None:
        _bm25_engine = BM25SearchEngine()
        _bm25_engine.load_index()  # Try to load on first access
    return _bm25_engine

def search_bm25(query: str, k: int = 10) -> List[Tuple[str, float]]:
    """Simple interface for BM25 search."""
    engine = get_bm25_engine()
    return engine.search(query, k)

def refresh_bm25_index():
    """Refresh the global BM25 engine by reloading from disk."""
    global _bm25_engine
    if _bm25_engine is not None:
        _bm25_engine.load_index()