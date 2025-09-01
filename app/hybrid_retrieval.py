"""
Hybrid retrieval system with RRF (Reciprocal Rank Fusion) combining multiple search channels:
- Image similarity (OpenCLIP visual)
- Text-to-image similarity (OpenCLIP text→image)  
- OCR text similarity (OpenCLIP text→OCR text)
- BM25 lexical search (keyword matching)
"""
import faiss
import numpy as np
import logging
from typing import List, Dict, Tuple, Optional, Set
from dataclasses import dataclass
from .embeddings import encode_text
from .bm25_search import get_bm25_engine
from .utils import DATA, read_json

logger = logging.getLogger(__name__)

@dataclass
class SearchResult:
    """Container for search results from a single channel."""
    image_id: str
    score: float
    rank: int
    channel: str
    metadata: Dict = None

@dataclass
class HybridSearchResult:
    """Container for fused hybrid search results."""
    image_id: str
    rrf_score: float
    individual_scores: Dict[str, float]  # channel_name -> score
    individual_ranks: Dict[str, int]     # channel_name -> rank  
    channels_matched: List[str]
    final_rank: int

class HybridRetriever:
    """Multi-channel hybrid retrieval with RRF fusion."""
    
    def __init__(self, rrf_k: int = 60):
        """
        Initialize hybrid retriever.
        
        Args:
            rrf_k: RRF constant (typical: 60, higher = more conservative fusion)
        """
        self.rrf_k = rrf_k
        self.image_index = None
        self.ocr_embeds = None
        self.image_ids = None
        self.bm25_engine = get_bm25_engine()
        self._loaded = False
        
    def load_indices(self) -> bool:
        """Load all required indices and embeddings."""
        try:
            # Load image FAISS index
            index_path = DATA / "img.index"
            if index_path.exists():
                self.image_index = faiss.read_index(str(index_path))
            else:
                logger.warning("Image index not found")
                return False
            
            # Load OCR embeddings  
            ocr_emb_path = DATA / "ocr_embeds.npy"
            if ocr_emb_path.exists():
                self.ocr_embeds = np.load(ocr_emb_path)
            else:
                logger.warning("OCR embeddings not found")
                return False
                
            # Load image IDs
            ids_path = DATA / "ids.json"
            if ids_path.exists():
                self.image_ids = read_json(ids_path)
            else:
                logger.warning("Image IDs not found")
                return False
            
            # Verify dimensions match
            if len(self.image_ids) != self.image_index.ntotal:
                logger.error("Image IDs and index size mismatch")
                return False
                
            if len(self.image_ids) != len(self.ocr_embeds):
                logger.error("Image IDs and OCR embeddings size mismatch") 
                return False
            
            self._loaded = True
            logger.info(f"Hybrid retriever loaded: {len(self.image_ids)} images")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load indices: {e}")
            return False
    
    def _ensure_loaded(self):
        """Ensure all indices are loaded."""
        if not self._loaded:
            if not self.load_indices():
                raise RuntimeError("Hybrid retrieval indices not available")
    
    def _search_image_similarity(self, query: str, k: int) -> List[SearchResult]:
        """Search using text→image similarity (OpenCLIP)."""
        self._ensure_loaded()
        
        try:
            query_vec = encode_text(query).astype(np.float32)  # (1, d)
            distances, indices = self.image_index.search(query_vec, k)
            
            results = []
            for rank, (idx, score) in enumerate(zip(indices[0], distances[0])):
                if idx >= 0:  # Valid index
                    image_id = self.image_ids[idx]
                    results.append(SearchResult(
                        image_id=image_id,
                        score=float(score),
                        rank=rank + 1,
                        channel="image_similarity"
                    ))
            return results
            
        except Exception as e:
            logger.error(f"Image similarity search failed: {e}")
            return []
    
    def _search_ocr_similarity(self, query: str, k: int) -> List[SearchResult]:
        """Search using text→OCR text similarity (OpenCLIP)."""
        self._ensure_loaded()
        
        try:
            query_vec = encode_text(query)  # (1, d)
            
            # Compute cosine similarity with OCR embeddings
            query_norm = query_vec / np.linalg.norm(query_vec, axis=1, keepdims=True)
            ocr_norms = self.ocr_embeds / np.linalg.norm(self.ocr_embeds, axis=1, keepdims=True)
            similarities = np.dot(query_norm, ocr_norms.T)[0]  # (num_images,)
            
            # Get top-k indices and scores
            top_indices = np.argsort(-similarities)[:k]
            
            results = []
            for rank, idx in enumerate(top_indices):
                score = similarities[idx]
                if score > 0:  # Only include positive similarities
                    image_id = self.image_ids[idx]
                    results.append(SearchResult(
                        image_id=image_id,
                        score=float(score),
                        rank=rank + 1,
                        channel="ocr_similarity"
                    ))
            return results
            
        except Exception as e:
            logger.error(f"OCR similarity search failed: {e}")
            return []
    
    def _search_bm25(self, query: str, k: int) -> List[SearchResult]:
        """Search using BM25 lexical matching."""
        try:
            bm25_results = self.bm25_engine.search(query, k)
            
            results = []
            for rank, (image_id, score) in enumerate(bm25_results):
                results.append(SearchResult(
                    image_id=image_id,
                    score=score,
                    rank=rank + 1,
                    channel="bm25_lexical"
                ))
            return results
            
        except Exception as e:
            logger.error(f"BM25 search failed: {e}")
            return []
    
    def _compute_rrf_scores(self, channel_results: Dict[str, List[SearchResult]]) -> List[HybridSearchResult]:
        """Compute RRF fusion scores across all channels."""
        # Collect all unique image IDs
        all_image_ids: Set[str] = set()
        for results in channel_results.values():
            all_image_ids.update(result.image_id for result in results)
        
        # Compute RRF score for each image
        hybrid_results = []
        
        for image_id in all_image_ids:
            rrf_score = 0.0
            individual_scores = {}
            individual_ranks = {}
            channels_matched = []
            
            # Sum RRF contributions from each channel
            for channel_name, results in channel_results.items():
                # Find this image in the channel results
                found = False
                for result in results:
                    if result.image_id == image_id:
                        # RRF formula: 1 / (k + rank)
                        rrf_contribution = 1.0 / (self.rrf_k + result.rank)
                        rrf_score += rrf_contribution
                        
                        individual_scores[channel_name] = result.score
                        individual_ranks[channel_name] = result.rank
                        channels_matched.append(channel_name)
                        found = True
                        break
                
                # If not found in this channel, no contribution to RRF
                if not found:
                    individual_scores[channel_name] = 0.0
                    individual_ranks[channel_name] = None
            
            hybrid_results.append(HybridSearchResult(
                image_id=image_id,
                rrf_score=rrf_score,
                individual_scores=individual_scores,
                individual_ranks=individual_ranks,
                channels_matched=channels_matched,
                final_rank=0  # Will be set after sorting
            ))
        
        # Sort by RRF score descending
        hybrid_results.sort(key=lambda x: x.rrf_score, reverse=True)
        
        # Set final ranks
        for rank, result in enumerate(hybrid_results):
            result.final_rank = rank + 1
        
        return hybrid_results
    
    def search(self, query: str, k: int = 20, channels: Optional[List[str]] = None) -> List[HybridSearchResult]:
        """
        Perform hybrid search across all channels with RRF fusion.
        
        Args:
            query: Search query text
            k: Number of results per channel (final results may be fewer after fusion)
            channels: List of channels to use (default: all available)
            
        Returns:
            List of HybridSearchResult sorted by RRF score
        """
        self._ensure_loaded()
        
        # Default to all channels if not specified
        if channels is None:
            channels = ["image_similarity", "ocr_similarity", "bm25_lexical"]
        
        # Search each channel
        channel_results = {}
        
        if "image_similarity" in channels:
            channel_results["image_similarity"] = self._search_image_similarity(query, k)
            
        if "ocr_similarity" in channels:
            channel_results["ocr_similarity"] = self._search_ocr_similarity(query, k)
            
        if "bm25_lexical" in channels:
            channel_results["bm25_lexical"] = self._search_bm25(query, k)
        
        # Filter out empty results
        channel_results = {name: results for name, results in channel_results.items() if results}
        
        if not channel_results:
            logger.warning("No search results from any channel")
            return []
        
        # Compute RRF fusion
        hybrid_results = self._compute_rrf_scores(channel_results)
        
        logger.info(f"Hybrid search: {len(hybrid_results)} unique results from {len(channel_results)} channels")
        return hybrid_results
    
    def search_with_debug(self, query: str, k: int = 20) -> Dict:
        """
        Perform search with detailed debugging information.
        
        Returns:
            Dict containing results and debug information
        """
        self._ensure_loaded()
        
        # Search each channel individually
        image_results = self._search_image_similarity(query, k)
        ocr_results = self._search_ocr_similarity(query, k)
        bm25_results = self._search_bm25(query, k)
        
        channel_results = {
            "image_similarity": image_results,
            "ocr_similarity": ocr_results,
            "bm25_lexical": bm25_results
        }
        
        # Compute hybrid results
        hybrid_results = self._compute_rrf_scores(channel_results)
        
        return {
            "query": query,
            "rrf_k": self.rrf_k,
            "channel_results": {
                name: [{"image_id": r.image_id, "score": r.score, "rank": r.rank} 
                       for r in results]
                for name, results in channel_results.items()
            },
            "hybrid_results": [
                {
                    "image_id": r.image_id,
                    "rrf_score": r.rrf_score,
                    "final_rank": r.final_rank,
                    "channels_matched": r.channels_matched,
                    "individual_scores": r.individual_scores,
                    "individual_ranks": r.individual_ranks
                }
                for r in hybrid_results[:k]
            ],
            "stats": {
                "total_unique_results": len(hybrid_results),
                "channels_active": len([r for r in channel_results.values() if r]),
                "avg_channels_per_result": np.mean([len(r.channels_matched) for r in hybrid_results]) if hybrid_results else 0
            }
        }
    
    def get_status(self) -> Dict:
        """Get retrieval system status."""
        return {
            "loaded": self._loaded,
            "rrf_k": self.rrf_k,
            "image_index_loaded": self.image_index is not None,
            "ocr_embeds_loaded": self.ocr_embeds is not None,
            "image_ids_count": len(self.image_ids) if self.image_ids else 0,
            "bm25_stats": self.bm25_engine.get_stats()
        }

# Global hybrid retriever instance
_hybrid_retriever = None

def get_hybrid_retriever(rrf_k: int = 60) -> HybridRetriever:
    """Get or create global hybrid retriever."""
    global _hybrid_retriever
    if _hybrid_retriever is None:
        _hybrid_retriever = HybridRetriever(rrf_k=rrf_k)
        _hybrid_retriever.load_indices()
    return _hybrid_retriever

def hybrid_search(query: str, k: int = 20, channels: Optional[List[str]] = None) -> List[HybridSearchResult]:
    """Simple interface for hybrid search."""
    retriever = get_hybrid_retriever()
    return retriever.search(query, k, channels)

def refresh_hybrid_retriever():
    """Refresh the global hybrid retriever by reloading indices."""
    global _hybrid_retriever
    if _hybrid_retriever is not None:
        _hybrid_retriever.load_indices()