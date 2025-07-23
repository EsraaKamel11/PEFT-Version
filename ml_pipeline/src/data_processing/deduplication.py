import logging
from typing import List, Dict, Any, Optional
import numpy as np
from tqdm import tqdm
import pandas as pd
from sentence_transformers import SentenceTransformer
import hashlib

class Deduplicator:
    def __init__(self, similarity_threshold: float = 0.95, method: str = "levenshtein"):
        """
        Initialize deduplicator with configurable threshold and method
        
        Args:
            similarity_threshold: Threshold for considering documents as duplicates (0.0-1.0)
            method: Deduplication method ('levenshtein', 'semantic', 'hybrid')
        """
        self.logger = logging.getLogger(self.__class__.__name__)
        self.similarity_threshold = similarity_threshold
        self.method = method
        
        # Initialize semantic model if needed
        self.semantic_model = None
        if method in ["semantic", "hybrid"]:
            try:
                self.semantic_model = SentenceTransformer('all-MiniLM-L6-v2')
                self.logger.info("Loaded semantic model for deduplication")
            except Exception as e:
                self.logger.warning(f"Failed to load semantic model: {e}")
                self.method = "levenshtein" if method == "semantic" else "levenshtein"

    def deduplicate(self, documents: List[Dict[str, Any]], text_column: str = "text") -> List[Dict[str, Any]]:
        """
        Remove near-duplicate documents using content similarity
        
        Args:
            documents: List of document dictionaries
            text_column: Column name containing the text content
            
        Returns:
            List of deduplicated documents
        """
        if not documents:
            self.logger.info("No documents to deduplicate")
            return []
        
        self.logger.info(f"Starting deduplication of {len(documents)} documents using {self.method} method")
        
        # Pre-filter exact duplicates using hash
        documents = self._remove_exact_duplicates(documents, text_column)
        
        if self.method == "levenshtein":
            return self._levenshtein_deduplication(documents, text_column)
        elif self.method == "semantic":
            return self._semantic_deduplication(documents, text_column)
        elif self.method == "hybrid":
            return self._hybrid_deduplication(documents, text_column)
        else:
            raise ValueError(f"Unknown deduplication method: {self.method}")

    def _remove_exact_duplicates(self, documents: List[Dict[str, Any]], text_column: str) -> List[Dict[str, Any]]:
        """Remove exact duplicates using content hash"""
        seen_hashes = set()
        unique_docs = []
        
        for doc in documents:
            content = doc.get(text_column, "")
            content_hash = hashlib.md5(content.encode()).hexdigest()
            
            if content_hash not in seen_hashes:
                seen_hashes.add(content_hash)
                unique_docs.append(doc)
        
        removed = len(documents) - len(unique_docs)
        if removed > 0:
            self.logger.info(f"Removed {removed} exact duplicates")
        
        return unique_docs

    def _levenshtein_deduplication(self, documents: List[Dict[str, Any]], text_column: str) -> List[Dict[str, Any]]:
        """Deduplicate using Levenshtein distance"""
        try:
            from Levenshtein import ratio
        except ImportError:
            self.logger.error("Levenshtein package not installed. Install with: pip install python-Levenshtein")
            return documents
        
        # Sort by length to compare long vs short (keep longer documents)
        sorted_docs = sorted(documents, key=lambda x: len(x.get(text_column, "")), reverse=True)
        
        unique_docs = [sorted_docs[0]]
        removed_count = 0
        
        for doc in tqdm(sorted_docs[1:], desc="Deduplicating with Levenshtein"):
            is_duplicate = False
            doc_content = doc.get(text_column, "")
            
            for u_doc in unique_docs:
                u_content = u_doc.get(text_column, "")
                
                # Skip if lengths are too different
                if abs(len(doc_content) - len(u_content)) / max(len(doc_content), len(u_content)) > 0.3:
                    continue
                
                # Calculate similarity
                similarity = ratio(doc_content, u_content)
                if similarity > self.similarity_threshold:
                    is_duplicate = True
                    removed_count += 1
                    break
            
            if not is_duplicate:
                unique_docs.append(doc)
        
        self.logger.info(f"Levenshtein deduplication: removed {removed_count} duplicates")
        return unique_docs

    def _semantic_deduplication(self, documents: List[Dict[str, Any]], text_column: str) -> List[Dict[str, Any]]:
        """Deduplicate using semantic similarity"""
        if self.semantic_model is None:
            self.logger.warning("Semantic model not available, falling back to Levenshtein")
            return self._levenshtein_deduplication(documents, text_column)
        
        # Extract texts
        texts = [doc.get(text_column, "") for doc in documents]
        
        # Compute embeddings
        self.logger.info("Computing semantic embeddings...")
        embeddings = self.semantic_model.encode(texts, show_progress_bar=True)
        
        # Find duplicates using cosine similarity
        unique_docs = [documents[0]]
        removed_count = 0
        
        for i, doc in enumerate(tqdm(documents[1:], desc="Deduplicating with semantic similarity")):
            is_duplicate = False
            doc_embedding = embeddings[i]
            
            for u_doc in unique_docs:
                u_idx = documents.index(u_doc)
                u_embedding = embeddings[u_idx]
                
                # Calculate cosine similarity
                similarity = self._cosine_similarity(doc_embedding, u_embedding)
                if similarity > self.similarity_threshold:
                    is_duplicate = True
                    removed_count += 1
                    break
            
            if not is_duplicate:
                unique_docs.append(doc)
        
        self.logger.info(f"Semantic deduplication: removed {removed_count} duplicates")
        return unique_docs

    def _hybrid_deduplication(self, documents: List[Dict[str, Any]], text_column: str) -> List[Dict[str, Any]]:
        """Deduplicate using both Levenshtein and semantic similarity"""
        # First pass: semantic deduplication
        self.logger.info("Hybrid deduplication: First pass (semantic)")
        semantic_docs = self._semantic_deduplication(documents, text_column)
        
        # Second pass: Levenshtein deduplication
        self.logger.info("Hybrid deduplication: Second pass (Levenshtein)")
        final_docs = self._levenshtein_deduplication(semantic_docs, text_column)
        
        return final_docs

    def _cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Calculate cosine similarity between two vectors"""
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return dot_product / (norm1 * norm2)

    def deduplicate_dataframe(self, df: pd.DataFrame, text_column: str = "text") -> pd.DataFrame:
        """Deduplicate pandas DataFrame"""
        documents = df.to_dict('records')
        deduplicated_docs = self.deduplicate(documents, text_column)
        return pd.DataFrame(deduplicated_docs)

    def get_deduplication_stats(self, original_count: int, final_count: int) -> Dict[str, Any]:
        """Get deduplication statistics"""
        removed_count = original_count - final_count
        reduction_percentage = (removed_count / original_count * 100) if original_count > 0 else 0
        
        return {
            "original_count": original_count,
            "final_count": final_count,
            "removed_count": removed_count,
            "reduction_percentage": reduction_percentage,
            "method": self.method,
            "threshold": self.similarity_threshold
        }

    def batch_deduplicate(self, documents: List[Dict[str, Any]], text_column: str = "text", 
                         batch_size: int = 1000) -> List[Dict[str, Any]]:
        """Deduplicate large datasets in batches"""
        if len(documents) <= batch_size:
            return self.deduplicate(documents, text_column)
        
        self.logger.info(f"Batch deduplication: processing {len(documents)} documents in batches of {batch_size}")
        
        # Process in batches
        all_unique_docs = []
        for i in range(0, len(documents), batch_size):
            batch = documents[i:i + batch_size]
            self.logger.info(f"Processing batch {i//batch_size + 1}/{(len(documents) + batch_size - 1)//batch_size}")
            
            batch_unique = self.deduplicate(batch, text_column)
            all_unique_docs.extend(batch_unique)
        
        # Final deduplication across all batches
        self.logger.info("Final cross-batch deduplication")
        final_unique = self.deduplicate(all_unique_docs, text_column)
        
        return final_unique 