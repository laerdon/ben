from typing import List, Optional
from datetime import datetime
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from .models import LogEntry, MemoryEntry, SearchResult
import uuid
import math
import os
import shutil
from pathlib import Path


class MemoryManager:
    def __init__(self, collection_name: str = "notion_logs"):
        # Create data directory in user's home folder
        self.data_dir = os.path.expanduser("~/notion_assistant_data")
        os.makedirs(self.data_dir, exist_ok=True)

        # Initialize Chroma client with new configuration
        self.client = chromadb.PersistentClient(path=self.data_dir)

        # Get or create collection
        self.collection = self.client.get_or_create_collection(
            name=collection_name, metadata={"hnsw:space": "cosine"}
        )

        # Initialize sentence transformer
        self.model = SentenceTransformer("all-MiniLM-L6-v2")

        # Recency bias parameters
        self.lambda_decay = 0.1  # Decay rate for recency bias
        self.recency_weight = 0.2  # Weight for recency in final score

    def clear_collection(self):
        """Clear all entries from the collection."""
        try:
            # Store collection name before deleting objects
            collection_name = (
                self.collection.name if hasattr(self, "collection") else "notion_logs"
            )

            # Close any existing connections
            if hasattr(self, "client"):
                del self.client
            if hasattr(self, "collection"):
                del self.collection

            # Delete the directory to ensure complete cleanup
            if os.path.exists(self.data_dir):
                # First try to remove all files
                for root, dirs, files in os.walk(self.data_dir, topdown=False):
                    for name in files:
                        try:
                            os.chmod(os.path.join(root, name), 0o666)
                            os.remove(os.path.join(root, name))
                        except Exception:
                            pass
                    for name in dirs:
                        try:
                            os.chmod(os.path.join(root, name), 0o777)
                            os.rmdir(os.path.join(root, name))
                        except Exception:
                            pass
                try:
                    os.rmdir(self.data_dir)
                except Exception:
                    pass

            # Recreate the directory with proper permissions
            os.makedirs(self.data_dir, mode=0o777, exist_ok=True)

            # Reinitialize the client and collection
            self.client = chromadb.PersistentClient(path=self.data_dir)
            self.collection = self.client.create_collection(
                name=collection_name, metadata={"hnsw:space": "cosine"}
            )
            print("Collection cleared successfully")
        except Exception as e:
            print(f"Error clearing collection: {e}")
            # If there's an error, try to reinitialize anyway
            self.client = chromadb.PersistentClient(path=self.data_dir)
            self.collection = self.client.get_or_create_collection(
                name=collection_name, metadata={"hnsw:space": "cosine"}
            )

    def _generate_embedding(self, text: str) -> List[float]:
        """Generate embedding for text using sentence transformer."""
        return self.model.encode(text).tolist()

    def _calculate_recency_score(self, entry_date: datetime) -> float:
        """Calculate recency score using exponential decay."""
        days_old = (datetime.now() - entry_date).days
        return math.exp(-self.lambda_decay * days_old)

    def store_entry(self, entry: LogEntry) -> str:
        """Store a log entry in Chroma with its embedding."""
        # Generate embedding from raw text
        embedding = self._generate_embedding(entry.raw_text or "")

        # Generate unique ID
        entry_id = str(uuid.uuid4())

        # Prepare metadata
        metadata = {"date": entry.date.isoformat()}

        # Store in Chroma
        self.collection.add(
            embeddings=[embedding],
            documents=[entry.raw_text or ""],
            metadatas=[metadata],
            ids=[entry_id],
        )

        return entry_id

    def search(self, query: str, top_k: int = 5) -> List[SearchResult]:
        """Search for entries using query and apply recency bias."""
        # Generate query embedding
        query_embedding = self._generate_embedding(query)

        # Search in Chroma
        results = self.collection.query(
            query_embeddings=[query_embedding], n_results=top_k
        )

        # Process results with recency bias
        search_results = []
        for i in range(len(results["ids"][0])):
            entry_id = results["ids"][0][i]
            similarity_score = results["distances"][0][i]
            metadata = results["metadatas"][0][i]

            # Calculate recency score
            entry_date = datetime.fromisoformat(metadata["date"])
            recency_score = self._calculate_recency_score(entry_date)

            # Calculate final score (normalize similarity score to 0-1 range)
            normalized_similarity = 1 - (
                similarity_score / 2
            )  # Convert L2 distance to similarity
            final_score = normalized_similarity + self.recency_weight * recency_score

            # Create search result
            search_results.append(
                SearchResult(
                    entry=LogEntry(
                        date=entry_date,
                        blocks=[],  # We don't store blocks in Chroma
                        raw_text=results["documents"][0][i],
                    ),
                    similarity_score=normalized_similarity,
                    final_score=final_score,
                )
            )

        # Sort by final score
        search_results.sort(key=lambda x: x.final_score, reverse=True)
        return search_results
