import logging
from typing import List, Dict, Any, Optional
import chromadb
from src.embeddings.embedding_generator import EmbeddedChunk

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ChromaVectorDB:
    """ChromaDB implementation for vector storage (fallback when Milvus-lite is unavailable)"""
    
    def __init__(
        self, 
        db_path: str = "./chroma_data",
        collection_name: str = "notebook_lm",
        embedding_dim: int = 384
    ):
        self.db_path = db_path
        self.collection_name = collection_name
        self.embedding_dim = embedding_dim
        self.client = None
        self.collection = None
        self.chunks_store = {}  # Store chunks in memory for retrieval
        
        self._initialize_client()
        self._setup_collection()
    
    def _initialize_client(self):
        try:
            # Initialize ChromaDB client with new client API (persistent storage)
            self.client = chromadb.PersistentClient(path=self.db_path)
            logger.info(f"ChromaDB PersistentClient initialized with database: {self.db_path}")
        except Exception as e:
            logger.error(f"Failed to initialize ChromaDB client: {str(e)}")
            raise
    
    def _setup_collection(self):
        try:
            # Get or create collection
            self.collection = self.client.get_or_create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"}
            )
            logger.info(f"Collection '{self.collection_name}' ready")
        except Exception as e:
            logger.error(f"Failed to setup collection: {str(e)}")
            raise
    
    def insert(self, chunks: List[EmbeddedChunk]) -> bool:
        """Insert embedded chunks into the vector database"""
        try:
            if not chunks:
                logger.warning("No chunks to insert")
                return False
            
            ids = []
            embeddings = []
            documents = []
            metadatas = []
            
            for chunk in chunks:
                chunk_id = f"{chunk.source_file}_{chunk.chunk_index}"
                ids.append(chunk_id)
                embeddings.append(chunk.embedding)
                documents.append(chunk.content)
                
                # Store chunk metadata
                metadata = {
                    "source_file": chunk.source_file,
                    "source_type": chunk.source_type,
                    "chunk_index": str(chunk.chunk_index),
                }
                if chunk.page_number is not None:
                    metadata["page_number"] = str(chunk.page_number)
                metadatas.append(metadata)
                
                # Store full chunk data for later retrieval
                self.chunks_store[chunk_id] = {
                    "content": chunk.content,
                    "source_file": chunk.source_file,
                    "source_type": chunk.source_type,
                    "page_number": chunk.page_number,
                    "chunk_index": chunk.chunk_index
                }
            
            # Add to collection
            self.collection.add(
                ids=ids,
                embeddings=embeddings,
                documents=documents,
                metadatas=metadatas
            )
            
            logger.info(f"Successfully inserted {len(chunks)} chunks")
            return True
            
        except Exception as e:
            logger.error(f"Error inserting chunks: {str(e)}")
            return False
    
    def search(self, query_embedding: List[float], top_k: int = 5) -> List[Dict[str, Any]]:
        """Search for similar chunks using vector similarity"""
        try:
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=top_k
            )
            
            if not results or not results.get('ids') or len(results['ids']) == 0:
                return []
            
            chunk_ids = results['ids'][0]
            distances = results.get('distances', [[]] * len(chunk_ids))[0]
            
            search_results = []
            for chunk_id, distance in zip(chunk_ids, distances):
                chunk_data = self.chunks_store.get(chunk_id)
                if chunk_data:
                    similarity = 1 - (distance / 2)  # Convert distance to similarity
                    search_results.append({
                        "chunk_id": chunk_id,
                        "content": chunk_data["content"],
                        "source_file": chunk_data["source_file"],
                        "source_type": chunk_data["source_type"],
                        "page_number": chunk_data["page_number"],
                        "chunk_index": chunk_data["chunk_index"],
                        "similarity": similarity
                    })
            
            return search_results
            
        except Exception as e:
            logger.error(f"Error searching: {str(e)}")
            return []
    
    def get_chunk_by_id(self, chunk_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve a specific chunk by ID"""
        try:
            chunk_data = self.chunks_store.get(chunk_id)
            if chunk_data:
                return chunk_data
            
            # Try to fetch from collection
            results = self.collection.get(ids=[chunk_id])
            if results and results['ids']:
                return {
                    "chunk_id": chunk_id,
                    "content": results['documents'][0] if results['documents'] else None,
                    "source_file": results['metadatas'][0].get('source_file') if results['metadatas'] else None,
                    "source_type": results['metadatas'][0].get('source_type') if results['metadatas'] else None,
                    "page_number": int(results['metadatas'][0].get('page_number', -1)) if results['metadatas'] and results['metadatas'][0].get('page_number') else None,
                    "chunk_index": int(results['metadatas'][0].get('chunk_index', -1)) if results['metadatas'] and results['metadatas'][0].get('chunk_index') else None
                }
            
            logger.warning(f"Chunk {chunk_id} not found")
            return None
            
        except Exception as e:
            logger.error(f"Error retrieving chunk: {str(e)}")
            return None
    
    def delete_collection(self):
        """Delete the entire collection"""
        try:
            self.client.delete_collection(name=self.collection_name)
            self.chunks_store.clear()
            logger.info(f"Collection '{self.collection_name}' deleted")
        except Exception as e:
            logger.warning(f"Error deleting collection: {str(e)}")
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """Get statistics about the collection"""
        try:
            count = self.collection.count()
            return {
                "collection_name": self.collection_name,
                "total_chunks": count,
                "embedding_dim": self.embedding_dim
            }
        except Exception as e:
            logger.error(f"Error getting collection stats: {str(e)}")
            return {}
