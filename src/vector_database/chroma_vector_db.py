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
    
    def create_index(self, use_binary_quantization: bool = False, **kwargs):
        """Create index - no-op for ChromaDB as it handles indexing automatically.
        
        This method exists for API compatibility with MilvusVectorDB.
        """
        logger.info("ChromaDB handles indexing automatically - no explicit index creation needed")
        return True

    def insert_embeddings(self, chunks: List[EmbeddedChunk]) -> List[str]:
        """Insert embedded chunks (alias for insert, for API compatibility with MilvusVectorDB)"""
        return self.insert(chunks)

    def insert(self, chunks: List[EmbeddedChunk]) -> List[str]:
        """Insert embedded chunks into the vector database"""
        try:
            if not chunks:
                logger.warning("No chunks to insert")
                return []
            
            ids = []
            embeddings = []
            documents = []
            metadatas = []
            
            for embedded_chunk in chunks:
                chunk = embedded_chunk.chunk
                chunk_id = f"{chunk.source_file}_{chunk.chunk_index}"
                ids.append(chunk_id)
                
                # Ensure embedding is a list
                emb_list = embedded_chunk.embedding.tolist() if hasattr(embedded_chunk.embedding, 'tolist') else embedded_chunk.embedding
                embeddings.append(emb_list)
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
            logger.info(f"Adding {len(ids)} items to ChromaDB collection...")
            coll = self.collection
            if coll is not None:
                coll.add(
                    ids=ids,
                    embeddings=embeddings,
                    documents=documents,
                    metadatas=metadatas
                )
                count = coll.count()
                logger.info(f"Successfully inserted {len(chunks)} chunks. Collection count: {count}")
            else:
                logger.error("Cannot insert: Collection not initialized")
            return ids
            
        except Exception as e:
            logger.error(f"Error inserting chunks: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            return []
    
    def search(self, query_embedding=None, top_k: int = 5, query_vector=None, 
               limit: Optional[int] = None, filter_expr: Optional[str] = None, **kwargs) -> List[Dict[str, Any]]:
        """Search for similar chunks using vector similarity.
        
        Accepts both ChromaVectorDB-style params (query_embedding, top_k) and 
        MilvusVectorDB-style params (query_vector, limit) for API compatibility.
        """
        try:
            # Support both param styles
            embedding = query_vector if query_vector is not None else query_embedding
            if embedding is None:
                logger.error("No query embedding provided")
                return []
            n_results = limit if limit is not None else top_k
            
            # Build ChromaDB where filter from filter_expr if provided
            where_filter = None
            if filter_expr:
                # Parse simple filter expressions like: field == "value"
                try:
                    import re
                    match = re.match(r'(\w+)\s*==\s*"([^"]*)"', filter_expr)
                    if match:
                        field, value = match.groups()
                        where_filter = {field: {"$eq": value}}
                except Exception as e:
                    logger.warning(f"Could not parse filter expression '{filter_expr}': {e}")
            
            query_kwargs: Dict[str, Any] = {
                "query_embeddings": [embedding],
                "n_results": n_results,
                "include": ["documents", "metadatas", "distances"]
            }
            if where_filter:
                query_kwargs["where"] = where_filter
            coll = self.collection
            if coll is None:
                logger.error("Search failed: Collection not initialized")
                return []
                
            results = coll.query(**query_kwargs)
            
            if not results or not results.get('ids') or not results['ids'] or len(results['ids'][0]) == 0:
                count = coll.count()
                logger.warning(f"No results found for query. Total items in collection: {count}")
                return []
            
            logger.info(f"Query returned {len(results['ids'][0])} results. Top score: {results.get('distances', [[]])[0][0] if results.get('distances') else 'N/A'}")
            
            chunk_ids = results['ids'][0]
            distances = results.get('distances', [[]])[0]
            documents = results.get('documents', [[]])[0]
            metadatas = results.get('metadatas', [[]])[0]
            
            search_results = []
            for i in range(len(chunk_ids)):
                chunk_id = chunk_ids[i]
                distance = distances[i] if i < len(distances) else 0
                content = documents[i] if i < len(documents) else ""
                meta = metadatas[i] if i < len(metadatas) else {}
                
                similarity = 1 - (distance / 2)  # Convert distance to similarity
                
                # Metadata string values might need conversion to int
                p_val = meta.get("page_number")
                p_num = None
                if p_val is not None:
                    try:
                        p_num = int(p_val)  # type: ignore
                    except (ValueError, TypeError):
                        p_num = None
                    
                c_val = meta.get("chunk_index")
                c_idx = 0
                if c_val is not None:
                    try:
                        c_idx = int(c_val)  # type: ignore
                    except (ValueError, TypeError):
                        c_idx = 0

                # Return in MilvusVectorDB-compatible format for rag.py
                search_result = {
                    "id": chunk_id,
                    "chunk_id": chunk_id,
                    "score": similarity,
                    "similarity": similarity,
                    "content": content,
                    "source_file": meta.get("source_file"),
                    "source_type": meta.get("source_type"),
                    "page_number": p_num,
                    "chunk_index": c_idx,
                    "citation": {
                        "source_file": meta.get("source_file"),
                        "source_type": meta.get("source_type"),
                        "page_number": p_num,
                        "chunk_index": c_idx,
                        "start_char": None,
                        "end_char": None,
                    },
                    "metadata": meta,
                    "embedding_model": None
                }
                search_results.append(search_result)
                
                # Update chunks_store cache
                self.chunks_store[chunk_id] = {
                    "content": content,
                    "source_file": meta.get("source_file"),
                    "source_type": meta.get("source_type"),
                    "page_number": p_num,
                    "chunk_index": c_idx
                }
            
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
