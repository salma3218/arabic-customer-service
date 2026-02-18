"""
Knowledge Base Manager - Creates and manages FAQ vector database
"""

import json
import chromadb
from chromadb.utils import embedding_functions
from pathlib import Path
from typing import List, Dict, Any, Optional
from backend.config.settings import RETRIEVAL_CONFIG, KNOWLEDGE_BASE_PATH
from backend.utils.logger import get_logger

logger = get_logger(__name__)


class KnowledgeBaseManager:
    """
    Manages the ChromaDB vector database for FAQ storage and retrieval
    """
    
    def __init__(
        self,
        persist_directory: Optional[str] = None,
        collection_name: Optional[str] = None,
        embedding_model: Optional[str] = None
    ):
        """
        Initialize ChromaDB and load knowledge base
        
        Args:
            persist_directory: Path to store ChromaDB data (uses RETRIEVAL_CONFIG if None)
            collection_name: Name of the collection (uses RETRIEVAL_CONFIG if None)
            embedding_model: Embedding model to use (uses RETRIEVAL_CONFIG if None)
        """
        
        # Use config defaults if not provided
        self.persist_directory = persist_directory or RETRIEVAL_CONFIG["persist_directory"]
        self.collection_name = collection_name or RETRIEVAL_CONFIG["collection_name"]
        self.embedding_model_name = embedding_model or RETRIEVAL_CONFIG["embedding_model"]
        
        logger.info(f"🔧 Initializing KnowledgeBaseManager")
        logger.info(f"   Embedding model: {self.embedding_model_name}")
        logger.info(f"   Collection: {self.collection_name}")
        
        # Create directory if doesn't exist
        Path(self.persist_directory).mkdir(parents=True, exist_ok=True)
        
        # Initialize ChromaDB client
        self.client = chromadb.PersistentClient(path=self.persist_directory)
        
        # ✅ Setup embedding function with new model
        logger.info(f"📥 Loading embedding model (this may take a moment)...")
        self.embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=self.embedding_model_name
        )
        logger.info(f"✅ Embedding model loaded successfully")
        
        # Get or create collection
        try:
            self.collection = self.client.get_collection(
                name=self.collection_name,
                embedding_function=self.embedding_function
            )
            logger.info(f"✅ Loaded existing collection: {self.collection_name}")
        except:
            self.collection = self.client.create_collection(
                name=self.collection_name,
                embedding_function=self.embedding_function,
                metadata={"description": "Arabic FAQ knowledge base"}
            )
            logger.info(f"✨ Created new collection: {self.collection_name}")
        
        logger.info(f"📊 Collection contains {self.collection.count()} documents")
    
    def load_knowledge_base(self, json_file: Optional[str] = None):
        """
        Load FAQ entries from JSON file into ChromaDB
        
        Args:
            json_file: Path to JSON file with FAQ entries (uses KNOWLEDGE_BASE_PATH if None)
        """
        
        # Use default path if not provided
        json_file = json_file or str(KNOWLEDGE_BASE_PATH)
        
        logger.info(f"📂 Loading knowledge base from {json_file}...")
        
        # Check if file exists
        if not Path(json_file).exists():
            logger.error(f"❌ Knowledge base file not found: {json_file}")
            raise FileNotFoundError(f"Knowledge base file not found: {json_file}")
        
        with open(json_file, 'r', encoding='utf-8') as f:
            faqs = json.load(f)
        
        logger.info(f"📖 Found {len(faqs)} FAQ entries")
        
        # Prepare data for ChromaDB
        documents = []
        metadatas = []
        ids = []
        
        for faq in faqs:
            # Combine question and answer for better semantic search
            document = f"{faq['question']}\n\n{faq['answer']}"
            documents.append(document)
            
            # Store metadata
            metadata = {
                "category": faq.get("category", "general"),
                "subcategory": faq.get("subcategory", ""),
                "question": faq["question"],
                "priority": faq.get("priority", "medium"),
                "keywords": ",".join(faq.get("keywords", [])),
                "last_updated": faq.get("last_updated", "")
            }
            metadatas.append(metadata)
            
            # Use FAQ ID
            ids.append(faq["id"])
        
        # Add to ChromaDB (or update if already exists)
        try:
            logger.info(f"💾 Adding documents to ChromaDB...")
            self.collection.upsert(
                documents=documents,
                metadatas=metadatas,
                ids=ids
            )
            logger.info(f"✅ Successfully loaded {len(documents)} documents into ChromaDB")
        except Exception as e:
            logger.error(f"❌ Failed to load documents: {e}")
            raise
    
    def search(
        self,
        query: str,
        n_results: int = 5,
        category_filter: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Search for relevant documents
        
        Args:
            query: Search query in Arabic
            n_results: Number of results to return (default: 5)
            category_filter: Filter by category (optional)
            
        Returns:
            Dictionary with documents, distances, and metadata
        """
        
        # Build filter
        where_filter = None
        if category_filter:
            where_filter = {"category": category_filter}
        
        try:
            results = self.collection.query(
                query_texts=[query],
                n_results=n_results,
                where=where_filter
            )
            
            logger.debug(f"🔍 Search returned {len(results['documents'][0])} results")
            
            return results
            
        except Exception as e:
            logger.error(f"❌ Search failed: {e}")
            return {"documents": [[]], "distances": [[]], "metadatas": [[]]}
    
    def get_stats(self) -> Dict[str, Any]:
        """Get knowledge base statistics"""
        
        count = self.collection.count()
        
        # Get sample to analyze categories
        if count > 0:
            sample = self.collection.peek(limit=min(count, 100))
            categories = {}
            
            if sample['metadatas']:
                for metadata in sample['metadatas']:
                    category = metadata.get('category', 'unknown')
                    categories[category] = categories.get(category, 0) + 1
        else:
            categories = {}
        
        return {
            "total_documents": count,
            "collection_name": self.collection_name,
            "embedding_model": self.embedding_model_name,
            "categories": categories,
            "persist_directory": self.persist_directory
        }
    
    def reset_collection(self):
        """Delete and recreate the collection (useful for testing)"""
        
        logger.warning(f"⚠️  Resetting collection: {self.collection_name}")
        
        try:
            self.client.delete_collection(name=self.collection_name)
            logger.info(f"🗑️  Deleted collection: {self.collection_name}")
        except:
            pass
        
        self.collection = self.client.create_collection(
            name=self.collection_name,
            embedding_function=self.embedding_function,
            metadata={"description": "Arabic FAQ knowledge base"}
        )
        logger.info(f"✨ Created new collection: {self.collection_name}")


def initialize_knowledge_base(reset: bool = False):
    """
    Initialize knowledge base from JSON file
    Run this once to populate ChromaDB
    
    Args:
        reset: If True, delete existing collection and start fresh
    """
    
    logger.info("🚀 Initializing knowledge base...")
    
    # Create manager
    kb = KnowledgeBaseManager()
    
    # Reset if requested
    if reset:
        kb.reset_collection()
    
    # Check if already populated
    if kb.collection.count() > 0 and not reset:
        logger.info(f"ℹ️  Collection already contains {kb.collection.count()} documents")
        logger.info("💡 Use reset=True to reload from JSON")
        
        # Show stats
        stats = kb.get_stats()
        logger.info(f"📊 Knowledge Base Stats:")
        logger.info(f"   Total Documents: {stats['total_documents']}")
        logger.info(f"   Embedding Model: {stats['embedding_model']}")
        logger.info(f"   Categories: {stats['categories']}")
        
        return kb
    
    # Load from JSON file
    try:
        kb.load_knowledge_base()
    except FileNotFoundError as e:
        logger.error(f"❌ {e}")
        logger.info("💡 Please create knowledge_base.json using the prompt provided")
        logger.info("   Location: backend/data/knowledge_base.json")
        return None
    
    # Show stats
    stats = kb.get_stats()
    logger.info(f"\n📊 Knowledge Base Stats:")
    logger.info(f"   Total Documents: {stats['total_documents']}")
    logger.info(f"   Embedding Model: {stats['embedding_model']}")
    logger.info(f"   Categories: {stats['categories']}")
    
    return kb


if __name__ == "__main__":
    import sys
    
    # Check for reset flag
    reset = "--reset" in sys.argv
    
    # Run initialization
    kb = initialize_knowledge_base(reset=reset)
    
    # Test search if successful
    if kb:
        logger.info("\n🧪 Testing search...")
        
        test_queries = [
            "ما هي أسعار الباقات؟",
            "كيف أعيد تعيين كلمة المرور؟",
            "ما هي ساعات العمل؟"
        ]
        
        for query in test_queries:
            from backend.utils.text_formatter import format_arabic_for_terminal
            
            query_display = format_arabic_for_terminal(query, 50)
            logger.info(f"\n🔍 Query: {query_display}")
            
            results = kb.search(query, n_results=3)
            
            for i, doc in enumerate(results['documents'][0][:3]):
                distance = results['distances'][0][i]
                similarity = 1 - distance
                metadata = results['metadatas'][0][i]
                
                question = format_arabic_for_terminal(metadata.get('question', ''), 50)
                
                logger.info(f"   [{i+1}] Similarity: {similarity:.2f} | Category: {metadata.get('category')}")
                logger.info(f"       {question}")
        
        logger.info("\n✅ Knowledge base initialization complete!")
        logger.info(f"💡 To reset and reload: python -m data.knowledge_base_manager --reset")