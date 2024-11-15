"""
Graph-based Retrieval Augmented Generation (GraphRAG) Implementation.

This module implements a knowledge graph-based approach to RAG systems, combining
semantic search with graph traversal for improved context retrieval. It uses
spaCy for entity extraction and Ollama for LLM integration and embeddings.

Key Components:
    - Knowledge graph construction from documents
    - Entity extraction and relationship mapping
    - Semantic search with embedding similarity
    - Graph traversal for context expansion
    - LLM-based question answering

Example:
    rag = GraphRAG()
    rag.process_document("path/to/document.txt")
    result = rag.query("What methodologies were used?")
"""

from typing import List, Dict, Optional
import networkx as nx
from langchain_ollama import OllamaLLM
from langchain_ollama import OllamaEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
import spacy
import logging
from tqdm import tqdm

# At the top of the file, after imports
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

MODEL_NAME = "hermes3:latest"
EMBEDDINGS_MODEL = "nomic-embed-text"


class GraphRAG:
    def __init__(self, llm_model: str = MODEL_NAME):
        """Initialize GraphRAG with specified models.

        Args:
            llm_model: Name of the Ollama model to use for generation
        """
        logger.info(f"Initializing GraphRAG with LLM: {llm_model}")
        self.llm = OllamaLLM(model=llm_model)

        # Use nomic-embed-text through Ollama
        logger.info("Initializing Nomic embeddings model via Ollama")
        self.embeddings = OllamaEmbeddings(model=EMBEDDINGS_MODEL)

        self.graph = nx.Graph()
        self.nlp = spacy.load("en_core_web_sm")
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=50,
        )

    def process_document(self, file_path: str) -> None:
        """Process a document and construct the knowledge graph.

        Breaks down the document into chunks, extracts entities, generates
        embeddings, and builds a connected graph representation.

        Args:
            file_path: Path to the text document to process

        Raises:
            FileNotFoundError: If document file cannot be found
            SpacyError: If entity extraction fails
            EmbeddingError: If embedding generation fails
        """
        logger.info(f"Processing document: {file_path}")

        # Load document and split into manageable chunks using the text splitter
        loader = TextLoader(file_path)
        documents = loader.load()
        chunks = self.text_splitter.split_documents(documents)
        logger.info(f"Split document into {len(chunks)} chunks")

        # Temporarily reduce logging level to prevent progress bar interference
        original_level = logging.getLogger().level
        logging.getLogger().setLevel(logging.WARNING)

        try:
            # Iterate through each chunk to process and build the knowledge graph
            for chunk_id, chunk in enumerate(tqdm(chunks, desc="Processing chunks"), 1):
                # Extract named entities using spaCy
                doc = self.nlp(chunk.page_content)
                entities = [(ent.text, ent.label_) for ent in doc.ents]

                # Generate semantic embeddings for the chunk content
                chunk_embedding = self.embeddings.embed_query(chunk.page_content)

                # Create a node for the chunk with its full content and embedding
                chunk_node_id = f"chunk_{chunk_id}"
                self.graph.add_node(
                    chunk_node_id,
                    type="CHUNK",
                    content=chunk.page_content,
                    embedding=chunk_embedding,
                )

                # Process extracted entities and build graph connections
                for entity, label in entities:
                    # Add entity nodes with their type labels
                    self.graph.add_node(entity, type=label)
                    # Connect entities to their source chunks
                    self.graph.add_edge(entity, chunk_node_id, relation="APPEARS_IN")

                    # Create connections between co-occurring entities in the chunk
                    for other_entity, other_label in entities:
                        if entity != other_entity:
                            self.graph.add_edge(
                                entity, other_entity, chunk_ref=chunk_node_id
                            )
        finally:
            # Restore original logging level after processing
            logging.getLogger().setLevel(original_level)

        # Log final graph statistics
        logger.info(
            f"Final graph has {len(self.graph.nodes)} nodes and {len(self.graph.edges)} edges"
        )

    def query(
        self, question: str, max_hops: int = 2, similarity_threshold: float = 0.7
    ) -> Dict:
        """Query the knowledge graph for answers using semantic search.

        Performs semantic search to find relevant nodes, traverses the graph to
        expand context, and generates an answer using the LLM.

        Args:
            question: The user's question to answer
            max_hops: Maximum graph traversal distance from similar nodes
            similarity_threshold: Minimum cosine similarity score (0-1)

        Returns:
            Dict containing:
                - answer: Generated response from the LLM
                - influential_nodes: List of nodes that influenced the answer,
                  including similarity scores and context previews

        Raises:
            ValueError: If graph is empty or similarity threshold invalid
            EmbeddingError: If embedding generation fails
            LLMError: If answer generation fails
        """
        logger.info(f"Processing query: '{question}' with max_hops={max_hops}")

        # Generate embedding vector for the input question
        question_embedding = self.embeddings.embed_query(question)

        # Temporarily reduce logging level for cleaner progress bar output
        original_level = logging.getLogger().level
        logging.getLogger().setLevel(logging.WARNING)

        try:
            # Generate embeddings for all graph nodes with progress tracking
            logger.info("Generating embeddings for all nodes...")
            node_embeddings = {}
            for node in tqdm(self.graph.nodes, desc="Generating embeddings"):
                node_embeddings[node] = self.embeddings.embed_query(str(node))
        finally:
            # Restore original logging level after embedding generation
            logging.getLogger().setLevel(original_level)

        # Calculate cosine similarity between question and all nodes
        from numpy import dot
        from numpy.linalg import norm

        def cosine_similarity(a, b):
            """Calculate cosine similarity between two vectors."""
            return dot(a, b) / (norm(a) * norm(b))

        # Find nodes with similarity above threshold
        similar_nodes = []
        for node, embedding in node_embeddings.items():
            similarity = cosine_similarity(question_embedding, embedding)
            if similarity > similarity_threshold:
                similar_nodes.append((node, similarity))

        # Sort nodes by similarity score in descending order
        similar_nodes.sort(key=lambda x: x[1], reverse=True)
        logger.info(f"Found {len(similar_nodes)} semantically similar nodes")

        # Extract relevant context from connected chunk nodes
        context = set()  # Use set to avoid duplicate chunks
        context_sources = []  # Track source information for context

        # Explore neighborhood of top similar nodes
        for node, similarity in similar_nodes[:5]:
            # Get subgraph within max_hops distance
            subgraph = nx.ego_graph(self.graph, node, radius=max_hops)

            # Extract content from connected chunk nodes
            for n in subgraph.nodes():
                node_data = self.graph.nodes[n]
                if node_data.get("type") == "CHUNK":
                    chunk_content = node_data.get("content", "")
                    if chunk_content:
                        context.add(chunk_content)
                        context_sources.append(
                            {
                                "source_node": node,
                                "chunk_node": n,
                                "context": chunk_content,
                                "similarity": similarity,
                                "node_type": self.graph.nodes[node].get(
                                    "type", "Unknown"
                                ),
                            }
                        )

        # Handle case where no relevant context is found
        if not context:
            logger.warning("No relevant context found for the query")
            return {
                "answer": "I couldn't find relevant information to answer your question.",
                "influential_nodes": [],
            }

        logger.info(f"Found {len(context)} unique context pieces")

        # Construct prompt for LLM using gathered context
        prompt = f"""Given the following relevant excerpts from a document:

{' '.join(context)}

Please provide a focused answer to this question: {question}

Important: Base your answer only on the provided context. If you can't find relevant information, say so."""

        # Generate answer using LLM
        logger.debug(f"Generated prompt: {prompt[:200]}...")
        response = self.llm.invoke(prompt)
        logger.info("Generated response successfully")

        # Prepare response with answer and supporting information
        return {
            "answer": response,
            "influential_nodes": [
                {
                    "node": cs["source_node"],
                    "type": cs["node_type"],
                    "similarity": cs["similarity"],
                    "connected_to": cs["chunk_node"],
                    "context_preview": cs["context"][:100] + "...",
                }
                for cs in sorted(
                    context_sources, key=lambda x: x["similarity"], reverse=True
                )
            ][
                :5
            ],  # Return top 5 most influential nodes
        }


# Example usage
if __name__ == "__main__":
    # Initialize GraphRAG
    rag = GraphRAG()

    # Example: Process a document
    rag.process_document("src/data/Computer_are_social_actors.txt")
    question = "What methodologies were used?"

    # Example: Query the graph
    result = rag.query(question, max_hops=2, similarity_threshold=0.7)

    print("\nAnswer:")
    print(result["answer"])

    print("\nMost Influential Nodes:")
    for node in result["influential_nodes"]:
        print(f"\nNode: {node['node']}")
        print(f"Type: {node['type']}")
        print(f"Similarity: {node['similarity']:.3f}")
        print(f"Connected to: {node['connected_to']}")
        print(f"Context: {node['context_preview']}")
