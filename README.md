# Model Adaptation Examples

[![CI Testing](https://github.com/leonvanbokhorst/model-adaptation/actions/workflows/ci.yml/badge.svg)](https://github.com/leonvanbokhorst/model-adaptation/actions/workflows/ci.yml)

This repository contains example code demonstrating various model adaptation techniques covered in the Model Adaptation lectures at Fontys University of Applied Sciences (FICT), minor [AI for Society](https://www.fontys.nl/en/Study-at-Fontys/Exchange-programmes/Artificial-Intelligence-For-Society.htm) by [Leon van Bokhorst](https://github.com/leonvanbokhorst).

## Overview

This repository demonstrates advanced model adaptation techniques, like SIFT (Selective Instance Fine-Tuning) for LLM optimization. The codebase includes real-time fine-tuning, semantic similarity search, and comprehensive evaluation metrics.

The codebase explores several key areas of model adaptation:

1. **Prompting Techniques** (`01_prompting.py`)
   - Basic prompting
   - Structured prompts
   - Chain-of-thought
   - Few-shot learning
   - Role playing
   - Task decomposition
   - Zero-shot learning
   - Self-consistency
   - Constrained generation
   - Socratic method
   - Reflective prompting
   - Guided feedback
   - Persona-based prompting
   - Template-based prompting
   - Comparative analysis
   - Iterative refinement
   - Scenario-based prompting
   - Self-verification
   - Logical verification

2. **Retrieval-Augmented Generation (RAG)** (`02_rag.py`)
   - Document storage and embedding generation
   - Semantic similarity search with cosine similarity
   - Query routing based on intent classification
   - LLM-based response generation
   - Multiple specialized demo modes:
     - Combined product information
     - Style and fashion advice
     - Technical specifications
     - Store availability
   - Support for both Ollama and OpenAI backends
   - Comprehensive product knowledge base
   - Dynamic content aggregation
   - Inventory management integration
   - Real-time availability tracking
   - Customer review analysis
   - Style recommendation engine

3. **Semantic Space Visualization** (`03_semantic_space.py`)
   - 2D and 3D visualization of semantic embeddings
   - Multiple dimensionality reduction techniques:
     - t-SNE
     - PCA
     - UMAP
     - MDS
   - Interactive 3D visualizations using Plotly
   - Semantic similarity heatmaps
   - Hierarchical clustering dendrograms
   - Word analogy visualization
   - Temporal embedding analysis
   - Cross-model embedding comparison

4. **Model Fine-tuning** (`04_fine-tuning.py`)
   - LoRA-based fine-tuning of LLMs
   - Optimized training parameters
   - Efficient memory management
   - Key features:
     - Custom tokenizer configuration
     - Dataset preparation and formatting
     - Gradient checkpointing
     - Configurable LoRA parameters
     - Inference optimization
     - Training monitoring and logging
   - Support for:
     - Hugging Face models
     - Custom datasets
     - Instruction fine-tuning
     - Performance optimization
     - Quantization techniques
     - Model pruning

5. **Synthetic Data Generation** (`05_synthetic_data.py`)
   - Instruction-based data generation
   - Quality assessment with complexity scoring
   - Dataset versioning and management
   - Balanced dataset creation
   - NLTK-based linguistic analysis
   - Data augmentation techniques
   - Quality validation pipelines
   - Automated labeling
   - Domain-specific generation
   - Cross-validation support

6. **Multi-Agent Research Simulation** (`06_multi_agent.py`)
   - Dynamic team composition based on research questions
   - Adaptive agents with personality models
   - Semantic routing of discussions
   - Real-time discussion quality monitoring
   - Advanced analytics and visualization
   - Key features:
     - Personality-driven agent behavior
     - Belief system evolution
     - Concept tracking and evolution
     - Discussion quality metrics
     - Knowledge base management
     - External system integration
     - Consensus building algorithms
     - Debate simulation
     - Research methodology adaptation
     - Group dynamics modeling
     - Emergent behavior analysis
     - Social network formation
     - Cognitive load simulation
     - Narrative field mapping

7. **Sentiment Analysis** (`07_sentiment_analysis.py`)
   - LLM-based sentiment analysis using Ollama
   - Asynchronous API endpoints with FastAPI
   - Key features:
     - Sentiment classification (positive/negative/neutral)
     - Detailed sentiment explanations
     - Robust error handling and logging
     - Resource management with context managers
     - Response parsing and validation
     - Cached client connections
     - RESTful API endpoints
   - Support for:
     - Custom text analysis
     - Scalable API deployment
     - Performance optimization
     - Structured response formats

8. **Research Question Generation** (`08_research_questions.py`)
   - LLM-based research question generation using Ollama
   - FastAPI-based REST endpoints
   - Key features:
     - Main question generation
     - Sub-question derivation
     - Academic rigor validation
     - Structured response format
   - Support for:
     - Asynchronous processing
     - Resource management
     - Error handling
     - Response validation
     - Client caching
     - RESTful API design

9. **Semantic Data Analysis** (`09_semantic_data_analysis.py`)

- Intelligent data field analysis and categorization
- Pattern recognition using embeddings
- Data categories:
  - Numeric, Text, Metadata
  - Mixed types
  - JSON structures
- Field patterns:
  - Identifiers
  - Personal information
  - Temporal data
  - Financial data
  - Categorical data
  - Measurements
  - Location data
  - Contact information
  - System metadata
  - User preferences
- Visualization features:
  - Color-coded categories
  - Quality indicators
  - Complexity markers
  - Pattern grouping
  - Detailed statistics

10. **Complex Data Analysis** (`10_semantic_complex_data_analysis.py`)
    - Advanced analysis for enterprise data structures
    - Enhanced pattern recognition
    - Support for:
      - Nested JSON structures
      - Mixed data types
      - Multi-value fields
      - Inconsistent formats
    - Analysis features:
      - Hierarchical data handling
      - Format validation
      - Anomaly detection
      - Quality scoring
    - Enterprise patterns:
      - Department hierarchies
      - Compensation structures
      - Performance metrics
      - System metadata
      - Contact details
      - Temporal sequences

11. **LLM Benchmarking** (`11_llm_benchmark.py`)
    - Comprehensive model comparison framework
    - Multiple evaluation metrics:
      - ROUGE scores (1, 2, L)
      - BLEU score
      - Perplexity
      - BM25 similarity
    - Complaint-specific metrics:
      - Negativity scoring
      - Emotional intensity
      - Structure analysis
      - Pattern density
    - Hardware optimization:
      - Apple Silicon (MPS) support
      - CUDA support
      - CPU fallback
    - Detailed analysis reporting:
      - Comparative metrics
      - Improvement percentages
      - Statistical significance
      - Visual progress tracking
    - Support for:
      - Custom datasets
      - Multiple model comparisons
      - Batch processing
      - Metric visualization

12. **GGUF Model Conversion** (`12_llm_gguf_conversion.py`)
    - Automated GGUF format conversion pipeline
    - Multiple quantization levels:
      - 2-bit to 8-bit options
      - FP16 and FP32 support
      - Size/quality trade-off variants
    - Hardware-specific optimizations:
      - Metal support for Apple Silicon
      - Multi-threaded processing
    - Hugging Face integration:
      - Automatic model download
      - GGUF model upload
      - Repository management
    - Key features:
      - Automated llama.cpp setup
      - Progress monitoring
      - Error handling
      - Resource management
    - Support for:
      - Custom quantization methods
      - Model verification
      - Batch processing
      - Version control

13. **Meeting Summary** (`14_meeting_summary.py`)
    - Real-time meeting recording and transcription
    - Core components:
      - Audio recording with sounddevice
      - Whisper transcription model
      - Ollama LLM summarization
      - Multi-threaded processing
    - Key features:
      - Live audio capture
      - Streaming transcription
      - Structured summaries
      - Progress monitoring
    - Audio handling:
      - Configurable sample rate
      - Chunked processing
      - Temporary file management
      - Auto cleanup
    - Transcription:
      - Large-v3 Whisper model
      - CUDA/CPU support
      - Dutch language support
      - Real-time feedback
    - Summary generation:
      - Context extraction
      - Key point identification
      - Action item detection
      - Structured formatting
    - Output management:
      - Timestamped files
      - Subject-based naming
      - Progress tracking
      - Error handling
    - Technical features:
      - Thread synchronization
      - Resource management
      - Graceful shutdown
      - Comprehensive logging

13. **Semantic Router** (`15_semantic_router.py`)
    - Semantic-based query routing system
    - Core components:
      - SentenceTransformer embeddings
      - Cosine similarity matching
      - Configurable similarity threshold
      - Route registration system
    - Key features:
      - Dynamic route registration
      - Semantic similarity scoring
      - Detailed route matching logs
      - Fallback handling
    - Route management:
      - Description-based routing
      - Handler function mapping
      - Embedding caching
      - Similarity threshold tuning
    - Support for:
      - Custom embedding models
      - Multiple route handlers
      - Query preprocessing
      - Similarity visualization
    - Example handlers:
      - Greeting responses
      - Weather information
      - Time queries
    - Diagnostic features:
      - Route similarity scoring
      - Match confidence tracking
      - Route selection logging
      - Error handling

14. **MVNX Data Processing** (`16_mvnx_to_csv.py`)
    - MVNX motion capture data conversion
    - Core functionality:
      - XML parsing with ElementTree
      - Sensor data extraction
      - DataFrame generation
      - CSV export
    - Data handling:
      - Frame-by-frame processing
      - Multi-sensor support
      - Time series alignment
      - Missing data handling
    - Motion data types:
      - Orientation
      - Position
      - Velocity
      - Acceleration
      - Angular velocity
    - Features:
      - Automatic sensor detection
      - Frame rate extraction
      - Subject metadata parsing
      - Data validation
    - Output format:
      - Time-indexed DataFrame
      - Consistent column naming
      - Sensor-specific channels
      - Motion parameters
    - Error handling:
      - XML validation
      - Data type conversion
      - Missing frame handling
      - Exception tracking

**PyTorch Experiments** (`src/poc/`)
    - Neural Network Architecture Studies
      - ResNet Implementation (`resnet.py`, `resnet_02.py`, `resnet_03.py`)
        - Traditional vs ResNet comparison
        - Gradient flow visualization
        - Skip connection analysis
        - Layer activation tracking
        - Performance benchmarking

    - MNIST Classification (`pytorch_poc.py`)
      - Custom neural network implementation
      - Apple Silicon (MPS) optimization
      - Training visualization
      - Batch processing
      - Progress tracking
      - Real-time accuracy monitoring

    - Loss Function Analysis (`pytorch_visualize_loss.py`)
      - Cross-entropy loss visualization
      - Prediction confidence analysis
      - Scenario-based comparisons
      - Error analysis tools

    - Hardware Optimization
      - Apple Silicon (M-series) support
      - CUDA compatibility
      - CPU fallback handling
      - Memory management
      - Batch size optimization

    - Training Features
      - Real-time visualization
      - Progress tracking
      - Metric logging
      - Model checkpointing
      - Early stopping
      - Learning rate scheduling

**19. SIFT Fine-Tuning** (`19_llm_fine_tuning_sift.py`)
    - Selective Instance Fine-Tuning (SIFT) Implementation
      - Real-time model adaptation
      - Semantic similarity search
      - Uncertainty tracking
      - Dynamic example selection
      - Progress visualization

    - Core Components
      - TextDataLoader: Streaming dataset management
      - SubsetSampler: Training data sampling
      - MetricsComputer: Performance tracking
      - SIFTVisualizer: Training visualization
      - SIFTTrainer: Fine-tuning orchestration

    - Training Features
      - Adaptive learning rates
      - Gradient accumulation
      - Loss thresholding
      - Uncertainty monitoring
      - Example selection
      - Progress tracking
      - Clear console output

    - Model Configuration
      - Base model: unsloth/Llama-3.2-1B
      - Embedding model: BAAI/bge-large-en-v1.5
      - FAISS indexing
      - Streaming dataset: openwebtext
      - Configurable batch sizes
      - Memory optimization

    - Monitoring & Metrics
      - Loss tracking
      - Uncertainty estimation
      - Example statistics
      - Global/local best tracking
      - Real-time visualization
      - Training summaries

## Development Standards

- PEP 8 and black formatting
- Type hints (PEP 484)
- Comprehensive testing (pytest)
- CI/CD pipeline integration
- Documentation with Google-style docstrings

## Installation

```bash
pip install -r requirements.txt
```

## Testing

Tests are written using pytest and can be run with:

```bash
pytest
```

For integration tests only:

```bash
pytest -m integration
```

## License

This project is licensed under the Apache License 2.0 - see the LICENSE file for details.
