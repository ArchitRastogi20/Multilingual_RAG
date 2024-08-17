# RAG on Multilingual Documents

This project implements a **Retrieval-Augmented Generation (RAG)** system for processing Italian documents(for now). The RAG architecture combines document retrieval with a language model to generate informative responses based on the content of the documents.

## Project Overview

The system is designed to retrieve relevant information from a corpus of Italian documents using different retrieval techniques and then generate responses using a Language Model (LLM). This implementation uses:

### Retrieval Methods:
- **Word2Vec**: A vector-based retrieval technique that uses word embeddings to find semantically similar documents.
- **BM25**: A probabilistic-based information retrieval model that ranks documents based on their relevance to the query.

### Language Model:
- **ChatGPT**: A large language model used for generating natural language responses based on the retrieved documents.
