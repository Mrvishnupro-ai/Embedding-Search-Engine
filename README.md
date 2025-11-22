# Document Retrieval System

A modular document retrieval system built with Python, FastAPI, and Streamlit. It uses Google Gemini for embeddings and FAISS for efficient similarity search.

[![Video on Embedding-Search-Engine](https://github.com/user-attachments/assets/bdb44161-f031-46e1-86cf-b4df9d23ce57)](https://youtu.be/40n0XKy9rEU)

## Features
- **Semantic Search**: Finds documents based on meaning, not just keywords.
- **Query Expansion**: Automatically expands queries with synonyms using WordNet.
- **Caching**: Caches embeddings to avoid redundant processing.
- **Clean UI**: Simple and intuitive interface built with Streamlit.
- **REST API**: Backend API built with FastAPI.

## Setup

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd <repository-folder>
   ```

2. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Environment Variables**
   Create a `.env` file in the root directory and add your Google API key:
   ```
   GOOGLE_API_KEY=your_api_key_here
   ```

4. **Generate Data**
   Download the dataset:
   ```bash
   python data-set.py
   ```

## Usage

### 1. Start the Backend (API)
```bash
uvicorn api:app --reload
```
The API will start at `http://localhost:8000`. It will automatically process documents and build the search index on startup.

### 2. Start the Frontend (UI)
Open a new terminal and run:
```bash
streamlit run ui.py
```
The UI will open in your browser.

## Technical Architecture & File Descriptions

This project is structured as a modular microservices-like architecture, separating the frontend (UI), backend (API), and core logic (Search & Embedding). This design ensures scalability, maintainability, and clear separation of concerns.

<img width="1319" height="792" alt="image" src="https://github.com/user-attachments/assets/9d6aca23-1cd7-4842-8ea2-34ee2a42c5cd" />


### Core Components

#### 1. `api.py` (Backend Orchestrator)
**Purpose**: The central entry point of the backend, built with **FastAPI**. It orchestrates the initialization of the embedding model, search index, and caching system.
- **Key Functions**:
    - `startup_event()`: Asynchronously loads documents, triggers the embedding process, and builds the FAISS index upon server start.
    - `search(request)`: The main endpoint that accepts user queries, coordinates query expansion, embedding generation, and vector search.
- **Connection**: Acts as the bridge between the `ui.py` frontend and the logic modules (`embedder.py`, `search_engine.py`).

#### 2. `search_engine.py` (Vector Search Logic)
**Purpose**: Manages the vector database and search algorithms using **FAISS (Facebook AI Similarity Search)**.
- **Key Functions**:
    - `build_index()`: Converts document embeddings into a FAISS `IndexFlatIP` (Inner Product) index. We normalize vectors to ensure Inner Product equals **Cosine Similarity**.
    - `search_with_embedding()`: Performs the nearest neighbor search. It calculates multiple metrics (**Cosine Similarity**, **Euclidean Distance**, **Manhattan Distance**) to provide a comprehensive similarity assessment.
    - `expand_query()`: Uses **NLTK WordNet** to find synonyms (e.g., "car" -> "automobile"), improving search recall.
- **Reasoning**: FAISS is chosen for its high performance and scalability with dense vectors.

#### 3. `embedder.py` (Semantic Embedding)
**Purpose**: Handles the conversion of text into high-dimensional vectors using **Google's Gemini Embedding Model** (`gemini-embedding-001`).
- **Key Functions**:
    - `process_and_embed()`: Manages the batch processing of documents. It implements **multiprocessing** for parallel text cleaning to maximize performance.
    - `embed_query()`: Generates embeddings for incoming search queries.
- **Connection**: Integrates with `CacheManager` to prevent redundant API calls.

#### 4. `cleaning.py` (Data Preprocessing Pipeline)
**Purpose**: A robust pipeline to clean and normalize text before embedding, ensuring high-quality vector representations.
- **Key Functions**:
    - `clean_text_pipeline()`: Executes a sequence of operations:
        - **Encoding Fixes**: Handles UTF-8 issues.
        - **Noise Removal**: Strips HTML tags, email headers, URLs, and special characters.
        - **Normalization**: Lowercasing and whitespace trimming.
        - **NLP Tasks**: Tokenization, Stopword removal, and **Lemmatization** (reducing words to their base root).
- **Reasoning**: Clean data is crucial for accurate semantic search. Removing noise prevents irrelevant patterns from skewing the embeddings.

#### 5. `cache_manager.py` (Optimization Layer)
**Purpose**: Implements a persistent caching mechanism to store generated embeddings.
- **Key Functions**:
    - `get_hash()`: Generates a **SHA256 hash** of the document content.
    - `save_embedding()` / `get_embedding()`: Stores/Retrieves embeddings from a JSON file based on the content hash.
- **Reasoning**: Drastically reduces latency and API costs by ensuring we only compute embeddings for new or modified documents.

#### 6. `ui.py` (Frontend Interface)
**Purpose**: A user-friendly interface built with **Streamlit**.
- **Key Functions**:
    - Provides interactive controls for `top_k` results and similarity `threshold`.
    - Visualizes search results with metric dashboards and content previews.
    - Displays the "Expanded Query" to show users how their search was augmented.

#### 7. `data-set.py` (Data Ingestion)
**Purpose**: A utility script to fetch the **20 Newsgroups dataset** (specifically scientific/electronics topics) to populate the system with real-world test data.

---

## Technical Highlights & Design Decisions

- **Vector-Based Semantic Search**: Unlike keyword search (TF-IDF), this system understands *meaning*. For example, a search for "circuit" will find documents about "electronics" even if the exact word isn't present.
- **Hybrid Distance Metrics**: We provide Cosine Similarity (direction), Euclidean (magnitude), and Manhattan distances to give a multi-dimensional view of similarity.
- **Smart Caching**: The content-addressable storage (hashing) makes the system robust to file renames and efficient during restarts.
- **Query Expansion**: By enriching user queries with synonyms, we bridge the vocabulary gap between the user and the technical documents.

