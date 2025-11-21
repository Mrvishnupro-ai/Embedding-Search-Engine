# Document Retrieval System

A modular document retrieval system built with Python, FastAPI, and Streamlit. It uses Google Gemini for embeddings and FAISS for efficient similarity search.

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

## Project Structure
- `api.py`: FastAPI backend application.
- `ui.py`: Streamlit frontend application.
- `embedder.py`: Handles text embedding using Google Gemini.
- `search_engine.py`: Manages FAISS index and search logic.
- `cache_manager.py`: Handles caching of embeddings.
- `cleaning.py`: Text preprocessing pipeline.
- `data-set.py`: Script to generate/download sample data.
