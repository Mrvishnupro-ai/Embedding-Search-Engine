from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import os
import uvicorn
from embedder import Embedder
from search_engine import SearchEngine
from cache_manager import CacheManager
from cleaning import clean_text_pipeline

app = FastAPI(title="Document Retrieval API")

embedder = None
search_engine = None
cache_manager = None

class SearchRequest(BaseModel):
    query: str
    top_k: int = 5
    threshold: float = 0.0

class SearchResultItem(BaseModel):
    doc_id: str
    score: float
    metrics: dict
    preview: str
    explanation: dict

class SearchResponse(BaseModel):
    results: List[SearchResultItem]
    expanded_query: str

class CleanRequest(BaseModel):
    text: str

@app.on_event("startup")
async def startup_event():
    global embedder, search_engine, cache_manager
    
    cache_manager = CacheManager()
    embedder = Embedder(cache_manager=cache_manager)
    search_engine = SearchEngine()
    
    print("Startup: Loading and processing documents...")
    data_dir = os.path.join(os.getcwd(), 'data')
    
    documents = []
    filenames = []
    if os.path.exists(data_dir):
        for filename in os.listdir(data_dir):
            filepath = os.path.join(data_dir, filename)
            if os.path.isfile(filepath):
                try:
                    with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                        documents.append(f.read())
                        filenames.append(filename)
                except Exception as e:
                    print(f"Error reading {filename}: {e}")
    
    if documents:
        print(f"Processing {len(documents)} documents...")
        processed_data = embedder.process_and_embed(documents, filenames)
        print("Building index...")
        search_engine.build_index(processed_data)
        print("Index ready.")
    else:
        print("No documents found in data/ folder.")

@app.post("/search", response_model=SearchResponse)
async def search(request: SearchRequest):
    global embedder, search_engine
    
    if not search_engine or not search_engine.index:
        raise HTTPException(status_code=503, detail="Search engine not initialized or index empty")
    
    expanded_query = search_engine.expand_query(request.query)
    print(f"Original: {request.query} -> Expanded: {expanded_query}")
    
    query_embedding = embedder.embed_query(expanded_query)
    
    if query_embedding is None:
        raise HTTPException(status_code=500, detail="Failed to generate query embedding")
    
    results = search_engine.search_with_embedding(query_embedding, k=request.top_k, threshold=request.threshold)
    
    response_items = []
    for res in results:
        doc_text = res.get('full_text', '')
        explanation = search_engine.explain_result(request.query, doc_text)
        
        response_items.append(SearchResultItem(
            doc_id=res['doc_id'],
            score=res['score'],
            metrics=res['metrics'],
            preview=res['preview'],
            explanation=explanation
        ))
        
    return SearchResponse(results=response_items, expanded_query=expanded_query)


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

