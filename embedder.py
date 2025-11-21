import os
from dotenv import load_dotenv
from google import genai
from google.genai import types
from cache_manager import CacheManager
from cleaning import clean_text_pipeline
import multiprocessing

load_dotenv()

class Embedder:
    def __init__(self, api_key=None, cache_manager=None):
        self.api_key = api_key or os.getenv("GOOGLE_API_KEY")
        if not self.api_key:
            raise ValueError("API Key not found. Please set GOOGLE_API_KEY in .env file.")
            
        self.client = genai.Client(api_key=self.api_key)
        self.cache_manager = cache_manager if cache_manager else CacheManager()

    def _clean_text_worker(self, text):
        return clean_text_pipeline(text)

    def process_and_embed(self, documents, filenames, batch_size=10):
        results = []
        docs_to_process = []
        cached_results = []
        
        print("Checking cache...")
        for doc, filename in zip(documents, filenames):
            doc_hash = self.cache_manager.get_hash(doc)
            cached_data = self.cache_manager.get_embedding(doc_hash)
            
            if cached_data:
                cached_results.append(cached_data)
            else:
                docs_to_process.append((doc, filename, doc_hash))
        
        print(f"Found {len(cached_results)} cached docs. {len(docs_to_process)} new docs to process.")
        
        if not docs_to_process:
            return cached_results

        print("Cleaning texts...")
        raw_texts = [item[0] for item in docs_to_process]
        
        with multiprocessing.Pool() as pool:
            cleaned_texts = pool.map(clean_text_pipeline, raw_texts)
            
        print("Generating embeddings...")
        new_results = []
        
        for i in range(0, len(cleaned_texts), batch_size):
            batch_texts = cleaned_texts[i:i+batch_size]
            batch_indices = range(i, min(i+batch_size, len(cleaned_texts)))
            
            try:
                response = self.client.models.embed_content(
                    model="gemini-embedding-001",
                    contents=batch_texts,
                    config=types.EmbedContentConfig(output_dimensionality=768)
                )
                
                if response.embeddings:
                    for j, embedding_obj in enumerate(response.embeddings):
                        idx = batch_indices[j]
                        original_idx = idx 
                        
                        doc_info = docs_to_process[original_idx]
                        filename = doc_info[1]
                        doc_hash = doc_info[2]
                        cleaned_text = batch_texts[j]
                        
                        embedding_values = embedding_obj.values
                        if hasattr(embedding_values, 'tolist'):
                            embedding_values = embedding_values.tolist()
                            
                        data = {
                            'filename': filename,
                            'cleaned_text': cleaned_text,
                            'embedding': embedding_values,
                            'doc_hash': doc_hash
                        }
                        
                        self.cache_manager.save_embedding(doc_hash, data)
                        new_results.append(data)
                        
            except Exception as e:
                print(f"Error processing batch {i}: {e}")
                
        return cached_results + new_results

    def embed_query(self, text):
        cleaned_text = clean_text_pipeline(text)
        try:
            result = self.client.models.embed_content(
                model="gemini-embedding-001",
                contents=[cleaned_text],
                config=types.EmbedContentConfig(output_dimensionality=768)
            )
            if result.embeddings:
                vals = result.embeddings[0].values
                if hasattr(vals, 'tolist'):
                    return vals.tolist()
                return vals
        except Exception as e:
            print(f"Error embedding query: {e}")
            return None

