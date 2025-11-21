import faiss
import numpy as np
import pickle
import os
import nltk
from nltk.corpus import wordnet
from nltk.tokenize import word_tokenize
from cleaning import clean_text_pipeline

try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')
    nltk.download('omw-1.4')
    nltk.download('punkt')

class SearchEngine:
    def __init__(self, index_file="faiss_index.bin", metadata_file="faiss_metadata.pkl", dimension=768):
        self.index_file = index_file
        self.metadata_file = metadata_file
        self.dimension = dimension
        self.index = None
        self.metadata = []

    def _initialize_index(self):
        self.index = faiss.IndexFlatIP(self.dimension)

    def build_index(self, processed_data):
        self._initialize_index()
        self.metadata = []
        
        embeddings = []
        for item in processed_data:
            emb = np.array(item['embedding']).astype('float32')
            faiss.normalize_L2(emb.reshape(1, -1))
            embeddings.append(emb)
            self.metadata.append(item)
            
        if embeddings:
            embeddings_np = np.array(embeddings).astype('float32')
            self.index.add(embeddings_np)
            
        self.save_index()

    def save_index(self):
        if self.index:
            faiss.write_index(self.index, self.index_file)
            with open(self.metadata_file, 'wb') as f:
                pickle.dump(self.metadata, f)

    def load_index(self):
        if os.path.exists(self.index_file) and os.path.exists(self.metadata_file):
            self.index = faiss.read_index(self.index_file)
            with open(self.metadata_file, 'rb') as f:
                self.metadata = pickle.load(f)
            return True
        return False

    def expand_query(self, query):
        tokens = word_tokenize(query)
        expanded_tokens = set(tokens)
        
        for token in tokens:
            for syn in wordnet.synsets(token):
                for lemma in syn.lemmas():
                    expanded_tokens.add(lemma.name().replace('_', ' '))
        
        return ' '.join(list(expanded_tokens))

    def explain_result(self, query, doc_text):
        query_tokens = set(word_tokenize(query.lower()))
        doc_tokens = set(word_tokenize(doc_text.lower()))
        
        overlap = query_tokens.intersection(doc_tokens)
        overlap_ratio = len(overlap) / len(query_tokens) if query_tokens else 0
        
        return {
            "why": "High semantic similarity based on vector embedding.",
            "keywords_overlapped": list(overlap),
            "overlap_ratio": f"{overlap_ratio:.2f}",
        }

    def search(self, query, k=5, expand=True):
        if self.index is None:
            if not self.load_index():
                return []

        if expand:
            query = self.expand_query(query)
        pass

    def search_with_embedding(self, query_embedding, k=5, threshold=0.0):
        if self.index is None or self.index.ntotal == 0:
            return []
            
        query_embedding_np = np.array([query_embedding]).astype('float32')
        faiss.normalize_L2(query_embedding_np)
        
        distances, indices = self.index.search(query_embedding_np, k)
        
        results = []
        for i, idx in enumerate(indices[0]):
            if idx != -1 and idx < len(self.metadata):
                # Cosine is the score itself
                cosine = float(distances[0][i])
                
                if cosine < threshold:
                    continue

                meta = self.metadata[idx]
                
                # Calculate additional metrics
                doc_emb = np.array(meta['embedding']).astype('float32')
                # Normalize doc embedding to match query (since we used Cosine/Normalized IP)
                faiss.normalize_L2(doc_emb.reshape(1, -1))
                
                # Query embedding is already normalized as query_embedding_np[0]
                q_emb = query_embedding_np[0]
                d_emb = doc_emb
                
                # Euclidean Distance (L2)
                euclidean = np.linalg.norm(q_emb - d_emb)
                
                # Manhattan Distance (L1)
                manhattan = np.sum(np.abs(q_emb - d_emb))

                results.append({
                    "doc_id": meta.get('filename', 'unknown'),
                    "score": cosine,
                    "metrics": {
                        "Cosine Similarity": f"{cosine:.4f}",
                        "Euclidean Distance": f"{euclidean:.4f}",
                        "Manhattan Distance": f"{manhattan:.4f}"
                    },
                    "preview": meta.get('cleaned_text', '')[:200] + "...",
                    "full_text": meta.get('cleaned_text', ''),
                    "original_filename": meta.get('filename')
                })
        return results

