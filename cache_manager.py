import json
import os
import hashlib

class CacheManager:
    def __init__(self, cache_file="embeddings_cache.json"):
        self.cache_file = cache_file
        self.cache = self._load_cache()

    def _load_cache(self):
        if os.path.exists(self.cache_file):
            try:
                with open(self.cache_file, 'r') as f:
                    return json.load(f)
            except json.JSONDecodeError:
                return {}
        return {}

    def save_cache(self):
        with open(self.cache_file, 'w') as f:
            json.dump(self.cache, f)

    def get_hash(self, text):
        return hashlib.sha256(text.encode('utf-8')).hexdigest()

    def get_embedding(self, text_hash):
        return self.cache.get(text_hash)

    def save_embedding(self, text_hash, data):
        self.cache[text_hash] = data
        self.save_cache()

    def is_changed(self, text_hash):
        return text_hash not in self.cache

