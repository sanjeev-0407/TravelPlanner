from langchain.embeddings.base import Embeddings
from typing import List
import requests

class JinaEmbeddings(Embeddings):
    def __init__(self):
        self.api_key = "jina_d1f20ceaa138457c8f8fe46db436665b3xRYUXn_hqA5lUuHnlDF9xH5kpGZ"
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        embeddings = []
        for text in texts:
            payload = {
                "model": "jina-clip-v2",
                "dimensions": 1024,
                "normalized": True,
                "embedding_type": "float",
                "input": [{"text": text}]
            }
            response = requests.post(
                "https://api.jina.ai/v1/embeddings", 
                json=payload, 
                headers=self.headers
            )
            if response.status_code == 200:
                embeddings.append(response.json()['data'][0]['embedding'])
            else:
                raise Exception(f"API Request Failed: {response.status_code}")
        return embeddings
    
    def embed_query(self, text: str) -> List[float]:
        return self.embed_documents([text])[0]