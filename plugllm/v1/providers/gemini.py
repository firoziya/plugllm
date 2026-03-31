import requests
from ..config import CONFIG
from .base import ProviderBase

class GeminiProvider(ProviderBase):
    def send(self, messages):
        url = CONFIG["base_url"] or f"https://generativelanguage.googleapis.com/v1beta/models/{CONFIG['model']}:generateContent?key={CONFIG['api_key']}"
        contents = [{"parts": [{"text": m["content"]}]} for m in messages]
        payload = {"contents": contents}
        response = requests.post(url, json=payload)
        return response.json()["candidates"][0]["content"]["parts"][0]["text"]
