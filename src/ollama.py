import datetime
from typing import AsyncIterator, Callable
import aiohttp
import json
from .data_types import LLMGenerateFn, TVRequest, TVResponse
import time
import logging

async def make_ollama_request(
    url: str, 
    model_name: str, 
    prompt: str,
    on_chunk: Callable[[str], None]
) -> AsyncIterator[str]:
    """Helper function to make requests to Ollama API"""
    async with aiohttp.ClientSession() as session:
        async with session.post(
            url,
            json={"model": model_name, "prompt": prompt, "stream": True}
        ) as response:
            if response.status != 200:
                error_text = await response.text()
                raise Exception(f"Ollama API error (status {response.status}): {error_text}")
                
            async for line in response.content:
                if line:
                    data = json.loads(line)
                    if "response" in data:
                        chunk = data["response"]
                        if on_chunk:
                            on_chunk(chunk)
                        yield chunk

def create_ollama_client(
    model_name: str,
    url: str = "http://localhost:11434/api/generate"
) -> LLMGenerateFn:
    """Creates a direct streaming connection to Ollama API"""
    async def generate_llm_response(
        llm_request: TVRequest, 
        on_chunk: Callable[[str], None] = None,
        json_response: bool = True
    ) -> TVResponse:
        start_time = time.time()
        chunks = []
        
        try:
            async for chunk in make_ollama_request(url, model_name, llm_request.prompt, on_chunk):
                chunks.append(chunk)
            
            response = ''.join(chunks)

            if json_response and response.strip():
                try:
                    response = json.loads(response)
                except json.JSONDecodeError:
                    response = {"raw_text": response}

        except Exception as e:
            response = {"error": str(e)}

        return TVResponse(
            generated_at=datetime.datetime.now().isoformat(),
            request=llm_request,
            raw_response=response,
            time_in_seconds=round(time.time() - start_time, 2),
            model_name=model_name,
            model_provider="ollama"
        )
    
    return generate_llm_response
