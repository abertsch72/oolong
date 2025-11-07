"""
To run evaluation, you need an OpenAI-compatible endpoint. This example uses LiteLLM.
You should overwrite this with your own API client.
"""

import os
import litellm


class APIClient():
    def __init__(self):
        from secret import LITELLM_KEY
        os.environ["LITELLM_API_KEY"] = LITELLM_KEY


    def completion(self, model, data, api_args):
        response = litellm.completion(
            api_key=os.environ.get("LITELLM_API_KEY"),
            base_url="https://[your org].litellm.ai", # optionally specify this 
            tools=[],
            api_version="2024-12-01",
            extra_headers={"anthropic-beta": "context-1m-2025-08-07"},
            **data,
            **api_args,
        )
    
    def batch_completion(self, model, messages, api_args):
        response = litellm.batch_completion(
            api_key=os.environ.get("LITELLM_API_KEY"),
            base_url="https://[your org].litellm.ai",
            tools=[],
            api_version="2024-12-01",
            extra_headers={"anthropic-beta": "context-1m-2025-08-07"},
            **messages,
            **api_args,
        )