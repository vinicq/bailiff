import logging

import httpx
import instructor
import openai

logger = logging.getLogger("bailiff.features.assistant.llm")


class LLMClientSettings:
    def __init__(self, api_key: str, base_url: str = "https://api.openai.com/v1", model: str = "gpt-4o-mini"):
        self.api_key = api_key
        self.base_url = base_url
        self.model = model


class LLMClient:
    """
    Client for interacting with Large Language Models.

    Wraps the OpenAI client with instructor.from_openai to provide a streamlined interface for
    sending messages and receiving completion responses from the configured LLM.
    """
    def __init__(self, settings: LLMClientSettings):
        self._openai = openai.OpenAI(
            api_key=settings.api_key,
            base_url=settings.base_url,
            timeout=httpx.Timeout(30.0, connect=5.0),
            max_retries=2,
        )
        self.structured = instructor.from_openai(self._openai)
        self.model = settings.model

    def chat(self, messages: list[dict]) -> str:
        response = self._openai.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=0.3,
            timeout=30,
        )
        content = response.choices[0].message.content
        if content is None:
            logger.warning("LLM returned message with content=None")
            return "[no response]"
        return content
