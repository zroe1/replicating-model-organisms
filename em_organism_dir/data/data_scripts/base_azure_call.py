import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()


client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
)

class AzureRequest:
    """
    Class for making OpenAI requests.
    Args:
        model: The model to use (default: gpt-4o)
        max_tokens: Maximum number of tokens to generate
        temperature: Temperature for sampling
    Returns:
        completion: The completion from the OpenAI request
    """

    def __init__(self, model: str = "gpt-4o", max_tokens: int = 200, temperature: float = 1):
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature

    async def request(self, messages) -> dict:
        completion = client.chat.completions.create(
            model=self.model,
            messages=messages,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
        )
        return completion.choices[0].message.content