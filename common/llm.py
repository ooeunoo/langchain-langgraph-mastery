from pydantic import SecretStr
from config.settings import settings
from langchain_openai import ChatOpenAI

def get_llm(openai_api_key: str = settings.OPENAI_API_KEY, openai_model: str = settings.OPENAI_MODEL, openai_temperature: float = 0.4):
    return ChatOpenAI(
        model=openai_model,
        temperature=openai_temperature,
        api_key=SecretStr(openai_api_key),
    )