from functools import lru_cache
from pathlib import Path
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


# 프로젝트 루트 디렉토리
PROJECT_ROOT = Path(__file__).parent.parent


class Settings(BaseSettings):
    # OpenAI
    OPENAI_API_KEY: str = Field(..., description="OpenAI API 키")
    OPENAI_MODEL: str = Field(default="gpt-4o-mini", description="OpenAI 모델명")
    OPENAI_TEMPERATURE: float = Field(default=0.7, description="OpenAI Temperature")
    
    # Vector Store
    CHROMA_PERSIST_DIR: str = Field(
        default="./data/chroma",
        description="ChromaDB 데이터 저장 경로"
    )
    
    # LangSmith (Optional)
    LANGCHAIN_API_KEY: str = Field(default="", description="LangSmith API 키")
    LANGCHAIN_TRACING_V2: bool = Field(default=False, description="LangSmith 추적 활성화")
    LANGCHAIN_PROJECT: str = Field(default="langchain-langgraph-mastery", description="LangSmith 프로젝트명")
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="ignore"
    )


@lru_cache
def get_settings() -> Settings:
    """설정 인스턴스를 반환 (캐싱됨)"""
    return Settings()  # type: ignore[call-arg]


# 편의를 위한 전역 인스턴스
settings = get_settings()