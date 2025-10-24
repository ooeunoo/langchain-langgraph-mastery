"""
Langchain의 다양한 기능을 포함한 프로덕션 레벨 체인 구성
- RunnableWithMessageHistory: 대화 메모리 관리
- RunnableWithFallbacks: Fallback chain
- RunnableRetry: Retry 로직
- callbacks: 토큰/비용 추적, 로깅
- cache: 중복 호출 방지
- async: 비동기 처리
"""


import os
import asyncio
from datetime import datetime
from typing import Dict, Any, List, Optional
import logging

# LangChain Core
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import (
    RunnablePassthrough,
    RunnableLambda,
    RunnableWithFallbacks,
    RunnableConfig
)
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import AIMessage, HumanMessage, trim_messages
from langchain_core.chat_history import BaseChatMessageHistory, InMemoryChatMessageHistory

# LangChain OpenAI
from langchain_openai import ChatOpenAI

# LangChain Community
from langchain_community.callbacks import get_openai_callback
from langchain_community.cache import InMemoryCache

# Callbacks
from langchain_core.callbacks import (
    BaseCallbackHandler,
    AsyncCallbackHandler,
    StdOutCallbackHandler
)
from langchain_core.outputs import LLMResult
from pydantic import SecretStr

from config import settings
from utils.logger import setup_logger

from langsmith import Client
from langchain_core.globals import set_llm_cache
set_llm_cache(InMemoryCache())

logger = setup_logger('production_chain')

class CallbackHandler(BaseCallbackHandler):
    """
    Callback Handler - 모니터링
    """
    def __init__(self):
        self.metrics = {
            "total_calls": 0,
            "total_tokens": 0,
            "total_cost": 0,
            "errors": []
        }
        self.start_time = None

    def on_llm_start(self, serialized: Dict[str, Any], prompt: List[str], **kwargs):
        self.start_time = datetime.now()

    def on_llm_end(self, response: LLMResult, **kwargs):
        if self.start_time:
            latency = (datetime.now() - self.start_time).total_seconds() * 1000

        self.metrics['total_calls'] += 1

        if response.llm_output:
            token_usage = response.llm_output.get("token_usage", {})
            tokens = token_usage.get('total_tokens', 0)
            self.metrics['total_tokens'] += tokens

    def on_llm_error(self, error: Exception, **kwargs):
        self.metrics['errors'].append({
            'error': str(error),
            'timestamp': datetime.now().isoformat()
        })

    def on_chain_start(self, serialized: Dict[str, Any], inputs: Dict[str, Any], **kwargs):
        logger.info(f"Chain 시작")
    
    def on_chain_end(self, outputs: Dict[str, Any], **kwargs):
        logger.info(f"Chain 완료")

    def get_metrics(self) -> Dict[str, Any]:
        return self.metrics
    

class RateLimitCallbackHandler(BaseCallbackHandler):
    """
    Rate Limiting Callback Handler
    """
    def __init__(self, max_call_per_minutes: int = 60):
        self.max_calls_per_minute = max_call_per_minutes
        self.call_timestamps: List[datetime] = []

    def on_llm_start(self, serialized: Dict[str, Any], prompts: List[str], **kwargs):
        now = datetime.now()

        self.call_timestamps = [
            ts for ts in self.call_timestamps
            if (now - ts).total_seconds() < 60
        ]

        if len(self.call_timestamps) >= self.max_calls_per_minute:
            raise Exception(f"Rate Limit 초과: {self.max_calls_per_minute}회/분")
        
        self.call_timestamps.append(now)

class SessionStore:
    """
    세션 별 메시지 히스토리 저장소 - RunnableWithMessageHistory
    """
    def __init__(self):
        self.store: Dict[str, InMemoryChatMessageHistory] = {}

    def get_session_history(self, session_id: str) -> BaseChatMessageHistory:
        if session_id not in self.store:
            self.store[session_id] = InMemoryChatMessageHistory()
        return self.store[session_id]
    
    def list_sessions(self) -> List[str]:
        return list(self.store.keys())
    
    def clear_session(self, session_id: str):
        if session_id in self.store:
            self.store[session_id].clear()

    def delete_session(self, session_id: str):
        if session_id in self.store:
            del self.store[session_id]


class InputValidationRunnable(RunnableLambda):
    """
    입력 검증 Runnable
    """
    def __init__(self, max_length: int = 10000):
        self.max_length = max_length
        super().__init__(self._validate)

    def _validate(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        text = inputs.get("input", "")

        # 빈 값 체크
        if not text or not isinstance(text, str):
            raise ValueError("입력이 비어있거나 문자열이 아닙니다.")
        
        # 길이 체크
        if len(text) > self.max_length:
            raise ValueError(f"입력이 너무 깁니다: {len(text)} > {self.max_length}")
        
        # 위험 패턴 체크
        dangerouse_pattern = ['<script>', 'DROP TABLE', 'DELETE FROM']
        text_lower = text.lower()
        for pattern in dangerouse_pattern:
            if pattern.lower() in text_lower:
                raise ValueError(f"위험한 입력 패턴 감지: {pattern}")
            
        inputs['input'] = text.strip()
        return inputs
    
class ChainBuilder:
    """
    Langchain 빌더
    """

    def __init__(
        self,
        openai_api_key: str,
        openai_model: str,
        temperature: float = 0,
        max_tokens: int = 1000,
        timeout: int = 30,
        max_retries: int = 3
    ):
        self.openai_api_key = openai_api_key
        self.openai_model = openai_model
        self.temperature = temperature
        self.max_token = max_tokens
        self.timeout = timeout
        self.max_retries = max_retries

        self.session_store = SessionStore()
        
        self.callback = CallbackHandler()
        self.rate_limit_callback = RateLimitCallbackHandler(max_call_per_minutes=60)


    def build_chain(self) -> RunnableWithMessageHistory:
        """
        체인 구성
        1. Input Validation
        2. Prompt Template With Message History
        3. LLM with Retry & Timeout
        4. Fallback Chain
        5. Output Parser
        6. Message History Integration
        """

        # llm 초기화
        primary_llm  = self._generate_primary_llm()
        fallback_llm = self._generate_fallback_llm()

        llm_with_fallback = primary_llm.with_fallbacks([fallback_llm])

        # Message Trimmer (토큰 제한)
        trimmer = trim_messages(
            max_tokens=2000,
            strategy='last',
            token_counter=len
        )

        # Prompt template
        prompt = ChatPromptTemplate.from_messages([
            ("system", "당신은 친절하고 도움이 되는 AI 어시스턴트입니다."),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}")
        ])

        # Input Validation
        input_validator = InputValidationRunnable(max_length=10000)

        # Chain Combination
        chain = (
            input_validator
            | RunnablePassthrough.assign(
                chat_history=lambda x: trimmer.invoke(x.get('chat_history', []))
            )
            | prompt
            | llm_with_fallback
            | StrOutputParser()
        )

        # Message History Integration
        chain_with_history = RunnableWithMessageHistory(
            chain,
            get_session_history=self.session_store.get_session_history,
            input_messages_key='input',
            history_messages_key='chat_history'
        ) 

        return chain_with_history 

    def build_async_chain(self) -> RunnableWithMessageHistory:
        return self.build_chain()

    def _generate_primary_llm(self):
        return ChatOpenAI(
            api_key=SecretStr(self.openai_api_key),
            model=self.openai_model,
            temperature=self.temperature,
            max_completion_tokens=self.max_token,
            timeout=self.timeout,
            max_retries=self.max_retries,
            cache=True,
            callbacks=[self.callback, self.rate_limit_callback]
        )
    
    def _generate_fallback_llm(self):
        return ChatOpenAI(
            api_key=SecretStr(self.openai_api_key),
            model=self.openai_model,
            temperature=self.temperature,
            max_completion_tokens=500,
            timeout=15,
            max_retries=1,
            cache=True,
            callbacks=[self.callback]   
        )
    

class ChainManager:
    """
    체인 관리자 - 생성, 실행, 모니터링 통합 관리
    """

    def __init__(self, openai_api_key: str, openai_model):
        self.builder = ChainBuilder(openai_api_key=openai_api_key, openai_model=openai_model)
        self.chain = self.builder.build_chain()
        self.async_chain = self.builder.build_async_chain()

    def chat(self, message: str, session_id: str = 'default') -> Dict[str, Any]:
        try :
            config = RunnableConfig(
                configurable={"session_id": session_id}
            )

            with get_openai_callback() as cb:
                response = self.chain.invoke(
                    {"input": message},
                    config=config
                )

                return {
                    "success": True,
                    "response": response,
                    "session_id": session_id,
                    "tokens": cb.total_tokens,
                    "cost": cb.total_cost,
                    "prompt_tokens": cb.prompt_tokens,
                    "completion_tokens": cb.completion_tokens
                }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "error_type": type(e).__name__
            }
        

    async def chat_async(self, message: str, session_id: str = "default") -> Dict[str, Any]:
        try:
            config = RunnableConfig(
                configurable={"session_id": session_id}
            )

            response = await self.async_chain.ainvoke(
                {"input": message},
                config=config
            )
            return {
                "success": True,
                "response": response,
                "session_id": session_id
            }

        except Exception as e:
            return {
                "success": True,
                "error": str(e),
                "error_type": type(e).__name__
            }
        
    def stream_chat(self, message: str, session_id: str = 'default'):
        try:
            config = RunnableConfig(
                configurable={"session_id": session_id}
            )

            for chunk in self.chain.stream(
                {"input": message},
                config=config
            ):
                yield chunk


        except Exception as e:
            yield f"Error: {str(e)}"

    def list_sessions(self) -> List[str]:
        return self.builder.session_store.list_sessions()
    
    def clear_session(self, session_id: str):
        self.builder.session_store.clear_session(session_id)

    def delete_session(self, session_id: str):
        return self.builder.session_store.delete_session(session_id)
    
    def get_session_history(self, session_id: str) -> List[Any]:
        history = self.builder.session_store.get_session_history(session_id)
        return history.messages
    
    def get_metrics(self) -> Dict[str, Any]:
        return self.builder.callback.get_metrics()
    

def main():
    OPENAI_API_KEY = settings.OPENAI_API_KEY
    OPENAI_MODEL = settings.OPENAI_MODEL

    manager = ChainManager(openai_api_key=OPENAI_API_KEY, openai_model=OPENAI_MODEL)

    def first_session():
        conversations = [
            "안녕하세요. 제 이름은 홍길동입니다.",
            "저의 직업은 AI 개발자에요.",
            "제 이름과 직업이 뭐였죠?"
        ]

        for msg in conversations:
            result = manager.chat(msg, session_id="first_session")
            if result["success"]:
                logger.info(f"User: {msg}")
                logger.info(f"AI: {result['response']}")
                logger.info(f"Token: {result['tokens']} (Prompt: {result['prompt_tokens']}, Completion: {result['completion_tokens']})")
                logger.info(f"Cose: {result['cost']:.6f}")


    def second_session():
        question = "AI 개발자가 되려면 어떤걸 공부해야해?"
        logger.info(f"User: {question}")
        logger.info(f"AI: ")
        for chunk in manager.stream_chat(question, session_id="second_session"):
            print(chunk, end="", flush=True)
    
    first_session()
    second_session()

if __name__ == "__main__":
    main()