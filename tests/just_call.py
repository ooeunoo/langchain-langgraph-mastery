"""
프로덕션 레벨 LangChain 체인 (세션 기반 메모리 관리)

포함 요소:
- 세션 매니저를 통한 대화 메모리 관리 (여러 대화 세션 지원)
- Retry with Exponential Backoff
- Circuit Breaker
- Rate Limiter
- Caching
- Input Validation
- Fallback Chain
- 상세한 메트릭 추적
- Structured Logging

장점:
- 여러 주제의 대화를 별도 세션으로 분리 가능
- 세션별 독립적인 컨텍스트 유지
- 세션 목록 조회 및 관리 용이
"""

import time
from datetime import datetime
from typing import Dict, Any, Optional, List, Tuple
import logging
import hashlib

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, AIMessage

from pydantic import SecretStr
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
)

from config import settings

# Logging 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ===== Configuration =====
class ChainConfig:
    """체인 설정"""
    MAX_INPUT_LENGTH = 10000
    TIMEOUT_SECONDS = 30
    MAX_RETRIES = 3
    CACHE_TTL = 3600  # 1시간
    MAX_TOKENS = 1000
    RATE_LIMIT_PER_MINUTE = 60
    MEMORY_WINDOW_SIZE = 10  # 최근 10개 대화만 유지


# ===== Simple Memory =====
class SimpleConversationMemory:
    """
    간단한 대화 메모리 (Window 방식)
    
    최근 N개의 대화만 유지하여 토큰 비용 절감
    """
    
    def __init__(self, window_size: int = 10):
        """
        Args:
            window_size: 유지할 최근 대화 개수 (user-ai 페어 기준)
        """
        self.window_size = window_size
        self.conversations: List[Tuple[str, str]] = []  # [(user_msg, ai_msg), ...]
    
    def add_conversation(self, user_message: str, ai_message: str):
        """대화 추가"""
        self.conversations.append((user_message, ai_message))
        
        # Window size 유지 (최근 N개만)
        if len(self.conversations) > self.window_size:
            self.conversations = self.conversations[-self.window_size:]
    
    def get_messages(self) -> List:
        """메시지 목록 반환 (LangChain 형식)"""
        messages = []
        for user_msg, ai_msg in self.conversations:
            messages.append(HumanMessage(content=user_msg))
            messages.append(AIMessage(content=ai_msg))
        return messages
    
    def get_conversations(self) -> List[Tuple[str, str]]:
        """대화 목록 반환"""
        return self.conversations.copy()
    
    def clear(self):
        """메모리 초기화"""
        self.conversations = []
    
    def get_conversation_count(self) -> int:
        """저장된 대화 개수"""
        return len(self.conversations)


# ===== Session Manager =====
class SessionManager:
    """
    세션 기반 메모리 관리
    
    여러 대화 세션을 독립적으로 관리
    예: "Python 공부" 세션, "여행 계획" 세션 등
    """
    
    def __init__(self, memory_window_size: int = 10):
        self.memory_window_size = memory_window_size
        self.sessions: Dict[str, Dict[str, Any]] = {}
        self.default_session_id = "default"
    
    def create_session(self, session_name: str | None = None) -> str:
        """
        새 세션 생성
        
        Args:
            session_name: 세션 이름 (없으면 자동 생성)
        
        Returns:
            생성된 세션 ID
        """
        if session_name:
            session_id = self._generate_session_id_from_name(session_name)
        else:
            session_id = self._generate_session_id()
        
        if session_id in self.sessions:
            return session_id
        
        self.sessions[session_id] = {
            "memory": SimpleConversationMemory(window_size=self.memory_window_size),
            "metadata": {
                "session_id": session_id,
                "session_name": session_name or session_id,
                "created_at": datetime.now().isoformat(),
                "message_count": 0,
                "last_updated": datetime.now().isoformat()
            }
        }
        
        return session_id
    
    def get_session(self, session_id: str | None = None) -> Dict[str, Any]:
        """
        세션 가져오기 (없으면 기본 세션 반환)
        
        Args:
            session_id: 세션 ID (없으면 기본 세션)
        
        Returns:
            세션 딕셔너리
        """
        if session_id is None:
            session_id = self.default_session_id
        
        # 세션이 없으면 자동 생성
        if session_id not in self.sessions:
            logger.info(f"ℹ️  세션 '{session_id}'가 없어 자동 생성합니다")
            self.create_session(session_name=session_id)
        
        return self.sessions[session_id]
    
    def add_message(self, user_message: str, ai_message: str, session_id: str | None = None):
        """
        세션에 메시지 추가
        
        Args:
            user_message: 사용자 메시지
            ai_message: AI 응답
            session_id: 세션 ID
        """
        session = self.get_session(session_id)
        memory = session["memory"]
        
        # 메모리에 저장
        memory.add_conversation(user_message, ai_message)
        
        # 메타데이터 업데이트
        session["metadata"]["message_count"] += 1
        session["metadata"]["last_updated"] = datetime.now().isoformat()
    
    def get_history(self, session_id: str | None = None) -> Dict[str, Any]:
        """
        세션의 대화 히스토리 조회
        
        Returns:
            {"conversations": [...], "metadata": {...}}
        """
        session = self.get_session(session_id)
        memory = session["memory"]
        
        return {
            "conversations": memory.get_conversations(),
            "metadata": session["metadata"]
        }
    
    def get_messages(self, session_id: str | None = None) -> List:
        """세션의 메시지 목록 (LangChain 형식)"""
        session = self.get_session(session_id)
        memory = session["memory"]
        return memory.get_messages()
    
    def list_sessions(self) -> List[Dict[str, Any]]:
        """모든 세션 목록 조회"""
        return [
            {
                "session_id": session_id,
                **session["metadata"]
            }
            for session_id, session in self.sessions.items()
        ]
    
    def delete_session(self, session_id: str):
        """세션 삭제"""
        if session_id in self.sessions:
            del self.sessions[session_id]
            logger.info(f"🗑️  세션 삭제: {session_id}")
        else:
            logger.warning(f"⚠️  세션 '{session_id}'를 찾을 수 없습니다")
    
    def clear_session(self, session_id: str | None = None):
        """세션 메모리 초기화 (메타데이터는 유지)"""
        session = self.get_session(session_id)
        session["memory"].clear()
        session["metadata"]["message_count"] = 0
        session["metadata"]["last_updated"] = datetime.now().isoformat()
        logger.info(f"🧹 세션 메모리 초기화: {session_id or self.default_session_id}")
    
    def _generate_session_id(self) -> str:
        """고유한 세션 ID 생성"""
        timestamp = datetime.now().isoformat()
        return hashlib.md5(timestamp.encode()).hexdigest()[:8]
    
    def _generate_session_id_from_name(self, name: str) -> str:
        """이름으로부터 세션 ID 생성"""
        return hashlib.md5(name.encode()).hexdigest()[:8]


# ===== Metrics =====
class ProductionMetrics:
    """프로덕션 메트릭 수집"""
    
    def __init__(self):
        self.calls: List[Dict[str, Any]] = []
        self.cache_hits = 0
        self.cache_misses = 0
        
    def record_call(self, 
                   success: bool, 
                   latency_ms: float,
                   tokens: int = 0,
                   cost: float = 0.0,
                   error_type: Optional[str] = None,
                   cached: bool = False):
        """호출 기록"""
        self.calls.append({
            "timestamp": datetime.now().isoformat(),
            "success": success,
            "latency_ms": latency_ms,
            "tokens": tokens,
            "cost": cost,
            "error_type": error_type,
            "cached": cached
        })
    
    def record_cache_hit(self):
        """캐시 히트 기록"""
        self.cache_hits += 1
    
    def record_cache_miss(self):
        """캐시 미스 기록"""
        self.cache_misses += 1
    
    def get_stats(self) -> Dict[str, Any]:
        """통계 조회"""
        if not self.calls:
            return {"status": "no_calls_yet"}
        
        successful = [c for c in self.calls if c["success"]]
        failed = [c for c in self.calls if not c["success"]]
        
        avg_latency = sum(c["latency_ms"] for c in successful) / len(successful) if successful else 0
        
        # P95 계산
        if successful:
            sorted_latencies = sorted([c["latency_ms"] for c in successful])
            p95_index = int(len(sorted_latencies) * 0.95)
            p95_latency = sorted_latencies[p95_index] if p95_index < len(sorted_latencies) else sorted_latencies[-1]
        else:
            p95_latency = 0
        
        total_cache = self.cache_hits + self.cache_misses
        cache_hit_rate = (self.cache_hits / total_cache * 100) if total_cache > 0 else 0
        
        return {
            "total_calls": len(self.calls),
            "successful": len(successful),
            "failed": len(failed),
            "success_rate": f"{len(successful)/len(self.calls)*100:.2f}%",
            "avg_latency_ms": f"{avg_latency:.2f}",
            "p95_latency_ms": f"{p95_latency:.2f}",
            "total_tokens": sum(c["tokens"] for c in self.calls),
            "total_cost": f"${sum(c['cost'] for c in self.calls):.4f}",
            "cache_hit_rate": f"{cache_hit_rate:.2f}%",
            "error_types": list({c["error_type"] for c in failed if c["error_type"]})
        }


# ===== Cache =====
class SimpleCache:
    """간단한 메모리 캐시"""
    
    def __init__(self, ttl: int):
        self.cache: Dict[str, Tuple[Any, float]] = {}
        self.ttl = ttl
    
    def get(self, key: str) -> Optional[Any]:
        """캐시 조회"""
        if key in self.cache:
            value, timestamp = self.cache[key]
            if time.time() - timestamp < self.ttl:
                return value
            else:
                del self.cache[key]
        return None
    
    def set(self, key: str, value: Any):
        """캐시 저장"""
        self.cache[key] = (value, time.time())
    
    def clear(self):
        """캐시 초기화"""
        self.cache.clear()


# ===== Circuit Breaker =====
class CircuitBreaker:
    """Circuit Breaker 패턴"""
    
    def __init__(self, failure_threshold: int = 5, timeout: int = 60):
        self.failure_threshold = failure_threshold
        self.timeout = timeout
        self.failures = 0
        self.last_failure_time = None
        self.state = "CLOSED"
    
    def call(self, func):
        """Circuit Breaker를 통한 함수 호출"""
        if self.state == "OPEN":
            if  self.last_failure_time and time.time() - self.last_failure_time > self.timeout:
                self.state = "HALF_OPEN"
                logger.info("🔄 Circuit Breaker: HALF_OPEN - 복구 테스트 중")
            else:
                raise Exception(f"Circuit Breaker가 OPEN 상태입니다. {self.timeout}초 후 재시도하세요.")
        
        try:
            result = func()
            if self.state == "HALF_OPEN":
                self.state = "CLOSED"
                self.failures = 0
                logger.info("✅ Circuit Breaker: CLOSED - 정상 복구됨")
            return result
        except Exception as e:
            self.failures += 1
            self.last_failure_time = time.time()
            
            if self.failures >= self.failure_threshold:
                self.state = "OPEN"
                logger.error(f"🚨 Circuit Breaker: OPEN - 장애 감지 (연속 실패: {self.failures}회)")
            raise e


# ===== Rate Limiter =====
class RateLimiter:
    """Rate Limiter - API 호출 횟수 제한"""
    
    def __init__(self, max_calls: int, time_window: int = 60):
        self.max_calls = max_calls
        self.time_window = time_window
        self.calls: List[float] = []
    
    def check(self):
        """Rate limit 체크"""
        now = time.time()
        self.calls = [t for t in self.calls if now - t < self.time_window]
        
        if len(self.calls) >= self.max_calls:
            raise Exception(f"⚠️ Rate Limit 초과: {self.max_calls}회/{self.time_window}초")
        
        self.calls.append(now)


# ===== Input Validation =====
def validate_and_sanitize_input(text: str, max_length: int = 10000) -> str:
    """입력 검증 및 정제"""
    if not text or not isinstance(text, str):
        raise ValueError("입력이 비어있거나 문자열이 아닙니다")
    
    if len(text) > max_length:
        raise ValueError(f"입력이 너무 깁니다: {len(text)} > {max_length}")
    
    dangerous_patterns = ['<script>', 'DROP TABLE', 'DELETE FROM', '<?php']
    text_lower = text.lower()
    for pattern in dangerous_patterns:
        if pattern.lower() in text_lower:
            raise ValueError(f"위험한 입력 패턴 감지: {pattern}")
    
    return text.strip()


# ===== Production Chain =====
class ProductionChain:
    """프로덕션 레벨 LangChain 체인 (세션 기반)"""
    
    def __init__(self, openai_api_key: str, config: ChainConfig | None = None):
        self.config = config or ChainConfig()
        self.openai_api_key = openai_api_key
        
        # 컴포넌트 초기화
        self.session_manager = SessionManager(memory_window_size=self.config.MEMORY_WINDOW_SIZE)
        self.metrics = ProductionMetrics()
        self.cache = SimpleCache(ttl=self.config.CACHE_TTL)
        self.circuit_breaker = CircuitBreaker()
        self.rate_limiter = RateLimiter(max_calls=self.config.RATE_LIMIT_PER_MINUTE)
        
        # LLM 초기화
        self.llm = ChatOpenAI(
            model=settings.OPENAI_MODEL,
            temperature=0,
            api_key=SecretStr(openai_api_key),
            timeout=self.config.TIMEOUT_SECONDS,
            max_completion_tokens=self.config.MAX_TOKENS
        )
        
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        before_sleep=lambda retry_state: logger.warning(
            f"🔄 재시도 중... ({retry_state.attempt_number}/3)"
        )
    )
    def _call_llm_with_retry(self, prompt_messages: List, input_text: str, 
                            chat_history: List) -> Tuple[str, int, float]:
        """Retry 로직이 포함된 LLM 호출"""
        prompt = ChatPromptTemplate.from_messages(prompt_messages)
        chain = prompt | self.llm | StrOutputParser()
        
        response = chain.invoke({
            "chat_history": chat_history,
            "input": input_text
        })
        
        # 토큰/비용 추정
        estimated_tokens = len(input_text + response) // 4
        estimated_cost = estimated_tokens * 0.000001
        
        return response, estimated_tokens, estimated_cost
    
    def _fallback_response(self, error_msg: str) -> str:
        """LLM 실패 시 대체 응답"""
        return (
            f"죄송합니다. 현재 요청을 처리할 수 없습니다. "
            f"잠시 후 다시 시도해주세요.\n(오류: {error_msg})"
        )
    
    def chat(self, text: str, session_id: str | None = None, use_cache: bool = True) -> Dict[str, Any]:
        """
        대화 처리
        
        Args:
            text: 사용자 입력
            session_id: 세션 ID (없으면 기본 세션)
            use_cache: 캐시 사용 여부
        
        Returns:
            처리 결과 딕셔너리
        """
        start_time = time.time()
        
        try:
            # 1. Rate limiting 체크
            self.rate_limiter.check()
            
            # 2. Input validation
            validated_text = validate_and_sanitize_input(text, self.config.MAX_INPUT_LENGTH)
            
            # 3. 세션 가져오기
            session = self.session_manager.get_session(session_id)
            current_session_id = session["metadata"]["session_id"]
            
            # 4. Cache 체크
            cache_key = f"chat:{hash(validated_text)}:{current_session_id}"
            
            if use_cache:
                cached_result = self.cache.get(cache_key)
                if cached_result:
                    self.metrics.record_cache_hit()
                    latency_ms = (time.time() - start_time) * 1000
                    
                    logger.info(f"💾 캐시 HIT - Latency: {latency_ms:.2f}ms")
                    
                    self.metrics.record_call(success=True, latency_ms=latency_ms, cached=True)
                    
                    return {
                        "success": True,
                        "response": cached_result,
                        "cached": True,
                        "session_id": current_session_id,
                        "message_count": session["metadata"]["message_count"],
                        "latency_ms": latency_ms,
                        "tokens": 0,
                        "cost": 0.0
                    }
            
            self.metrics.record_cache_miss()
            
            # 5. 대화 히스토리 로드
            chat_history = self.session_manager.get_messages(session_id)
            
            # 6. 프롬프트 구성
            prompt_messages = [
                ("system", "당신은 친절하고 도움이 되는 AI 어시스턴트입니다."),
                ("placeholder", "{chat_history}"),
                ("human", "{input}")
            ]
            
            # 7. Circuit Breaker를 통한 LLM 호출
            def llm_call():
                return self._call_llm_with_retry(prompt_messages, validated_text, chat_history)
            
            response, tokens, cost = self.circuit_breaker.call(llm_call)
            
            # 8. Cache 저장
            if use_cache:
                self.cache.set(cache_key, response)
            
            # 9. 세션에 저장
            self.session_manager.add_message(validated_text, response, session_id)
            
            # 10. 메트릭 기록
            latency_ms = (time.time() - start_time) * 1000
            self.metrics.record_call(
                success=True,
                latency_ms=latency_ms,
                tokens=tokens,
                cost=cost,
                cached=False
            )
            
            
            return {
                "success": True,
                "response": response,
                "cached": False,
                "session_id": current_session_id,
                "message_count": session["metadata"]["message_count"],
                "latency_ms": latency_ms,
                "tokens": tokens,
                "cost": cost
            }
            
        except ValueError as e:
            latency_ms = (time.time() - start_time) * 1000
            self.metrics.record_call(success=False, latency_ms=latency_ms, error_type="ValidationError")
            return {
                "success": False,
                "response": None,
                "error": f"입력 검증 실패: {str(e)}",
                "error_type": "ValidationError",
                "latency_ms": latency_ms
            }
            
        except Exception as e:
            latency_ms = (time.time() - start_time) * 1000
            error_type = type(e).__name__
            self.metrics.record_call(success=False, latency_ms=latency_ms, error_type=error_type)
            fallback_result = self._fallback_response(str(e))
            
            return {
                "success": False,
                "response": fallback_result,
                "error": str(e),
                "error_type": error_type,
                "latency_ms": latency_ms,
                "fallback_used": True
            }
    
    # ===== 세션 관리 메서드 =====
    
    def create_session(self, session_name: str | None = None) -> str:
        """새 세션 생성"""
        return self.session_manager.create_session(session_name)
    
    def list_sessions(self) -> List[Dict[str, Any]]:
        """모든 세션 목록"""
        return self.session_manager.list_sessions()
    
    def get_session_history(self, session_id: str | None = None) -> Dict[str, Any]:
        """세션 히스토리 조회"""
        return self.session_manager.get_history(session_id)
    
    def delete_session(self, session_id: str):
        """세션 삭제"""
        self.session_manager.delete_session(session_id)
    
    def clear_session(self, session_id: str | None = None):
        """세션 메모리 초기화"""
        self.session_manager.clear_session(session_id)
    
    def clear_cache(self):
        """캐시 초기화"""
        self.cache.clear()
    
    def get_metrics(self) -> Dict[str, Any]:
        """메트릭 조회"""
        return self.metrics.get_stats()


# ===== 실행 예제 =====
def main():
    """메인 실행 함수"""
    
    import os
    OPENAI_API_KEY = settings.OPENAI_API_KEY
    
    # 체인 초기화
    chain = ProductionChain(openai_api_key=OPENAI_API_KEY)
    
    print("="*80)
    print("🚀 프로덕션 레벨 LangChain 체인 테스트 (세션 기반)")
    print("="*80)
    
    # ===== 시나리오 1: 기본 세션에서 대화 =====
    print("\n" + "="*80)
    print("📘 시나리오 1: 기본 세션에서 대화")
    print("="*80)
    
    conversations_1 = [
        "안녕하세요! 제 이름은 홍길동입니다.",
        "저는 AI 개발자예요.",
        "제 이름이 뭐였죠?",
    ]
    
    for text in conversations_1:
        print(f"\n👤 User: {text}")
        result = chain.chat(text)
        if result["success"]:
            print(f"🤖 AI: {result['response'][:150]}...")
    
    # ===== 시나리오 2: 새로운 세션 생성 (Python 공부) =====
    print("\n" + "="*80)
    print("📗 시나리오 2: 'Python 공부' 세션 생성")
    print("="*80)
    
    python_session = chain.create_session("Python 공부")
    
    conversations_2 = [
        "Python의 데코레이터에 대해 설명해줘",
        "실용적인 예제를 보여줘",
    ]
    
    for text in conversations_2:
        print(f"\n👤 User: {text}")
        result = chain.chat(text, session_id=python_session)
        if result["success"]:
            print(f"🤖 AI: {result['response'][:150]}...")
    
    # ===== 시나리오 3: 또 다른 세션 생성 (여행 계획) =====
    print("\n" + "="*80)
    print("📙 시나리오 3: '여행 계획' 세션 생성")
    print("="*80)
    
    travel_session = chain.create_session("여행 계획")
    
    conversations_3 = [
        "제주도 3박 4일 여행 계획을 세우고 싶어",
        "숙소는 어디가 좋을까?",
    ]
    
    for text in conversations_3:
        print(f"\n👤 User: {text}")
        result = chain.chat(text, session_id=travel_session)
        if result["success"]:
            print(f"🤖 AI: {result['response'][:150]}...")
    
    # ===== 세션 목록 조회 =====
    print("\n" + "="*80)
    print("📋 전체 세션 목록")
    print("="*80)
    
    sessions = chain.list_sessions()
    for session in sessions:
        print(f"\n세션: {session['session_name']}")
        print(f"  - ID: {session['session_id']}")
        print(f"  - 메시지 수: {session['message_count']}")
        print(f"  - 생성 시간: {session['created_at']}")
        print(f"  - 마지막 업데이트: {session['last_updated']}")
    
    # ===== 특정 세션 히스토리 조회 =====
    print("\n" + "="*80)
    print(f"📚 'Python 공부' 세션 히스토리")
    print("="*80)
    
    history = chain.get_session_history(python_session)
    for i, (user_msg, ai_msg) in enumerate(history["conversations"], 1):
        print(f"\n[대화 #{i}]")
        print(f"👤: {user_msg}")
        print(f"🤖: {ai_msg[:100]}...")
    
    # ===== 최종 메트릭 =====
    print("\n" + "="*80)
    print("📊 최종 메트릭")
    print("="*80)
    
    metrics = chain.get_metrics()
    for key, value in metrics.items():
        print(f"{key}: {value}")
    
    print("\n" + "="*80)
    print("✅ 테스트 완료")
    print("="*80)


if __name__ == "__main__":
    main()