from typing import Dict, Any, List
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from pydantic import SecretStr, BaseModel, Field

from common.llm import get_llm
from config.settings import settings
from utils.logger import setup_logger

logger = setup_logger(__name__)

def custom_chain_example_1():
    """
    예제 1: 커스텀 체인
    """
    # llm 초기화
    llm: ChatOpenAI = get_llm()

    # 커스텀 전처리 함수 정의
    def preprocess(inputs: Dict[str, Any]) -> Dict[str, Any]:
        # 입력 텍스트를 소문자로 변환
        text = inputs["input_text"]
        
        # 텍스트 길이
        length = len(text)

        # 처리 레벨
        if length < 10:
            level = "easy"
        elif length < 50:
            level = "medium"
        else:
            level = "hard"

        return {
            "text": text,
            "length": length,
            "level": level
        }
    
    # 커스텀 후처리 함수 정의
    def postprocess(response: str, metadata: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "response": response,
            "metadata": metadata,
            "response_length": len(response)
        }
    
    # 프롬프트 정의
    prompt = ChatPromptTemplate.from_template(
        "텍스트 복잡도: {level}"
        "다음 텍스트를 분석하세요: \n'{text}'"
    )

    # Output Parser 정의
    output_parser = StrOutputParser()

    # 체인 구성
    chain = (
        RunnableLambda(preprocess)
        | RunnablePassthrough.assign(
            response=lambda x: (prompt | llm | output_parser).invoke(x)
        )
        | RunnablePassthrough(lambda x: postprocess(x["response"], { "length": x["length"], "level": x["level"]})) # type: ignore
    )

    # 체인 실행
    input_text = "인공지능은 현대 사회의 핵심 기술입니다."
    result = chain.invoke({"input_text": input_text})

    logger.info(f"입력\n: {input_text}")
    logger.info(f"응답:\n{result}")
    logger.info("=" * 40)


def custom_chain_example_2():
    """
    예제 2: 상태 유지 체인
    """
    # llm 초기화
    llm = get_llm()
    
    # 상태 저장소
    class ChainState:
        def __init__(self):
            self.history: List[Dict[str, Any]] = []
            self.call_count = 0

        def add_history(self, inputs: Dict[str, Any], output: str):
            self.call_count += 1
            self.history.append({
                "call_number": self.call_count,
                "inputs": inputs,
                "output": output
            })

        def get_summary(self) -> str:
            if not self.history:
                return "히스토리가 없습니다"
            
            summary = f"총 {self.call_count}번 호출됨\n"
            for item in self.history[-3:]: # 최근 3개만
                summary += f"  - 호출 {item['call_number']}: {item['inputs'].get('query', 'N/A')}\n"
            return summary
        
    state = ChainState()

    # 상태를 포함한 처리
    def process_with_state(inputs: Dict[str, Any]) -> Dict[str, Any]:
        query = inputs["query"]

        context = state.get_summary()

        prompt = ChatPromptTemplate.from_template(
            """이전 대화 컨텍스트:\n{context}
            현재 질문:\n{query}

            답변하세요: 
            """
        )

        chain = prompt | llm | StrOutputParser()
        result = chain.invoke({
            "context": context,
            "query": query
        })

        state.add_history(inputs, result)

        return {
            "query": query,
            "result": result,
            "call_number": state.call_count,
            "history_size": len(state.history)
        }
    
    chain = RunnableLambda(process_with_state)

    queries = [
        "인공지능이란 무엇인가요?",
        "그것의 장점은?",
        "단점도 알려주세요"
    ]

    for query in queries:
        result = chain.invoke({"query": query})
        logger.info(f"질문:\n{query}")
        logger.info(f"응답:\n{result['result']}")


if __name__ == "__main__":
    # custom_chain_example_1()
    custom_chain_example_2()