from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate, FewShotPromptTemplate, ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.example_selectors import SemanticSimilarityExampleSelector, LengthBasedExampleSelector
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from pydantic import SecretStr

from common.llm import get_llm
from config.settings import settings
from utils.logger import setup_logger

logger = setup_logger(__name__)


def few_shot_prompts_example_1():
    """
    예제 1: 기본 Few-Shot 프롬프트
    """
    # llm 초기화
    llm = get_llm()

    # few_shot data
    examples = [
        {
            "input": "행복하다",
            "output": "긍정적"
        },
        {
            "input": "슬프다",
            "output": "부정적"
        },
        {
            "input": "그냥 그렇다",
            "output": "중립적"
        }
    ]

    example_template = """
입력: {input},
감정: {output}
"""

    example_prompt = PromptTemplate(
        input_variables=["input", "output"],
        template=example_template
    )

    # Few Shot 프롬프트 템플릿
    few_shot_prompt = FewShotPromptTemplate(
        examples=examples,
        example_prompt=example_prompt,
        prefix="다음 예시를 참고하여 텍스트의 감정을 분류하세요:",
        suffix="입력: {input}\n감정:",
        input_variables=["input"]
    )

    # 체인 구성
    chain = few_shot_prompt | llm | StrOutputParser()

    # 테스트
    inputs = [
        "정말 기쁘다!",
        "화가 난다",
        "평범한 하루인거같다"
    ]

    for input in inputs:
        result = chain.invoke({"input": input})
        logger.info(f"입력:\n{input}")
        logger.info(f"결과(감정):\n{result}")

    return chain

def few_shot_prompts_example_2():
    """
    예제 2: 코드 생성 Few Shot
    """
    llm = get_llm()

    examples = [
        {
            "description": "리스트의 모든 요소를 2배로 만들기",
            "code": "result = [x * 2 for x in numbers]"
        },
        {
            "description": "리스트에서 짝수만 필터링",
            "code": "result = [x for x in numbers if x % 2 == 0]"
        }
    ]

    example_template = """
    구현 요구 사항: {description},
    코드: {code}
    """
    
    example_prompt = PromptTemplate(
        input_variables=['description', 'code'],
        template=example_template
    )

    # Few Shot
    few_shot_prompt = FewShotPromptTemplate(
        examples=examples,
        example_prompt=example_prompt,
        prefix="다음 예시를 참고하여 Python 코드를 작성하세요:",
        suffix="구현 요구 사항: {description}\n코드:",
        input_variables=["description"]
    )
    

    # 체인 구성
    chain = few_shot_prompt | llm | StrOutputParser()

    inputs = [
        "리스트에서 최댓값 찾기",
        "리스트를 역순으로 정렬하기"
    ]

    for input in inputs:
        result = chain.invoke({"description": input})
        logger.info(f"구현 요구 사항:\n{input}")
        logger.info(f"응답:\n{result}")

def few_shot_prompts_example_3():
    """
    예제 3: 의미 유사도 기반 예시 선택  (챗봇 만들때 좋을 거 같음 또는 항상 비슷한 질문에 비슷한 답변을 요구하는 Rag 시스템을 만들때)
    """
    llm = get_llm()
    
    # 예시 데이터
    examples = [
        {
            "input": "파이썬으로 리스트를 정렬하는 방법은?",
            "output": "list.sort() 또는 sorted(list)를 사용하세요."
        },
        {
            "input": "딕셔너리에 키를 추가하는 방법은?",
            "output": "dict[key] = value 형태로 추가하세요."
        },
        {
            "input": "문자열을 대문자로 변환하는 방법은?",
            "output": "str.upper() 메서드를 사용하세요."
        },
        {
            "input": "리스트에서 중복을 제거하는 방법은?",
            "output": "set(list)를 사용하거나 리스트 컴프리헨션을 활용하세요."
        }
    ]

    example_template = """
    질문: {input}
    답변: {output}
    """
    
    example_prompt = PromptTemplate(
        input_variables=["input", "output"],
        template=example_template
    )

    example_selector = SemanticSimilarityExampleSelector.from_examples(
        examples,
        OpenAIEmbeddings(api_key=SecretStr(settings.OPENAI_API_KEY)),
        Chroma,
        k=2 # 가장 유사한 2개 선택
    )

    few_shot_prompt = FewShotPromptTemplate(
        example_selector=example_selector,
        example_prompt=example_prompt,
        prefix="다음 예시를 참고하여 답변하세요",
        suffix="질문: {input}\n답변:",
        input_variables=['input']
    )

    chain = few_shot_prompt | llm | StrOutputParser()

    inputs = [
        "리스트를 역순으로 만드는 방법은?",
        "딕셔너리의 모든 키를 가져오는 방법은?"
    ]

    for input in inputs:
        selected_example = example_selector.select_examples({"input": input })
        logger.info(f"질문:\n{input}")
        logger.info(f"선택된 예시:\n{selected_example}")

        result = chain.invoke({"input": input})
        logger.info(f"답변:\n{result}")


if __name__ == "__main__": 
    # few_shot_prompts_example_1()
    # few_shot_prompts_example_2()
    few_shot_prompts_example_3()


