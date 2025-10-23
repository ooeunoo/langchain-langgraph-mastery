"""
1. LCEL (|) 연산자로 체인 구성
    - prompt | llm | output_parser

2. invoke() vs stream()
    - invoke(): 전체 응답을 한 번에 반환
    - stream(): 응답을 청크 단위로 스트리밍

3. ChatPromptTemplate
    - from_template(): 단일 프롬프트 템플릿 생성
    - from_messages(): 시스템 및 유저 메시지를 포함한 다중 메시지 템플릿 생성

4. OutputParser
    - StrOutputParser: 문자열 형태의 출력을 처리
    - 다양한 OutputParser를 사용하여 출력 형식 지정 가능
"""

from langchain_openai import ChatOpenAI
from pydantic import SecretStr
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from config import settings
from utils.logger import setup_logger

logger = setup_logger("simple_chain")


def simple_chain_example_1():
    """
    예제 1: 기본 구조의 체인
    """

    # llm 초기화
    llm = ChatOpenAI(
        model=settings.OPENAI_MODEL,
        api_key=SecretStr(settings.OPENAI_API_KEY),
        temperature=0
    )

    # 프롬프트 정의
    prompt = ChatPromptTemplate.from_template(
        "{country}의 수도는 어디인가요?"
    ) 

    # Output Parser 정의
    output_parser = StrOutputParser()

    # LCEL 체인 구성
    chain = prompt | llm | output_parser

    # 체인 실행
    country = "대한민국"
    result = chain.invoke({"country": country})

    logger.info(f"질문:\n{country}의 수도는 어디인가요?")
    logger.info(f"응답:\n{result}")

    logger.info("=" * 40)

    return chain

def simple_chain_example_2():
    """
    예제 2: 여러 입력 변수를 사용하는 체인
    """
    # llm 초기화
    llm = ChatOpenAI(
        model=settings.OPENAI_MODEL,
        api_key=SecretStr(settings.OPENAI_API_KEY),
        temperature=0
    )

    # 프롬프트 정의
    prompt = ChatPromptTemplate.from_template(
        "{country1}과 {country2}의 수도는 각각 어디인가요?"
    )

    # Output Parser 정의
    output_parser = StrOutputParser()

    # LCEL 체인 구성
    chain = prompt | llm | output_parser

    # 체인 실행
    country1 = "대한민국"
    country2 = "일본"
    result = chain.invoke({"country1": country1, "country2": country2})

    logger.info(f"질문:\n{country1}과 {country2}의 수도는 각각 어디인가요?")
    logger.info(f"응답:\n{result}")
    logger.info("=" * 40)

    return chain

def simple_chain_example_3():
    """
    예제 3: System Message를 활용한 체인
    """

    # llm 초기화
    llm = ChatOpenAI(
        model=settings.OPENAI_MODEL,
        api_key=SecretStr(settings.OPENAI_API_KEY),
        temperature=0
    )

    # 시스템 프롬프트, 유저 프롬프트 정의
    prompt = ChatPromptTemplate.from_messages([
        {"role": "system", "content": "당신은 {domain}에 대한 전문적인 지식이 있는 어시스턴트입니다."},
        {"role": "user", "content": "{country}의 수도는 어디이고 가장 유명한 랜드마크는 무엇인가요?"}
    ])

    # Output Parser 정의
    output_parser = StrOutputParser()

    # LCEL 체인 구성
    chain = prompt | llm | output_parser

    # 체인 실행 - 첫 번째 호출
    domain = "지리학"
    country = "프랑스"
    result = chain.invoke({"domain": domain, "country": country})

    logger.info(f"질문:\n{country}의 수도는 어디이고 가장 유명한 랜드마크는 무엇인가요?")
    logger.info(f"응답:\n{result}")
    logger.info("=" * 40)

    return chain

def simple_chain_example_4():
    """
    예제 4: 스트리밍 출력
    """

    # llm 초기화 
    llm = ChatOpenAI(
        model=settings.OPENAI_MODEL,
        api_key=SecretStr(settings.OPENAI_API_KEY),
        temperature=0,
    )

    # 프롬프트 정의
    prompt = ChatPromptTemplate.from_template(
        "{country}에 대해 다방면으로 설명해주세요?"
    )

    # Output Parser 정의
    output_parser = StrOutputParser()

    # LCEL 체인 구성
    chain = prompt | llm | output_parser

    # 체인 실행
    country = "이집트"
    logger.info(f"질문:\n{country}에 대해 다방면으로 설명해주세요?")
    for chunk in chain.stream({"country": country}):
        print(chunk, end="", flush=True)

    logger.info("\n" + "=" * 40)

    return chain

if __name__ == "__main__":
    simple_chain_example_1()
    simple_chain_example_2()
    simple_chain_example_3()
    simple_chain_example_4()


