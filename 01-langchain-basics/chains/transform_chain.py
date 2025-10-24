"""
1. RunnableLambda
    - 커스텀 Python 함수를 체인에 통합
    - 데이터 변환, 정제, 추출 등

2. 전처리 패턴
    - 여러 전처리 단계를 순차적으로 적용
    - 텍스트 정제, 소문자 변환, 불용어 제거 등
"""


from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
import re
import json

from common.llm import get_llm
from config.settings import settings
from utils.logger import setup_logger

logger = setup_logger('transform_chain')


def transform_chain_example_1():
    """
    예제 1: 텍스트 정제 변환 (불필요한 공백, 특수문자 제거)
    """
    # llm 초기화
    llm = get_llm()

    # 텍스트 정제 함수
    def clean_text(text: str) -> str:
        # 불필요한 공백 제거
        text = re.sub(r'\s+', ' ', text).strip()
        # 특수문자 제거
        text = re.sub(r'[!@#$%^&*(),]', '', text)
        return text
    
    # 변환 체인 - RunnablePassthrough.assign()을 사용하여 기존 키 유지
    transform_chain = RunnablePassthrough.assign(
        cleaned_text=lambda x: clean_text(x["input_text"])
    )
    
    # 프롬프트 정의 - cleaned_text를 사용
    prompt = ChatPromptTemplate.from_template(
        "다음 문장을 요약하세요:\n'{cleaned_text}'"
    )

    # Output Parser 정의
    output_parser = StrOutputParser()

    # LCEL 체인 구성
    chain = (
        transform_chain
        | RunnablePassthrough.assign(
            summary=lambda x: (prompt | llm | output_parser).invoke(x)
        )
    )

    # 체인 실행
    input_text = "안녕하세요!!!   저는### LangChain을 사용하고 있습니다.   이@것은 정말$$ 멋진 라)()이브)러리입니다...@@@"
    result = chain.invoke({"input_text": input_text})
    logger.info(f"원본 텍스트:\n{input_text}")
    logger.info(f"정제된 텍스트:\n{result['cleaned_text']}")    
    logger.info(f"요약:\n{result['summary']}")

    logger.info("=" * 40)

    return chain

def transform_chain_example_2():
    """
    예제 2: 데이터 포맷 변환
    """ 
    # llm 초기화
    llm = get_llm()

    # CSV to Dict 변환 함수
    def csv_to_dict(csv_text: str) -> dict:
        lines = csv_text.strip().split('\n')
        headers = [h.strip() for h in lines[0].split(',')]
        values = [v.strip() for v in lines[1].split(',')]
        return dict(zip(headers, values)) 
    
    # Dict to JSON 변환 함수
    def dict_to_json(data_dict: dict) -> str:
        return json.dumps(data_dict, ensure_ascii=False, indent=2)
    
    # 변환 체인
    transform_chain = (
        RunnableLambda(lambda x: {"data_dict": csv_to_dict(x["csv_data"])})  # type: ignore
        | RunnablePassthrough.assign(
            json_data=lambda x: dict_to_json(x["data_dict"])
        )
    )
    
    # 프롬프트 정의
    prompt = ChatPromptTemplate.from_template(
        "다음 데이터를 분석하여 인사이트를 제공하세요.:\n'{json_data}'"
    )

    # Output Parser 정의
    output_parser = StrOutputParser()

    # LCEL 체인 구성
    chain = (
        transform_chain
        | RunnablePassthrough.assign(
            insights=lambda x: (prompt | llm | output_parser).invoke(x)
        )
    )

    # 체인 실행
    csv_data = "이름, 나이, 도시, 직업\n홍길동, 30, 서울, 개발자"
    result = chain.invoke({"csv_data": csv_data})
    logger.info(f"원본 CSV 데이터:\n{csv_data}")
    logger.info(f"변환된 JSON 데이터:\n{result['json_data']}")
    logger.info(f"분석 인사이트:\n{result['insights']}")
    logger.info("=" * 40)

    return chain


def transform_chain_example_3():
    """
    예제 3: 데이터 추출 변환
    """
    # llm 초기화
    llm = get_llm()

    # 이메일 추출 
    def extract_emails(text: str) -> list[str]:
        return re.findall(r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}', text)

    # URL 추출
    def extract_urls(text: str) -> list[str]:
        return re.findall(r'(https?://[^\s]+)', text)
    
    # 전화번호 
    def extract_phone_numbers(text: str) -> list[str]:
        return re.findall(r'\b\d{2,4}[-.\s]??\d{3,4}[-.\s]??\d{4}\b', text)

    # 변환 체인
    transform_chain = RunnableLambda(lambda x: {
        "text": x["input_text"], # type: ignore
        "emails": extract_emails(x["input_text"]), # type: ignore
        "urls": extract_urls(x["input_text"]), # type: ignore
        "phone_numbers": extract_phone_numbers(x["input_text"]) # type: ignore
    })

    # 프롬프트 정의
    prompt = ChatPromptTemplate.from_template(
        "다음 텍스트에서 추출된 이메일, URL, 전화번호를 분석하세요.\n"
        "이메일: {emails}\n"
        "URL: {urls}\n"
        "전화번호: {phone_numbers}\n"
    )

    # LCEL 체인 구성
    chain = (
        transform_chain
        | RunnablePassthrough.assign(
            analysis=lambda x: (prompt | llm | StrOutputParser()).invoke(x)
        )
    )

    # 체인 실행
    input_text = "연락처 정보:" \
    "이메일: example@email.com, test@example.com" \
    "웹사이트: https://www.example.com, http://test.com" \
    "전화번호: 010-1234-5678, 02-987-6543"

    result = chain.invoke({"input_text": input_text})
    logger.info(f"원본 텍스트:\n{input_text}")
    logger.info(f"추출된 이메일: {result['emails']}")
    logger.info(f"추출된 URL: {result['urls']}")
    logger.info(f"추출된 전화번호: {result['phone_numbers']}")
    logger.info(f"분석 결과:\n{result['analysis']}")

    logger.info("=" * 40)
    return chain

def transform_chain_example_4():
    """
    예제 4: 전처리 파이프라인
    """
    # llm 초기화
    llm = get_llm()

    # 전처리 함수 1 - 텍스트 정제
    def clean_text(text: str) -> str:
        text = re.sub(r'\s+', ' ', text).strip()
        text = re.sub(r'[!@#$%^&*(),]', '', text)
        return text

    # 전처리 함수 2 - 소문자 변환
    def to_lowercase(text: str) -> str:
        return text.lower()
    
    # 전처리 함수 3 - 숫자 제거
    def remove_numbers(text: str) -> str:
        return re.sub(r'\d+', '', text)
    
    
    # 전처리 함수 4 - 불용어 제거 
    def remove_stopwords(text: str) -> str:
        stopwords = ['은', '는', '이', '가', '을', '를', '의', '에', '에서']
        words = text.split()
        filtered_words = [word for word in words if word not in stopwords]
        return ' '.join(filtered_words)
    

    # 통계 계산
    def calculate_stats(text: str) -> dict:
        return {
            "length": len(text),
            "word_count": len(text.split()),
            "char_count": len(text.replace(' ', ''))
        }
    

    # LCEL 체인 구성
    chain = (
        RunnableLambda(lambda x: {"input_text": x["text"]}) # type: ignore
        | RunnablePassthrough.assign(step1=lambda x: clean_text(x["input_text"]))
        | RunnablePassthrough.assign(step2=lambda x: to_lowercase(x["step1"]))
        | RunnablePassthrough.assign(step3=lambda x: remove_numbers(x["step2"]))
        | RunnablePassthrough.assign(cleaned_text=lambda x: remove_stopwords(x["step3"]))
        | RunnablePassthrough.assign(
            stats=lambda x: calculate_stats(x["cleaned_text"])
        )
    )

    # 체인 실행
    text = "안녕하세요!!! 저는 LangChain을 사용하고 있습니다. 이것은 정말 멋진 라이브러리입니다. 오늘은 2024년 6월 15일입니다."
    result = chain.invoke({"text": text})
    logger.info(f"원본 텍스트:\n{text}")
    logger.info(f"정제된 텍스트:\n{result['cleaned_text']}")
    logger.info(f"텍스트 통계:\n{result['stats']}")     
    logger.info("=" * 40)
    return chain


if __name__ == "__main__":
    # transform_chain_example_1()
    # transform_chain_example_2()
    # transform_chain_example_3()
    transform_chain_example_4()