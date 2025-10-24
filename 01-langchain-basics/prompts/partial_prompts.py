from datetime import datetime
from functools import partial
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from pydantic import SecretStr

from common.llm import get_llm
from config.settings import settings
from utils.logger import setup_logger

logger = setup_logger(__name__)

def partial_prompts_example_1():
    """
    예제 1: 기본 Partial 프롬프트
    """
    llm = get_llm()

    template = PromptTemplate(
        input_variables=['role', 'topic', 'style'],
        template="당신은 {role}입니다. {topic}에 대해 {style} 스타일로 설명하세요."
    )

    partial_template = template.partial(role="AI 전문가")

    # 체인 구성
    chain = partial_template | llm | StrOutputParser()

    # 체인 실행
    value = {
        "topic": "머신러닝",
        "style": "초보자도 이해하기 쉽게"
    }
    result = chain.invoke(value)

    logger.info(f"주제:\n{value['topic']}")
    logger.info(f"스타일:\n{value['style']}")
    logger.info(f"응답:\n{result}")

    return chain


def partial_prompts_example_2():
    """
    예제 2: 동적 날짜 삽입
    """

    llm = get_llm()

    def get_current_date():
        return datetime.now().strftime('%Y년 %m월 %d일')

    def get_current_time():
        return datetime.now().strftime('%H:%M:%S')
    
    # 템플릿 생성
    template = PromptTemplate(
        input_variables=['date', 'time', 'topic'],
        template="오늘은 {date}, 현재 시간은 {time}입니다. {topic}에 대한 오늘의 뉴스나 트렌드를 알려주세요."
    )

    partial_template = template.partial(
        date=get_current_date,
        time=get_current_time
    )

    chain = partial_template | llm | StrOutputParser()

    value = {
        "topic": "AI 인공지능"
    }
    result = chain.invoke({'topic': "AI 인공지능"})

    logger.info(f"현재 날짜/시간:\n{get_current_date()}/{get_current_time()}")
    logger.info(f"주제:\n{value['topic']}")
    logger.info(f"응답:\n{result}")
    
    return chain

if __name__ == "__main__":
    partial_prompts_example_1()
    partial_prompts_example_2()
    