"""
1. JsonOutputParser
    - LLM 응답을 JSON으로 파싱
    - 구조화된 데이터 추출
"""


from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser

from common.llm import get_llm
from utils.logger import setup_logger

logger = setup_logger(__name__)

def json_parser_example_1():
    """
    예제 1: 기본 JSON 파서
    """
    llm = get_llm()

    parser = JsonOutputParser()

    prompt = ChatPromptTemplate.from_template(
        "다음 국가에 대한 정보를 JSON 형식으로 제공해주세요:\n"
        "- continent: 국가가 있는 대륙\n"
        "- capital: 국가의 수도\n"
        "- popualation: 인구수 \n"
        "- famouse_for: 유명한 것\n\n"
        "{format_instructions}\n\n"
        "국가: {country}"
    )

    chain = prompt | llm | parser

    value = {
        "country": "이집트",
        "format_instructions": parser.get_format_instructions()
    }
    result = chain.invoke(value)

    logger.info(f"format instruction:\n{parser.get_format_instructions()}")
    logger.info(f"국가:\n{value['country']}")
    logger.info(f"응답:\n{result}")

    return chain



if __name__ == "__main__":
    json_parser_example_1()