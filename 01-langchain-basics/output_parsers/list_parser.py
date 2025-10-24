from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import CommaSeparatedListOutputParser

from common.llm import get_llm
from utils.logger import setup_logger

logger = setup_logger(__name__)


def list_parser_example_1():
    """
    예제 1: 기본 리스트 파싱
    """
    llm = get_llm()

    parser = CommaSeparatedListOutputParser()

    prompt = ChatPromptTemplate.from_template(
        "{topic}과 관련된 키워드 5개를 쉼표로 구분하여 나열해주세요.\n\n"
        "{format_instructions}"
    )

    chain = prompt | llm | parser

    value = {
        "topic": "인공지능",
        "format_instructions": parser.get_format_instructions()
    }
    result = chain.invoke(value)

    logger.info(f"주제:\n{value['topic']}")
    logger.info(f"키워드:\n{result}")


if __name__ == "__main__":
    list_parser_example_1()