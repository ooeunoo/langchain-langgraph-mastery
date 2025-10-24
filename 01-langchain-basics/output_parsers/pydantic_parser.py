from typing import List, Optional
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser

from common.llm import get_llm
from utils.logger import setup_logger

logger = setup_logger(__name__)


def pydantic_parser_example_1():
    """
    예제 1: 기본 Pydantic 파싱
    """
    llm = get_llm()

    class Person(BaseModel):
        """사람 정보"""
        name: str = Field(description="이름")
        age: int = Field(description="나이")
        occupation: str = Field(description="직업")
        hobbies: List[str] = Field(description="취미 리스트")

    parser = PydanticOutputParser(pydantic_object=Person)

    prompt = ChatPromptTemplate.from_template(
        "다음 인물에 대한 정보를 제공해주세요.\n\n"
        "{format_instructions}\n\n"
        "인물: {person}"
    )

    chain = prompt | llm | parser


    value = {
        "person": "스티브 잡스",
        "format_instructions": parser.get_format_instructions()
    }

    result = chain.invoke(value)

    logger.info(f"인물:\n{value['person']}")
    logger.info(f"이름:\n{result.name}")
    logger.info(f"나이:\n{result.age}")
    logger.info(f"직업:\n{result.occupation}")
    logger.info(f"취미:\n{', '.join(result.hobbies)}")

    return chain


def pydantic_parser_example_2():
    """
    예제 2: Optional Pydantic 필드
    """
    llm = get_llm()

    class Restaurant(BaseModel):
        name: str = Field(description="레스토랑 이름")
        cuisine: str = Field(description="음식 종류")
        price_range: str = Field(description="가격대 ($, $$, $$$)")
        rating: float = Field(description="평점 (0-5)")
        address: Optional[str] = Field(default=None, description="주소")


    parser = PydanticOutputParser(pydantic_object=Restaurant)

    prompt = ChatPromptTemplate.from_template(
        "다음 레스토랑에 대한 정보를 제공해주세요.\n\n"
        "{format_instructions}\n\n"
        "레스토랑: {restaurant}"
    )

    chain = prompt | llm | parser

    value = {
        "restaurant": "강남의 유명한 이탈리안 레스토랑",
        "format_instructions": parser.get_format_instructions()
    }
    result = chain.invoke(value)

    logger.info(f"레스토랑:\n{value['restaurant']}")
    logger.info(f"이름: {result.name}")
    logger.info(f"음식: {result.cuisine}")
    logger.info(f"가격: {result.price_range}")
    logger.info(f"평점: {result.rating}/5")
    logger.info(f"주소: {result.address if result.address else '정보 없음'}")

    return chain

if __name__ == "__main__":
    # pydantic_parser_example_1()
    pydantic_parser_example_2()
