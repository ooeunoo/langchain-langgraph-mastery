from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableBranch, RunnablePassthrough
from pydantic import SecretStr

from config.settings import settings
from utils.logger import setup_logger

logger = setup_logger(__name__)


def router_chain_example_1():
    """
    예제 1: 하나의 체인으로 라우팅하는 방식
    """
    llm = ChatOpenAI(
        model=settings.OPENAI_MODEL,
        api_key=SecretStr(settings.OPENAI_API_KEY),
        temperature=0
    )

    routing_prompt = ChatPromptTemplate.from_template(
        "다음 질문에 가장 적합한 전문가를 선택하세요. "
        "'tech', 'health', 'travel' 중 하나만 답변하세요:\n"
        "질문: {question}\n"
        "전문가:"
    )

    tech_expert_prompt = ChatPromptTemplate.from_template(
        "기술 전문가로서 다음 질문에 답변하세요:\n{question}"
    )

    health_expert_prompt = ChatPromptTemplate.from_template(
        "건강 전문가로서 다음 질문에 답변하세요:\n{question}"
    )

    travel_expert_prompt = ChatPromptTemplate.from_template(
        "여행 전문가로서 다음 질문에 답변하세요:\n{question}"
    )

    output_parser = StrOutputParser()

    # 전문가별 체인 생성 - 답변만 반환
    tech_chain = tech_expert_prompt | llm | output_parser
    health_chain = health_expert_prompt | llm | output_parser
    travel_chain = travel_expert_prompt | llm | output_parser
    default_chain = (
        ChatPromptTemplate.from_template("다음 질문에 답변하세요:\n{question}")
        | llm
        | output_parser
    )

    chain = (
        {
            "question": RunnablePassthrough(),
            "expert": routing_prompt | llm | output_parser
        }
        | RunnablePassthrough.assign(
            answer=RunnableBranch(
                (lambda x: "tech" in x["expert"].lower(), tech_chain),
                (lambda x: "health" in x["expert"].lower(), health_chain),
                (lambda x: "travel" in x["expert"].lower(), travel_chain),
                default_chain
            )
        )
    )

    # 실행
    question = "세계에서 가장 아름다운 여행지에 대해 알려주세요."
    result = chain.invoke({"question": question})
    
    logger.info(f"질문: {result['question']}")
    logger.info(f"선택된 전문가: {result['expert']}")
    logger.info(f"답변: {result['answer']}")

    return chain


if __name__ == "__main__":
    router_chain_example_1()