"""
1. 라우터 체인 (Router Chain)
    - 여러 체인 중 하나로 라우팅하는 체인 구성 예제
    RunnableBranch(
        (조건1, 체인1),  # 1번째 확인
        (조건2, 체인2),  # 2번째 확인
        (조건3, 체인3),  # 3번째 확인
        기본_체인        # 모든 조건이 False일 때만 실행
    )

2. RunnableLambda
    - 간단한 람다 함수를 체인에 통합하는 방법 예제
    RunnableLambda(
        lambda x: "간단한 응답 생성"
    )
"""
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableBranch, RunnablePassthrough, RunnableLambda 
from pydantic import SecretStr

from config.settings import settings
from utils.logger import setup_logger

logger = setup_logger(__name__)

def router_chain_example_1():
    """
    예제 1: 입력값에 따른 라우팅 체인
    """
    # llm 초기화
    llm = ChatOpenAI(
        model=settings.OPENAI_MODEL,
        api_key=SecretStr(settings.OPENAI_API_KEY),
        temperature=0
    )

    # 라우팅 프롬프트 정의
    korean_prompt = ChatPromptTemplate.from_template(
        "다음 질문에 한국어로 답변하세요: \n질문: {question}"
    )

    english_prompt = ChatPromptTemplate.from_template(
        "Answer the following question in English: \nQuestion: {question}"
    )

    default_prompt = ChatPromptTemplate.from_template(
        "답변하세요: \n질문: {question}"
    )

    # Output Parser 정의
    output_parser = StrOutputParser()

    # 라우터 체인 구성
    korean_chain = korean_prompt | llm | output_parser
    english_chain = english_prompt | llm | output_parser
    default_chain = default_prompt | llm | output_parser

    router_chain = RunnableBranch(
        (lambda x: x["language"] == "korean", korean_chain),  # 한국어 질문일 때 # type: ignore
        (lambda x: x["language"] == "english", english_chain), # 영어 질문일 때 # type: ignore
        default_chain  # 그 외의 경우
    )

    queries = [
        { "language": "korean", "query": "한국의 수도는 어디인가요?"},
        { "language": "english", "query": "What is the capital of France?"},
        { "language": "spanish", "query": "¿Cuál es la capital de España?"}
    ]

    for query in queries:
        result = router_chain.invoke({"question": query["query"], "language": query["language"]})
        logger.info(f"질문 언어: {query['language']}")
        logger.info(f"질문:\n{query['query']}")
        logger.info(f"응답:\n{result}")
        logger.info("=" * 40)

    return router_chain


def router_chain_example_2():
    """
    예제 1: LLM의 응답에 따른 라우팅 체인
    """
    llm = ChatOpenAI(
        model=settings.OPENAI_MODEL,
        api_key=SecretStr(settings.OPENAI_API_KEY),
        temperature=0
    )

    # 라우팅 프롬프트 정의
    routing_prompt = ChatPromptTemplate.from_template(
        "다음 질문에 가장 적합한 전문가를 선택하세요. 관련 질문에 전문가가 없다면 'other'" \
        "'tech', 'health', 'finance', 'other' 중 하나만 답변하세요.\n" \
        "질문: {question}\n" \
        "전문가:"
    )

    # 각 도메인별 프롬프트 정의
    tech_prompt = ChatPromptTemplate.from_template(
        "당신은 기술 전문가입니다. 다음의 기술 관련 질문에 답변하세요:\n{question}"
    )

    health_prompt = ChatPromptTemplate.from_template(
        "당신은 건강 전문가입니다. 다음의 건강 관련 질문에 답변하세요:\n{question}"
    )

    finance_prompt = ChatPromptTemplate.from_template(
        "당신은 금융 전문가입니다. 다음의 금융 관련 질문에 답변하세요:\n{question}"
    )



    # Output Parser 정의
    output_parser = StrOutputParser()

    # 라우터 체인 구성
    chain = (
        {
            "question": RunnablePassthrough(),
            "expert": routing_prompt | llm | output_parser
        }
        | RunnablePassthrough.assign(
            answer = RunnableBranch(
                (lambda x: x["expert"].strip().lower() == "tech", tech_prompt | llm | output_parser),   # type: ignore
                (lambda x: x["expert"].strip().lower() == "health", health_prompt | llm | output_parser), # type: ignore
                (lambda x: x["expert"].strip().lower() == "finance", finance_prompt | llm | output_parser), # type: ignore
                RunnableLambda(lambda x: "죄송합니다. 해당 분야의 전문가가 없습니다.")            
            )
        )
    )

    # 체인 실행
    question = "요즘 가장 유명한 노래를 알려주세요."
    result = chain.invoke({"question": question})
    logger.info(f"질문:\n{question}")
    logger.info(f"선택된 전문가:\n{result['expert']}")
    logger.info(f"응답:\n{result['answer']}")
    logger.info("=" * 40)

    return chain

def router_chain_example_3():
    """
    예제 3: 감정 기반 라우터 체인
    """
    # llm 초기화
    llm = ChatOpenAI(
        model=settings.OPENAI_MODEL,
        api_key=SecretStr(settings.OPENAI_API_KEY),
        temperature=0
    )

    # 감정 분석 프롬프트 정의
    sentiment_prompt = ChatPromptTemplate.from_template(
        "다음 문장의 감정을 분석하세요. 긍정적이면 'positive', 부정적이면 'negative', 중립적이면 'neutral'을 답변하세요.\n" 
        "문장: {text}\n"
        "감정:"
    )

    # 각 감정별 프롬프트 정의
    positive_prompt = ChatPromptTemplate.from_template(
        "당신은 긍정적인 사람입니다. 다음 문장에 긍정적으로 답변하세요:\n{text}"
    )

    negative_prompt = ChatPromptTemplate.from_template(
        "당신은 부정적인 사람입니다. 다음 문장에 부정적으로 답변하세요:\n{text}"    
    )

    neutral_prompt = ChatPromptTemplate.from_template(
        "당신은 중립적인 사람입니다. 다음 문장에 중립적으로 답변하세요:\n{text}"    
    )

    # Output Parser 정의
    output_parser = StrOutputParser()

    # 라우트
    def sentiment_chain(text: str):
        sentiment = (sentiment_prompt | llm | output_parser).invoke({"text": text}).strip().lower()
        return sentiment
    
    # 라우터 체인 구성
    chain = RunnableBranch(
        (lambda x: sentiment_chain(x["text"]) == "positive", positive_prompt | llm | output_parser), # type: ignore
        (lambda x: sentiment_chain(x["text"]) == "negative", negative_prompt | llm | output_parser), # type: ignore
        (lambda x: sentiment_chain(x["text"]) == "neutral", neutral_prompt | llm | output_parser),    # type: ignore
        RunnableLambda(lambda x: "죄송합니다. 감정을 분석할 수 없습니다.")
    )

    # 체인 실행
    texts = [
        "오늘은 정말 멋진 날이에요! 모든 일이 잘 풀릴 것 같아요.",
        "모든 것이 엉망이에요. 아무것도 제대로 되지 않아요.",
        "그냥 그런 하루였어요. 특별한 일도 없고 지루했어요."    
    ]

    for text in texts:
        result = chain.invoke({"text": text})
        logger.info(f"문장:\n{text}")
        logger.info(f"응답:\n{result}")
        logger.info("=" * 40)


if __name__ == "__main__":
    # router_chain_example_1()
    # router_chain_example_2()
    router_chain_example_3()