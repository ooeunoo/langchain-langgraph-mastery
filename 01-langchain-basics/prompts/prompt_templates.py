from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain_core.output_parsers import StrOutputParser
from pydantic import SecretStr

from config.settings import settings
from utils.logger import setup_logger

logger = setup_logger(__name__)

def prompt_templates_example_1():
    """
    예제 1: 기본 프롬프트 템플릿
    """
    # llm 초기화 
    llm = ChatOpenAI(
        model=settings.OPENAI_MODEL,
        temperature=settings.OPENAI_TEMPERATURE,
        api_key=SecretStr(settings.OPENAI_API_KEY)
    )

    # 기본 템플릿
    template = PromptTemplate.from_template(
        "{topic}에 대해 설명해주세요."
    )

    # 체인 구성
    chain = template | llm | StrOutputParser()

    # 체인 실행
    topic = "Langchain"
    result = chain.invoke({"topic": topic})

    logger.info(f"입력:\n{topic}에 대해 설명해주세요.")
    logger.info(f"결과:\n{result}")

    return chain

def prompt_templates_example_2():
    """
    예제 2: 여러 변수를 포함한 템플릿
    """
    # llm 초기화 
    llm = ChatOpenAI(
        model=settings.OPENAI_MODEL,
        temperature=settings.OPENAI_TEMPERATURE,
        api_key=SecretStr(settings.OPENAI_API_KEY)
    )

    # 템플릿 
    template = PromptTemplate(
        input_variables=['role', 'topic', 'style'],
        template="""당신은 {role}입니다. {topic}에 대해 {style} 스타일로 설명해주세요."""
    )

    # 체인 구성
    chain = template | llm | StrOutputParser()

    # 체인 실행
    value = {
        "role": "AI 전문가",
        "topic": "머신런닝",
        "style": "초등학생도 이해하기 쉬운"        
    }
    result = chain.invoke(value)

    logger.info(f"역할:\n{value['role']}")
    logger.info(f"주제:\n{value['topic']}")
    logger.info(f"스타일:\n{value['style']}")
    logger.info(f"결과:\n{result}")

    return chain


def prompt_templates_example_3():
    """
    예제 3: ChatPromptTemplate - 시스템 메시지와 사용자 메시지 구분
    """
    # llm 초기화 
    llm = ChatOpenAI(
        model=settings.OPENAI_MODEL,
        temperature=settings.OPENAI_TEMPERATURE,
        api_key=SecretStr(settings.OPENAI_API_KEY)
    )

    # ChatPromptTemplate 
    template = ChatPromptTemplate.from_messages([
        ("system", "당신은 {domain} 전문가입니다. 항상 정확하고 명확하게 답변하세요."),
        ("human", "{question}")
    ])

    # 체인 구성
    chain = template | llm | StrOutputParser()

    # 체인 실행
    value = {
        "domain": "Python 프로그래밍",
        "question": "리스트 컴프리헨션이란 무엇인가요?"
    }
    result = chain.invoke(value)

    logger.info(f"도메인:\n{value['domain']}")
    logger.info(f"질문:\n{value['question']}")
    logger.info(f"결과:\n{result}")

    return chain

def prompt_templates_example_4():
    """
    예제 4: 메시지 템플릿 상세 제어 - System, Human 메시지 개별 제어
    """
    # llm 초기화 
    llm = ChatOpenAI(
        model=settings.OPENAI_MODEL,
        temperature=settings.OPENAI_TEMPERATURE,
        api_key=SecretStr(settings.OPENAI_API_KEY)
    )
    
    # 시스템 메시지 템플릿
    system_template = SystemMessagePromptTemplate.from_template(
        "당신은 {domain} 분아야의 전문가입니다.  {style} 정확하고 명확한 설명과 예시를 포함해주세요."
    )

    # 휴먼 메시지 템플릿
    human_template = HumanMessagePromptTemplate.from_template(
        "{question}"
    )

    # 프롬프트 템플릿
    chat_template = ChatPromptTemplate.from_messages([
        system_template,
        human_template
    ])

    # 체인 구성
    chain = chat_template | llm | StrOutputParser()

    # 체인 실행
    value = {
        "domain": "데이터 과학",
        "style": "초보자도 이해하기 쉽게",
        "question": "통계에서 평균과 중앙값의 차이는?"
    }
    result = chain.invoke(value)

    logger.info(f"도메인:\n{value['domain']}")
    logger.info(f"스타일:\n{value['style']}")
    logger.info(f"질문:\n{value['question']}")
    logger.info(f"결과:\n{result}")


if __name__ == "__main__":
    prompt_templates_example_1()
    prompt_templates_example_2()
    prompt_templates_example_3()
    prompt_templates_example_4()
