
"""
1. RunnablePassthrough
    - 입력을 그대로 다음 단계로 전달하는 역할을 합니다.

2. RunnablePassthrough.assign()
    - 입력 데이터를 기반으로 새로운 변수를 생성하고 할당하는 기능을 제공합니다.
    - 람다 함수를 사용하여 중간 결과를 계산하고, 이를 후속 단계에서 사용할 수 있도록 합니다.

3. RunnableParallel
    - 여러 작업을 병렬로 실행할 수 있도록 지원합니다.
    - 각 작업은 독립적으로 실행되며, 결과는 딕셔너리 형태로 반환됩니다.
"""

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableParallel

from common.llm import get_llm
from config import settings
from utils.logger import setup_logger

logger = setup_logger("sequential_chain")

def sequential_chain_example_1():
    """
    예제 1: 순차 체인 기본 구조
    """

    # llm 초기화
    llm = get_llm()

    # 프롬프트 정의
    prompt1 = ChatPromptTemplate.from_template(
        "{subject}에 대한 창의적인 주제를 생성하세요. 주세만 반환하고 다른 내용은 포함하지 마세요."
    )

    prompt2 = ChatPromptTemplate.from_template(
        "다음 주제에 관해 짧은 시를 작성하세요.\n주제: {topic}"
    )

    # Output Parser 정의
    output_parser = StrOutputParser()

    # LCEL 순차 체인 구성
    chain = (
        prompt1 
        | llm 
        | output_parser
        | {"topic": RunnablePassthrough()}  
        | prompt2 
        | llm 
        | output_parser
    )

    # 체인 실행
    subject = "나비"
    result = chain.invoke({"subject": subject})

    logger.info(f"입력 주제:\n{subject}")
    logger.info(f"시:\n{result}")
    
    logger.info("=" * 40)

    return chain

def sequential_chain_example_2():
    """
    예제 2: 중간 결과를 변수로 포함하는 순차 체인
    """
    # llm 초기화
    llm = get_llm()

    # 프롬프트 정의
    prompt1 = ChatPromptTemplate.from_template(
        "{subject}에 대한 창의적인 주제를 생성하세요. 주세만 반환하고 다른 내용은 포함하지 마세요."
    )

    prompt2 = ChatPromptTemplate.from_template(
        "다음 주제에 관해 짧은 시를 작성하세요.\n주제: {topic}"
    )

    # Output Parser 정의
    output_parser = StrOutputParser()

    # LCEL 순차 체인 구성
    chain = (
        { "subject": RunnablePassthrough()}
        | RunnablePassthrough.assign(
            topic=lambda x: (prompt1 | llm | output_parser).invoke(x)
        ) 
        | RunnablePassthrough.assign(
            poem=lambda x: (prompt2 | llm | output_parser).invoke({"topic": x["topic"]})
        )        
    )

    # 체인 실행
    subject = "비행기"
    result = chain.invoke({"subject": subject})

    logger.info(f"입력 주제:\n{subject}")
    logger.info(f"생성된 시 주제:\n{result['topic']}")
    logger.info(f"생성된 시:\n{result['poem']}")
    
    logger.info("=" * 40)

    return chain

def sequential_chain_example_3():
    """
    예제 3: 다단계 분석 
    """
    # llm 초기화
    llm = get_llm()

    # 프롬프트 정의
    prompt1 = ChatPromptTemplate.from_template(
        "{text}의 주요 아이디어를 요약하세요."
    )

    prompt2 = ChatPromptTemplate.from_template(
        "다음 요약을 기반으로 한 가지 토론 질문을 만드세요:\
        \n요약: {summary}"
    )

    prompt3 = ChatPromptTemplate.from_template(
        "다음 토론 질문에 대한 답변을 작성하세요:\n질문: {question}"
    )

    # Output Parser 정의
    output_parser = StrOutputParser()

    # LCEL 순차 체인 구성
    chain = (
        { "text": RunnablePassthrough()}
        | RunnablePassthrough.assign(
            summary=lambda x: (prompt1 | llm | output_parser).invoke(x)
        ) 
        | RunnablePassthrough.assign(
            question=lambda x: (prompt2 | llm | output_parser).invoke({"summary": x["summary"]})
        )  
        | RunnablePassthrough.assign(
            answer=lambda x: (prompt3 | llm | output_parser).invoke({"question": x["question"]})
        )        
    )

    # 체인 실행
    text = ("인공지능은 현대 사회에서 중요한 역할을 하고 있습니다. \
            다양한 산업 분야에서 혁신을 이끌고 있으며, 우리의 일상 생활에도 큰 영향을 미치고 있습니다. \
            앞으로 인공지능 기술의 발전은 더욱 가속화될 것으로 예상됩니다.")
    
    result = chain.invoke({"text": text})   
    
    logger.info(f"입력 텍스트:\n{text}")
    logger.info(f"생성된 요약:\n{result['summary']}")
    logger.info(f"생성된 토론 질문:\n{result['question']}")
    logger.info(f"생성된 답변:\n{result['answer']}")

    logger.info("=" * 40)
    return chain

def sequential_chain_example_4():
    """
    예제 4: 병렬 실행 체인
    """
    # llm 초기화
    llm = get_llm()

    # 프롬프트 정의
    prompt1 = ChatPromptTemplate.from_template(
        "{text}의 감정을 분석하세요. 부가적인 설명없이 답만 주세요."
    )
    prompt2 = ChatPromptTemplate.from_template(
        "{text}의 요약을 작성하세요. 부가적인 설명없이 답만 주세요."
    )
    prompt3 = ChatPromptTemplate.from_template(
        "{text}의 키워드를 추출하세요. 부가적인 설명없이 답만 주세요."
    )

    # Output Parser 정의
    output_parser = StrOutputParser()

    # LCEL 병렬 체인 구성
    chain = (
        { "text": RunnablePassthrough()}
        | RunnableParallel(
            sentiment=prompt1 | llm | output_parser,
            sumamry=prompt2 | llm | output_parser,
            keywords=prompt3 | llm | output_parser,
        )
    )

    # 체인 실행
    text = ("LangChain은 언어 모델을 활용한 애플리케이션 개발을 \
            쉽게 만들어주는 프레임워크입니다. 다양한 도구와 통합이 가능하며, \
            복잡한 작업도 간단하게 처리할 수 있습니다.")
    
    result = chain.invoke({"text": text})

    logger.info(f"입력 텍스트:\n{text}")
    logger.info(f"감정 분석:\n{result['sentiment']}")
    logger.info(f"요약:\n{result['sumamry']}")
    logger.info(f"키워드:\n{result['keywords']}")
    logger.info("=" * 40)

    return chain

def sequential_chain_example_5():
    """
    예제 5: 중첩된 순차 및 병렬 체인
    """
    # llm 초기화
    llm = get_llm()
    # 프롬프트 정의
    prompt = ChatPromptTemplate.from_template(
        "{continent}대륙에 있는 나라 하나를 선택해주세요. 부가적인 설명없이 나라 이름만 답해주세요."
    )   

    analysis_prompt = {
        "capital": ChatPromptTemplate.from_template(
            "{country}의 수도는 어디인가요?. 부가적인 설명없이 답만 주세요"
        ),        
        "population": ChatPromptTemplate.from_template(
            "{country}의 인구는 몇 명까요?. 부가적인 설명없이 답만 주세요"       
        ),
        "language": ChatPromptTemplate.from_template(
            "{country}의 공식 언어는 무엇인가요?. 부가적인 설명없이 답만 주세요"   
        ),
    }

    # Output Parser 정의
    output_parser = StrOutputParser()

    # LCEL 중첩 체인 구성
    chain = (
        { "continent": RunnablePassthrough()}
        | RunnablePassthrough.assign(
            country=lambda x: (prompt | llm | output_parser).invoke(x)
        ) 
        | RunnablePassthrough.assign(
            analysis=lambda x: RunnableParallel(
                capital=analysis_prompt["capital"] | llm | output_parser,
                population=analysis_prompt["population"] | llm | output_parser,
                language=analysis_prompt["language"] | llm | output_parser,
            ).invoke({"country": x["country"]})
        )        
    )

    # 체인 실행
    continent = "남아프리카"
    result = chain.invoke({"continent": continent})

    logger.info(f"입력 대륙:\n{continent}")
    logger.info(f"선택된 나라:\n{result['country']}")
    logger.info(f"수도:\n{result['analysis']['capital']}")
    logger.info(f"인구:\n{result['analysis']['population']}")
    logger.info(f"공식 언어:\n{result['analysis']['language']}")
    logger.info("=" * 40)
    return chain




if __name__ == "__main__":
    sequential_chain_example_1()
    sequential_chain_example_2()
    sequential_chain_example_3()
    sequential_chain_example_4()
    sequential_chain_example_5()

