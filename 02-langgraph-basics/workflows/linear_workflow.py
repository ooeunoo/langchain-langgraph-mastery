from typing import TypedDict, Annotated
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages, BaseMessage
from langchain.messages import HumanMessage, SystemMessage, AIMessage
from langgraph.graph.state import CompiledStateGraph
from common.llm import get_llm
from langchain_core.output_parsers import CommaSeparatedListOutputParser, StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, START, END

from utils.logger import setup_logger


logger = setup_logger("linear_workflow")

class GraphState(TypedDict):
    input_document: str
    key_points: str
    summary: str
    translation: str
    final_report: str
    messages: Annotated[BaseMessage, add_messages]

def extract_key_points(state: GraphState) -> GraphState:
    """
    Step 1: 핵심 포인트 추출 
    """
    llm = get_llm()
    parser = CommaSeparatedListOutputParser()
    
    # 메시지 리스트로 구성
    messages = [
        SystemMessage(
            content=f"""당신은 문서 분석 전문가입니다.
핵심 포인트를 3-5개 추출해주세요.
{parser.get_format_instructions()}"""
        ),
        HumanMessage(
            content=f"다음 문서를 분석해주세요:\n\n{state['input_document']}"
        )
    ]
    
    prompt = ChatPromptTemplate.from_messages(messages)    
    chain = prompt | llm | parser
    key_points_list = chain.invoke({})    
    key_points = "\n".join(f"- {point.strip()}" for point in key_points_list)
    
    return GraphState(
        key_points=key_points, 
        messages=state.get("messages", []) + messages + [AIMessage(content=key_points)]
    ) # type: ignore

def summarize_content(state: GraphState) -> GraphState:
    """
    Step2: 문서 요약 생성
    """
    llm = get_llm()
    parser = StrOutputParser()
    
    # 메시지 리스트로 구성
    messages = [
        SystemMessage(
            content=f"""당신은 문서 요약 전문가입니다. 사용자가 제공하는 문서를 요약해주세요"""
        ),
        HumanMessage(
            content=f"다음 문서를 요약해주세요:\n\n{state['input_document']}" # type: ignore
        )
    ]
    
    prompt = ChatPromptTemplate.from_messages(messages)    
    chain = prompt | llm | parser
    response = chain.invoke({})    
    
    return GraphState(
        summary=response,
        messages=state.get("messages", []) + messages + [AIMessage(content=response)]
    ) # type: ignore


def translate_to_english(state: GraphState) -> GraphState:
    """
    Step3: 문서 번역
    """
    llm = get_llm()
    parser = StrOutputParser()
    
    # 메시지 리스트로 구성
    messages = [
        SystemMessage(
            content=f"""당신은 문서 번역 전문가입니다. 사용자가 제공하는 문서를 번역해주세요"""
        ),
        HumanMessage(
            content=f"다음 문서를 번역해주세요:\n\n{state['input_document']}" # type: ignore
        )
    ]
    
    prompt = ChatPromptTemplate.from_messages(messages)    
    chain = prompt | llm | parser
    response = chain.invoke({})    
    
    return GraphState(
        translation=response, 
        messages=state.get("messages", []) + messages + [AIMessage(content=response)]
    ) # type: ignore


def general_final_report(state: GraphState) -> GraphState:
    """
    Step4: 문서 리포트 생성
    """
    llm = get_llm()
    parser = StrOutputParser()
    
    # 메시지 리스트로 구성
    messages = [
        SystemMessage(
            content=f"""당신은 종합 리포트 전문가입니다. 사용자가 제공하는 정보를 바탕으로 전문적인 리포트를 작성해주세요"""
        ),
        HumanMessage(
            content="다음 정보들을 종합하여 전문적인 리포트를 작성해주세요:\n" \
            "[핵심포인트]" \
            "{key_point}"

            "[한글 요약]"
            "{summary}"

            "[영문 번역]"
            "{translation}"
        )
    ]
    
    prompt = ChatPromptTemplate.from_messages(messages)    
    chain = prompt | llm | parser
    response = chain.invoke({
        "key_points": state.get('key_points'),
        "summary": state.get("summary"),
        "translation": state.get('translation')
    })    
    
    return GraphState(
        summary=response,
        messages=state.get("messages", []) + messages + [AIMessage(content=response)]
    ) # type: ignore


def create_linear_graph() -> CompiledStateGraph:
    builder = StateGraph(GraphState)

    builder.add_node('key_points', extract_key_points)
    builder.add_node('summary', summarize_content)
    builder.add_node('translation', translate_to_english)
    builder.add_node('report', general_final_report)


    builder.add_edge(START, 'key_points')
    builder.add_edge('key_points', 'summary')
    builder.add_edge('summary', 'translation')
    builder.add_edge('translation', 'report')
    builder.add_edge('report', END)

    return builder.compile()


def linear_workflow_example():
    graph = create_linear_graph()
    document = """
LangGraph는 LangChain 팀이 개발한 상태 기반 워크플로우 엔진입니다.
복잡한 AI 애플리케이션을 그래프 구조로 모델링할 수 있습니다.

주요 특징:
1. 상태 관리: TypedDict 기반의 명확한 상태 정의
2. 조건부 분기: 동적 라우팅 지원
3. Human-in-the-loop: 승인 프로세스 구현 가능
4. 병렬 처리: 여러 노드 동시 실행
5. 체크포인트: 상태 저장 및 복원

LangGraph를 사용하면 Multi-Agent 시스템, 복잡한 워크플로우, 
대화형 AI 등을 쉽게 구현할 수 있습니다.
"""

    state= GraphState(
        input_document=document,
        key_points="",
        summary="",
        translation="",
        final_report="",
        messages=[] # type: ignore
    )

    result = graph.invoke(state)

    for i, msg in enumerate(result['messages'], 1):
        role = msg.type
        content = msg.content

        logger.info(f"[{role}]: {content}")

if __name__ == "__main__":
    linear_workflow_example()