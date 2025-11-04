"""
1. LangGraph 기본 개념
    - State: 그래프 전체에서 공유되는 상태
    - Node: 실행 단위 (함수)
    - Edge: Node 간의 연결

2. StateGraph 구조
    - 상태 정의 (TypedDict)
    - 노드 추가
    - 엣지 연결
    - 컴파일

3. LangChain과의 차이
    - LangChain: 선형 체인 (A -> B -> C)
    - LangGraph: 그래프 구조 (조건부 분기, 루프 가능)

"""


from typing_extensions import Annotated, TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

from common.llm import get_llm
from utils.logger import setup_logger

logger = setup_logger(__name__)

class GraphState(TypedDict):
    messages: Annotated[list, add_messages]

class CounterState(TypedDict):
    count: int
    message: str


def chatbot_node(state: GraphState) -> GraphState:
    llm = get_llm()

    messages = state["messages"]

    response = llm.invoke(messages)

    return GraphState(messages=[response])

def greeting_node(state: GraphState) -> GraphState:
    greeting = AIMessage(content="안녕하세요! 무엇을 도와드릴까요?")

    return GraphState(messages=[greeting])

def counter_node(state: CounterState) -> CounterState:
    current_count = state["count"] 
    new_count = current_count + 1

    return CounterState(count=new_count, message=f"Count is now {new_count}")

# ======================
# 예제 
# ======================
def simple_graph_example_1():
    # StateGraph 생성 
    builder = StateGraph(GraphState)

    # Node 추가
    builder.add_node("chatbot", chatbot_node)

    # Edge 연결
    builder.add_edge(START, "chatbot") # 시작 -> chatbot
    builder.add_edge("chatbot", END) # chatbot -> 종료

    # 컴파일
    graph = builder.compile()

    # 실행
    result = graph.invoke(GraphState(messages=[HumanMessage(content="한국의 수도는 어디인가요?")]))

    for i, msg in enumerate(result["messages"], 1):
        role = msg.__class__.__name__
        logger.info(f"{i}. [{role}]: {msg.content}")

    return graph


def simple_graph_example_2():
    # StateGraph 생성
    builder = StateGraph(GraphState)

    # Node 추가
    builder.add_node('greeting', greeting_node)
    builder.add_node('chatbot', chatbot_node)

    # Edge 연결
    builder.add_edge(START, 'greeting')
    builder.add_edge('greeting', 'chatbot')
    builder.add_edge('chatbot', END)

    # 컴파일
    graph = builder.compile()

    # 실행
    result = graph.invoke(GraphState(messages=[HumanMessage(content="프랑스의 수도는 어디인가요?")]))

    for i, msg in enumerate(result["messages"], 1):
        role = msg.__class__.__name__
        logger.info(f"{i}. [{role}]: {msg.content}")

    return graph

def simple_graph_example_3():
    # StateGraph 생성
    builder = StateGraph(GraphState)

    # Node 추가
    builder.add_node('chatbot', chatbot_node)

    # Edge 연결
    builder.add_edge(START, 'chatbot')
    builder.add_edge('chatbot', END)

    # 컴파일
    graph = builder.compile()

    # 실행
    result = graph.invoke(GraphState(messages=[
        SystemMessage(content="당신은 친절한 여행 가이드입니다."),
        HumanMessage(content="도쿄에서 꼭 가봐야 할 곳 3곳을 추천해주세요.")
    ]))

    for i, msg in enumerate(result['messages'], 1):
        role = msg.__class__.__name__
        logger.info(f"{i}. [{role}]: {msg.content}")
    
    return graph


async def simple_graph_example_4():
    builder = StateGraph(GraphState)

    builder.add_node("greeting", greeting_node)
    builder.add_node("chatbot", chatbot_node)

    builder.add_edge(START, "greeting")
    builder.add_edge("greeting", "chatbot")
    builder.add_edge("chatbot", END)
    
    graph = builder.compile()

    # 스트리밍 실행
    async for event in graph.astream_events(
        {"messages": [HumanMessage(content="AI란 무엇인가요?")]},
        version="v2"
    ):
        if event["event"] == "on_chat_model_stream":
            content = event["data"]["chunk"].content # type: ignore
            if content:
                print(content, end="", flush=True)
    
    return graph

if __name__ == "__main__":
    import asyncio

    # simple_graph_example_1()
    # simple_graph_example_2()
    # simple_graph_example_3()
    asyncio.run(simple_graph_example_4())
