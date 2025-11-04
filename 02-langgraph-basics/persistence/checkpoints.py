
from typing import TypedDict, Annotated, Literal
from operator import add
from datetime import datetime

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph.message import add_messages

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langgraph.graph.state import CompiledStateGraph
from langchain_core.runnables import RunnableConfig
from common.llm import get_llm
from utils.logger import setup_logger

logger = setup_logger('checkpoints')

class ConversationState(TypedDict):
    messages: Annotated[list, add_messages]
    user_id: str
    user_name: str
    conversation_count: int
    last_updated: str


def initialize_conversation(state: ConversationState) -> ConversationState:
    count = state.get('conversation_count', 0)

    return ConversationState(conversation_count=count, last_updated=datetime.now().isoformat()) # type: ignore

def generate_response(state: ConversationState) -> ConversationState:
    llm = get_llm()
    
    messages = state['messages']
    user_name = state.get('user_name')

    system_prompt = f"""
당신은 친절한 AI 어시스턴트입니다.
사용자 이름: {user_name}

이전 대화 맥락을 기억하고 자연스럽게 대화하세요.
간결하고 도움이 되는 답변을 제공하세요.
"""

    full_messages = [
        SystemMessage(content=system_prompt)
    ] + messages

    response = llm.invoke(full_messages)

    return ConversationState(messages=[AIMessage(content=response.content)])  # type: ignore



def create_chatbot() -> CompiledStateGraph:
    builder = StateGraph(ConversationState)

    builder.add_node('init', initialize_conversation)
    builder.add_node('generate', generate_response)

    builder.add_edge(START, 'init')
    builder.add_edge('init', 'generate')
    builder.add_edge('generate', END)

    checkpointer = MemorySaver()

    return builder.compile(checkpointer=checkpointer)

def checkpoint_example_1():
    graph = create_chatbot()

    user_config_1 = RunnableConfig(
        configurable={
            "thread_id": "user_1"
        }
    )

    graph.invoke(
        {
            "messages": [HumanMessage(content="안녕하세요! 저는 홍길동입니다. 오늘은 Python 공부를 하고 있어요.")],
            "user_id": "홍길동",
            "user_name": "홍길동",
            "conversation_count": 0,
            "last_updated": ""
        },
        user_config_1
    )

    graph.invoke(
        {
            "messages": [HumanMessage(content="특히 데코레이터에 대해 궁금한데, 설명해 줄 수 있나요?")]
        },
        user_config_1
    )

    graph.invoke(
        {
            "messages": [HumanMessage(content="간단한 예제 코드 하나만 보여줄래요?")]
        },
        user_config_1
    )

    current_state = graph.get_state(user_config_1)

    for msg in current_state.values['messages']:
        if isinstance(msg, HumanMessage):
            logger.info(f" 사용자: {msg.content}")
        else: 
            logger.info(f" AI: {msg.content}")

if __name__ == "__main__":
    checkpoint_example_1()