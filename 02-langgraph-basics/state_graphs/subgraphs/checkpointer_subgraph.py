from typing import TypedDict, Annotated, Literal, Optional
from operator import add
from pydantic import BaseModel, Field

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph.message import add_messages
from langgraph.graph.state import CompiledStateGraph

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.runnables import RunnableConfig
from common.llm import get_llm
from utils.logger import setup_logger


logger = setup_logger('checkpointer_subgraph')

SentimentType = Literal['positive', 'neutral', 'negative', 'frustrated']

# model
class Sentiment(BaseModel):
    sentiment: SentimentType
    confidence: float = Field(ge=0.0, le=1.0)
    reason: str

# state
class ChatSubState(TypedDict):
    messages: Annotated[list, add_messages]
    user_name: str

class AnalysisSubState(TypedDict):
    last_message: str
    sentiment: Optional[Sentiment]
    analysis_count: int

class MainState(TypedDict):
    user_id: str
    user_name: str
    session_id: str
    current_message: str
    messages: Annotated[list, add_messages]
    sentiment: Optional[Sentiment]
    total_messages: int


def create_chat_subgraph() -> CompiledStateGraph:
    def generate_response(state: ChatSubState) -> ChatSubState:
        llm = get_llm()

        messages = state['messages']
        user_name = state['user_name']

        system_prompt = SystemMessage(content=f"""
당신은 친절한 AI어시스턴트입니다.
사용자 이름: {user_name}

간결하고 자연스럽게 대화하세요.
""")

        messages = [system_prompt] + messages

        response = llm.invoke(messages)

        return ChatSubState(messages=[AIMessage(response.content)]) # type: ignore
    
    builder = StateGraph(ChatSubState)

    builder.add_node('generate', generate_response)

    builder.add_edge(START, 'generate')
    builder.add_edge('generate', END)

    checkpointer= MemorySaver()

    return builder.compile(checkpointer=checkpointer)

def create_analysis_subgraph() -> CompiledStateGraph:
    def analyze_sentiment(state: AnalysisSubState) -> AnalysisSubState:
        llm = get_llm()

        message = state['last_message']
        count = state.get('analysis_count', 0)

        parser = PydanticOutputParser(pydantic_object=Sentiment)

        prompt = ChatPromptTemplate.from_template(
            """메시지의 감정을 분석하세요.

            메시지: {message}

            {format_instructions}

            감정 분류:
            - positive: 긍정적, 만족
            - neutral: 중립적
            - negative: 부정적, 불만
            - frustrated: 좌절, 화남
"""
        )

        chain = prompt | llm | parser
        print(parser.get_format_instructions())
        sentiment = chain.invoke({
            "message": message,
            "format_instructions": parser.get_format_instructions()
        })

        return AnalysisSubState(sentiment=sentiment, analysis_count=count + 1) # type: ignore
    
    builder = StateGraph(AnalysisSubState)

    builder.add_node('analyze', analyze_sentiment)

    builder.add_edge(START, 'analyze')
    builder.add_edge('analyze', END)

    checkpointer = MemorySaver()
    return builder.compile(checkpointer=checkpointer)

def create_main_graph():
    chat_subgraph = create_chat_subgraph()
    analysis_subgraph = create_analysis_subgraph()

    def process_message(state: MainState) -> MainState:
        user_id = state['user_id']
        user_name = state['user_name']
        session_id = state['session_id']
        current_msg = state['current_message']
        messages = state.get('messages', [])

        chat_config = RunnableConfig(
            configurable={
                "thread_id": f"chat-{session_id}"
            }
        )

        result = chat_subgraph.invoke(
            {
                "messages": messages + [HumanMessage(content=current_msg)],
                "user_name": user_name
            }, 
            chat_config
        )

        response = result['messages'][-1].content

        return MainState(
            messages=[
                HumanMessage(content=current_msg), 
                AIMessage(content=response)
            ], 
            total_messages=state.get('total_messages', 0) + 1
        ) # type: ignore
    
    def analyze_message(state: MainState) -> MainState:
        current_msg = state['current_message']
        session_id = state['session_id']

        analysis_config = RunnableConfig(
            configurable={
                "thread_id": f"analysis-{session_id}"
            }
        )

        result = analysis_subgraph.invoke(
            {
                "last_message": current_msg,
                "sentiment": None,
                "analysis_count": 0
            },
            analysis_config
        )

        sentiment = result['sentiment']

        return MainState(sentiment=sentiment) # type: ignore
    
    builder = StateGraph(MainState)

    builder.add_node('process', process_message)
    builder.add_node('analyze', analyze_message)

    builder.add_edge(START, 'process')
    builder.add_edge('process', 'analyze')
    builder.add_edge('analyze', END)

    checkpointer = MemorySaver()

    return builder.compile(checkpointer=checkpointer)


def checkpointer_subgraph_example_1():
    graph = create_main_graph()

    config_user_1 = RunnableConfig(
        configurable={
            "thread_id": "session-user_1"
        }
    )

    config_user_2 = RunnableConfig(
        configurable={
            "thread_id": "session-user_2"
        }
    )


    result_1 = graph.invoke({
        'user_id': "user_1",
        "user_name": "홍길동",
        "session_id": "user1",
        "current_message": "안녕하세요! 오늘 날씨가 좋네요.",
        "messages": [],
        "sentiment": None,
        "total_messages": 0
    }, config_user_1)


    result_2 = graph.invoke({
        'user_id': "user_2",
        "user_name": "김영희",
        "session_id": "user2",
        "current_message": "도와주세요! 문제가 있어요.",
        "messages": [],
        "sentiment": None,
        "total_messages": 0        
    }, config_user_2)

    result_1_2 = graph.invoke({
        "current_message": "맞아요 오늘 산책하기 좋은거 같아요. 제 이름이 뭐라구요?.",
    }, config_user_1) # type: ignore

    result_2_2 = graph.invoke({
        "current_message": "계속 오류가 나는데 정말 답답해요... 제 이름이 뭐라구요?"
    }, config_user_2) # type: ignore


    
    final_state = graph.get_state(config_user_1)
    messages = final_state.values.get('messages', [])
    
    for i, msg in enumerate(messages, 1):
        if isinstance(msg, HumanMessage):
            logger.info(f"👤 사용자: {msg.content}")
        elif isinstance(msg, AIMessage):
            logger.info(f"🤖 봇: {msg.content}")


    logger.info(f"{'='* 30}")

        
    final_state_2 = graph.get_state(config_user_2)
    messages_2 = final_state_2.values.get('messages', [])
    
    for i, msg in enumerate(messages_2, 1):
        if isinstance(msg, HumanMessage):
            logger.info(f"👤 사용자: {msg.content}")
        elif isinstance(msg, AIMessage):
            logger.info(f"🤖 봇: {msg.content}")


if __name__ == "__main__":
    checkpointer_subgraph_example_1()