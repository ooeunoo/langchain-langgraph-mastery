"""
조건부 엣지

1. Conditional Edges
    - 상태 기반 라우팅
    - 여러 경로 중 하나 선택

2. Command API
    - 노드에서 직접 다음 노드 지정
    - goto와 update 동시 사용

3. Router Pattern
    - 의도 분류
    - 동적 플로우 제어
"""


from typing import Optional, Union
from pydantic import BaseModel
from typing_extensions import Annotated, TypedDict, Literal
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.types import Command
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate

from common.llm import get_llm
from utils.logger import setup_logger

logger = setup_logger(__name__)


TicketPriority = Literal['urgent', 'high', 'normal', 'low']
TicketCategory = Literal['technical', 'billing', 'general']
ContentGenre = Literal['action', 'drama', 'comedy', 'documentary']
AgeGroup = Literal['child', 'teen', 'adult']
PaymentStatus = Literal['approved', 'pending', 'declined']

DEFAULT_PRIORITY: TicketPriority = 'normal'
DEFAULT_GENRE: ContentGenre = 'drama'
DEFAULT_STATUS: PaymentStatus = 'pending'


# Graph State
class SupportTicketState(TypedDict):
    messages: Annotated[list, add_messages]
    priority: Optional[TicketPriority]
    category: Optional[TicketCategory]
    ticket_id: str

class ContentRecommendationState(TypedDict):
    messages: Annotated[list, add_messages]
    genre: Optional[ContentGenre] 
    age_group: Optional[AgeGroup] 
    user_id: str

class OrderState(TypedDict):
    messages: Annotated[list, add_messages]
    payment_status: PaymentStatus
    order_id: str
    amount: float


def route_by_priority(state: SupportTicketState) -> Optional[TicketPriority]:
    return state.get('priority', DEFAULT_PRIORITY)

def route_by_genre(state: ContentRecommendationState) -> Optional[ContentGenre]:
    return state.get('genre', DEFAULT_GENRE)


def ticket_classifier_node(state: SupportTicketState) -> SupportTicketState:
    llm = get_llm()

    user_message = state['messages'][-1].content

    class ClassifyInquire(BaseModel):
        priority: TicketPriority
        category: TicketCategory

    parser = PydanticOutputParser(pydantic_object=ClassifyInquire)

    prompt = ChatPromptTemplate.from_template(
        "다음 고객 문의를 분석하여 우선순위와 카테고리를 결정하세요."
        "{format_instructions}\n\n"

        "고객 문의:\n{user_message}"
    )

    chain = prompt | llm | parser

    result: ClassifyInquire = chain.invoke({'user_message': user_message, 'format_instructions': parser.get_format_instructions()})

    return SupportTicketState(priority=result.priority, category=result.category) # type: ignore


def urgent_handler_node(state: SupportTicketState) -> SupportTicketState:
    response = AIMessage(content=f"긴급 우선 순위 티켓 #{state['ticket_id']}\n즉시 처리합니다.")
    return SupportTicketState(messages=[response]) # type: ignore

def high_handler_node(state: SupportTicketState) -> SupportTicketState:
    response = AIMessage(content=f"높은 우선 순위 티켓 #{state['ticket_id']}\n1~2시간 내 답변합니다.")    
    return SupportTicketState(message=[response]) # type: ignore


def normal_handler_node(state: SupportTicketState) -> SupportTicketState:
    response = AIMessage(content=f"일반 우선 순위 티켓 #{state['ticket_id']}\n24시간 내 답변합니다.")    
    return SupportTicketState(message=[response]) # type: ignore


def low_handler_node(state: SupportTicketState) -> SupportTicketState:
    response = AIMessage(content=f"낮은 우선 순위 티켓 #{state['ticket_id']}\n2~3일 내 답변합니다.")    
    return SupportTicketState(message=[response]) # type: ignore


def conditional_edges_example_1():
    builder = StateGraph(SupportTicketState)

    builder.add_node('classifier', ticket_classifier_node)
    builder.add_node('urgent', urgent_handler_node)
    builder.add_node("high", high_handler_node)
    builder.add_node("normal", normal_handler_node)
    builder.add_node("low", low_handler_node)

    builder.add_edge(START, 'classifier')

    builder.add_conditional_edges(
        'classifier',
        route_by_priority,
        {
            "urgent": 'urgent',
            "high": "high",
            "normal": "normal",
            "low": "low"
        }
    )

    builder.add_edge("urgent", END)
    builder.add_edge("high", END)
    builder.add_edge("normal", END)
    builder.add_edge("low", END)

    graph = builder.compile()


    example_inquires = [
        "서비스가 다운되었어요! 이용이 불가능한 상황입니다.",
        "결제에 문제가 있어요. 결제를 진행했는데 구독되지않습니다.",
        "새로운 기능을 추가 요청드려요. 이 기능이 있으면 더 좋을 거 같습니다",
        "결제한 뒤 사용법이 궁금합니다."
    ]

    for i, inquiry in enumerate(example_inquires, 1):
        state = SupportTicketState(
            messages=[HumanMessage(content=inquiry)],
            priority=None,
            category=None,
            ticket_id=f"TKT-{1000+i}"
        )
        result = graph.invoke(state)

        logger.info(f"테스트 문의:\n{inquiry}")
        logger.info(f"[결과]우선순위: {result['priority']}")
        logger.info(f"[결과]카테고리: {result['category']}")
        logger.info(f"[결과]응답 메시지: {result['messages'][-1].content}")

if __name__ == "__main__":
    conditional_edges_example_1()