import os
from typing import TypedDict, Literal, Optional
from dotenv import load_dotenv

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import interrupt
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableConfig

from common.llm import get_llm
from utils.logger import setup_logger

logger = setup_logger("human_in_loop")

class RiskAnalysis(BaseModel):
    risk_level: Literal['HIGH', 'MEDIUM', 'LOW'] = Field(
        description="작업의 위험도 수준"
    )
    reason: str = Field(
        description="위험도 판단 근거"
    )
    recommendation: str = Field(
        description="권장 조치사항"
    )

class WorkflowState(TypedDict):
    task: str
    analysis: Optional[RiskAnalysis]
    approved: bool
    feedback: str
    result: str


def analyze_task(state: WorkflowState) -> WorkflowState:
    llm = get_llm()
    llm_with_structured = llm.with_structured_output(RiskAnalysis)

    task = state["task"]

    parser = PydanticOutputParser(pydantic_object=RiskAnalysis)

    prompt = ChatPromptTemplate.from_messages([
        SystemMessage(content="""
당신은 작업 위험도 분석 전문가입니다.
주어진 작업의 위험도를 평가하고 분석 결과를 제공합니다.

위험도 기준:
- HIGH: 데이터 삭제, 시스템 중단, 보안 위험이 있는 작업
- MEDIUM: 데이터 수정, 설정 변경 등 주의가 필요한 작업
- LOW: 조회, 읽기 전용 등 안전한 작업

{format_instructions}
"""),
        ("human", "다음 작업을 분석해주세요:\n\n{task}")
    ])
    
    chain = prompt | llm_with_structured

    analysis = chain.invoke({
        "task": task,
        "format_instructions": parser.get_format_instructions()
    })

    return WorkflowState(analysis=analysis) # type: ignore

def human_approval_node(state: WorkflowState) -> WorkflowState:
    analysis = state['analysis']

    approval_data = interrupt(
        value={
            "task": state["task"],
            "analysis": {
                "risk_level": analysis.risk_level, # type: ignore
                "reason": analysis.reason, # type: ignore
                "recommendation": analysis.recommendation # type: ignore
            }
        }
    )
    
    if isinstance(approval_data, dict):
        approved = approval_data.get('approved', False)
        feedback = approval_data.get('feedback', '')
    else:
        approved = bool(approval_data)
        feedback = ""



    return WorkflowState(approved=approved, feedback=feedback)             # type: ignore

def execute_task(state: WorkflowState) -> WorkflowState:
    task = state['task']
    feedback = state.get('feedback', '')

    result = f"작업사항({task})에 대한 피드백({feedback})을 성공적으로 반영하였습니다!"

    return WorkflowState(result=result) # type: ignore

def cancel_task(state: WorkflowState) -> WorkflowState:
    return WorkflowState(result="사용자에 의해 작업이 취소되었습니다!") # type: ignore

def should_ask_approval(state: WorkflowState) -> Literal['ask', 'execute']:
    analysis = state['analysis']

    if analysis.risk_level in ['HIGH', 'MEDIUM']: # type: ignore
        return 'ask'
    else:
        return 'execute'
    
def approval_router(state: WorkflowState) -> Literal['execute', 'cancel']:
    return 'execute' if state.get('approved', False) else 'cancel'


def get_user_approval(state_values: dict) -> dict:
    """
    사용자로부터 승인/거부 입력 받기
    
    Args:
        state_values: 현재 상태 값 (task, analysis 등 포함)
    
    Returns:
        승인 여부와 피드백을 담은 딕셔너리
    """
    print("\n" + "="*60)
    print("사용자 입력 요청")
    print("="*60)
    
    # ✅ state 값을 사용하여 정보 표시
    task = state_values.get('task', '알 수 없는 작업')
    analysis = state_values.get('analysis', {})
    
    if isinstance(analysis, dict):
        risk_level = analysis.get('risk_level', 'UNKNOWN')
        reason = analysis.get('reason', '없음')
        recommendation = analysis.get('recommendation', '없음')
    else:
        # RiskAnalysis 객체인 경우
        risk_level = getattr(analysis, 'risk_level', 'UNKNOWN')
        reason = getattr(analysis, 'reason', '없음')
        recommendation = getattr(analysis, 'recommendation', '없음')
    
    print(f"\n작업: {task}")
    print(f"- 위험도: {risk_level}")
    print(f"- 근거: {reason}")
    print(f"- 권장사항: {recommendation}")
    print()
    
    # 승인 입력
    while True:
        approval_input = input("승인하시겠습니까? (y/n): ").strip().lower()
        
        if approval_input in ['y', 'yes', '예', 'ㅇ']:
            approved = True
            break
        elif approval_input in ['n', 'no', '아니오', 'ㄴ']:
            approved = False
            break
        else:
            print("잘못된 입력입니다. y 또는 n을 입력하세요.")
    
    # 피드백 입력
    feedback = ""
    if approved:
        feedback = input("피드백을 입력하세요 (선택사항, Enter로 건너뛰기): ").strip()
    else:
        feedback = input("거부 사유를 입력하세요 (선택사항, Enter로 건너뛰기): ").strip()
    
    return {
        "approved": approved,
        "feedback": feedback
    }


def create_graph():
    builder = StateGraph(WorkflowState)

    builder.add_node('analyze', analyze_task)
    builder.add_node('ask', human_approval_node)
    builder.add_node('execute', execute_task)
    builder.add_node('cancel', cancel_task)

    builder.add_edge(START, 'analyze')

    builder.add_conditional_edges(
        'analyze',
        should_ask_approval,
        {
            "ask": "ask",
            "execute": "execute"
        }
    )

    builder.add_conditional_edges(
        "ask",
        approval_router,
        {
            "execute": "execute",
            "cancel": "cancel"
        }
    )

    builder.add_edge("execute", END)
    builder.add_edge("cancel", END)

    checkpointer = MemorySaver()

    return builder.compile(checkpointer=checkpointer)



def main():
    graph = create_graph()

    state = WorkflowState(
        task="프로덕션 데이터베이스의 모든 사용자 테이블 삭제",
        analysis=None,
        approved=False,
        feedback="",
        result=""
    )

    config = RunnableConfig(
        configurable={
            "thread_id": "human_in_loop"
        }
    )

    for event in graph.stream(state, config, stream_mode="values"):
        pass

    state_snapshot = graph.get_state(config)

    if state_snapshot.next:
        approval_response = get_user_approval(state_snapshot.values)

        graph.update_state(
            config,
            approval_response,
            as_node="ask"
        )

        for event in graph.stream(None, config, stream_mode="values"):
            pass

    final_state = graph.get_state(config)
    logger.info(f"최종 결과: {final_state.values['result']}")


if __name__ == "__main__":
    main()
        