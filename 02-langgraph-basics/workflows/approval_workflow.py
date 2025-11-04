from typing import Any, TypedDict, Literal, cast
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, END
from common.llm import get_llm
from langchain_core.runnables import RunnableConfig

from utils.logger import setup_logger


logger = setup_logger("approval_workflow")

class RequestAnalysis(BaseModel):
    summary: str = Field(description="요청 내용 요약 (2-3줄)")
    estimated_cost: float = Field(description="예상 비용 (USD)")
    risk_level: Literal["low", "medium", "high"] = Field(description="위험 수준")
    recommendations: list[str] = Field(description="권장 사항 리스트")

class ExecutionDetails(BaseModel):
    """실행 상세 정보"""
    timestamp: str = Field(description="실행 시간")
    affected_resources: list[str] = Field(description="영향받은 리소스 목록")
    completion_time: str = Field(description="완료 소요 시간")
    notes: str = Field(description="추가 메모")

class TaskExecutionResult(BaseModel):
    """작업 실행 결과"""
    success: bool = Field(description="성공 여부")
    message: str = Field(description="실행 결과 메시지")
    details: ExecutionDetails = Field(description="상세 정보")

class ApprovalState(TypedDict):
    """승인 워크플로우 상태"""
    request: str                          # 요청 내용
    analysis: RequestAnalysis | None      # 분석 결과 (구조화됨!)
    approved: bool | None                 # 승인 여부
    approval_comment: str                 # 승인자 코멘트
    execution_result: TaskExecutionResult | None  # 실행 결과 (구조화됨!)
    status: str       

def analyze_request(state: ApprovalState) -> ApprovalState:
    """
    요청 분석
    """
    request = state["request"]
    
    llm = get_llm()
    structured_llm = llm.with_structured_output(RequestAnalysis)
    
    prompt = f"""
다음 작업 요청을 분석해주세요:

요청: {request}

요청 내용을 분석하여 요약, 예상 비용, 위험 수준, 권장사항을 제공해주세요.
"""
    
    analysis = cast(RequestAnalysis, structured_llm.invoke(prompt))
    
    return ApprovalState(
        analysis=analysis,
        status="분석 완료"
    ) # type: ignore

def wait_for_approval(state: ApprovalState) -> ApprovalState:
    analysis = state["analysis"]

    return ApprovalState(status="승인 대기 중") # type: ignore

def execute_approved_task(state: ApprovalState) -> ApprovalState:
    request = state['request']
    analysis = state['analysis']

    llm = get_llm()
    strucutred_llm = llm.with_structured_output(TaskExecutionResult)

    prompt = f"""
다음 작업을 실행했다고 가정하고 결과를 생성해주세요:

요청: {request}
분석: {analysis.summary if analysis else "N/A"}

실제 실행한 것처럼 구체적인 결과를 만들어주세요.
"""
    
    execution_result = cast(TaskExecutionResult, strucutred_llm.invoke(prompt))

    return ApprovalState(
        execution_result=execution_result, 
        status="실행 완료"
    ) # type: ignore

def handle_rejection(state: ApprovalState) -> ApprovalState:
    comment = state.get('approval_comment', '사유 없음')

    return ApprovalState(
        execution_result=TaskExecutionResult(
            success=False,
            message=f"작업이 거부되었습니다: {comment}",
            details=ExecutionDetails(
                timestamp="N/A",
                affected_resources=[],
                completion_time="즉시",
                notes=f"거부 사유: {comment}"
            )
        ),
        status="거부됨"
    ) # type: ignore

def route_after_approval(state: ApprovalState) -> Literal['execute', 'reject']:
    if state.get('approved') is True:
        return 'execute'
    else:
        return 'reject'

def create_approval_workflow():
    builder = StateGraph(ApprovalState)

    builder.add_node('analyze', analyze_request)
    builder.add_node('wait_approval', wait_for_approval)
    builder.add_node('execute', execute_approved_task)
    builder.add_node('reject', handle_rejection)

    builder.add_edge(START, 'analyze')
    builder.add_edge('analyze', 'wait_approval')

    builder.add_conditional_edges(
        'wait_approval',
        route_after_approval,
        {
            "execute": "execute",
            "reject": "reject"
        }
    )

    builder.add_edge("execute", END)
    builder.add_edge("reject", END)

    memory = MemorySaver()

    return builder.compile(checkpointer=memory)


def approval_workflow_example():
    graph = create_approval_workflow()
    config = RunnableConfig(
        configurable={
            'thread_id': 'example'
        }
    )

    state = ApprovalState(
        request="프로덕션 서버에 v2.0 배포하고 데이터베이스 스키마 업데이트",
        analysis=None,
        approved=None,
        approval_comment="",
        execution_result=None,
        status="초기"
    ) # type: ignore

    logger.info("=== 1단계: 분석 단계 실행 (interrupt 전까지) ===")
    # 첫 번째 stream: analyze 노드까지 실행되고 wait_approval 전에 중단
    for event in graph.stream(state, config, stream_mode="values", interrupt_before=['wait_approval']):
        if 'analysis' in event and event['analysis']:
            analysis = event['analysis']
            logger.info(f"📊 분석 결과:")
            logger.info(f"  - 요약: {analysis.summary}")
            logger.info(f"  - 예상 비용: ${analysis.estimated_cost}")
            logger.info(f"  - 위험 수준: {analysis.risk_level}")
            logger.info(f"  - 권장사항: {', '.join(analysis.recommendations)}")

    # interrupt 지점에서 상태 확인
    current_state = graph.get_state(config)
    logger.info(f"\n⏸️  현재 상태: {current_state.values.get('status')}")
    logger.info(f"다음 노드: {current_state.next}")  # ('wait_approval',) 이어야 함

    if not current_state.next:
        logger.error("❌ interrupt가 제대로 작동하지 않았습니다!")
        return

    logger.info(f"\n⏳ wait_approval 노드 전에 중단됨 - 승인 대기 중...")

    # 사용자 입력 시뮬레이션 (실제로는 여기서 사용자 입력을 받아야 함)
    logger.info("\n=== 2단계: 승인 처리 ===")
    user_decision = input("승인하시겠습니까? (y/n): ").strip().lower()

    # 상태 업데이트
    if user_decision == 'y':
        logger.info("✅ 승인됨")
        updated_state = ApprovalState(
            approved=True,
            approval_comment="승인합니다"
        ) # type: ignore
    else:
        logger.info("❌ 거부됨")
        updated_state = ApprovalState(
            approved=False,
            approval_comment="보안 검토 필요"
        ) # type: ignore

    # 상태 업데이트 후 재개
    logger.info("\n=== 3단계: 워크플로우 재개 ===")
    for event in graph.stream(updated_state, config, stream_mode="values"):
        if 'execution_result' in event and event['execution_result']:
            result = event['execution_result']
            if result.success:
                logger.info(f"✅ 실행 성공: {result.message}")
                logger.info(f"📋 상세 정보:")
                logger.info(f"  - 실행 시간: {result.details.timestamp}")
                logger.info(f"  - 영향받은 리소스: {', '.join(result.details.affected_resources)}")
                logger.info(f"  - 완료 시간: {result.details.completion_time}")
                logger.info(f"  - 메모: {result.details.notes}")
            else:
                logger.info(f"❌ 실행 실패: {result.message}")
                logger.info(f"사유: {result.details.notes}")

    logger.info(f"\n=== 최종 결과 ===")
    final_state = graph.get_state(config)
    logger.info(f"최종 상태: {final_state.values.get('status')}")
if __name__ == "__main__":
    approval_workflow_example()