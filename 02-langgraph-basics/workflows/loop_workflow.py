from typing import TypedDict, Annotated, Literal, cast
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages, BaseMessage
from langchain.messages import HumanMessage, SystemMessage, AIMessage
from langgraph.graph.state import CompiledStateGraph
from pydantic import BaseModel, Field
from common.llm import get_llm
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from utils.logger import setup_logger

logger = setup_logger("loop_workflow")

class  GraphState(TypedDict):
    messages: Annotated[BaseMessage, add_messages]
    task: str
    iteration: int
    max_interations: int
    current_response: str
    evaluation: str
    quality_score: float
    target_score: float
    feedback_history: list[str]

def generate_response(state: GraphState) -> GraphState:
    llm = get_llm()
    iteration = state.get('iteration', 0) + 1
    parser = StrOutputParser()

    if iteration == 1:
        messages = [
            SystemMessage(content="당신은 콘텐츠 작성 전문가입니다."),
            HumanMessage(content=f"{state.get('task')}")
        ]
    else:
        previous_feedback = state.get("feedback_history", [])[-1] if state.get("feedback_history") else ""
        messages = [
            SystemMessage(content="당신은 콘텐츠 개선 전문가입니다."),
            HumanMessage(content=f"""이전 응답을 개선해주세요.
[이전 응답]
{state.get("current_response")}

[피드백]
{previous_feedback}

위 피드백을 반영하여 더 나은 응답을 생성해주세요.""")
        ]

    prompt = ChatPromptTemplate.from_messages(messages)
    chain = prompt | llm | parser
    current_response = chain.invoke({})


    return GraphState(
        iteration=iteration, 
        current_response=current_response, 
        messages=state.get("messages", []) + messages + [AIMessage(content=current_response)]
    ) # type: ignore

def evaluate_response(state: GraphState) -> GraphState:
    llm = get_llm()

    class EvaluationScore(BaseModel):
        completeness: int = Field(description="완성도 점수")
        clarity: int = Field(description="명확성 점수")
        specificity: int = Field(description="구체성 점수")
        practicality: int = Field(description="실용성 점수")
        structure: int = Field(description="구조성 점수")

    llm_with_structured = llm.with_structured_output(EvaluationScore)

    messages = [
        SystemMessage(content="""당신은 까다로운 품질 평가자입니다.
다음 기준으로 평가하세요:
1. 완성도 (20점)
2. 명확성 (20점)
3. 구체성 (20점)
4. 실용성 (20점)
5. 구조 (20점)"""),
        HumanMessage(content=f"""다음 응답을 평가해주세요:

[원래 작업]
{state.get('task')}

[생성된 응답]
{state.get('current_response')}""")
    ]

    prompt = ChatPromptTemplate.from_messages(messages)
    chain = prompt | llm_with_structured
    evaluation = cast(EvaluationScore, chain.invoke({}))

    quality_score = (evaluation.completeness + evaluation.clarity + 
                    evaluation.specificity + evaluation.practicality + 
                    evaluation.structure)

    evaluation_text = f"""점수: {quality_score}
완성도: {evaluation.completeness}/20
명확성: {evaluation.clarity}/20
구체성: {evaluation.specificity}/20
실용성: {evaluation.practicality}/20
구조: {evaluation.structure}/20"""

    normalized_score = quality_score / 100.0
    
    feedback_history = state.get("feedback_history", []) + [evaluation_text]

    return GraphState(
        evaluation=evaluation_text,
        quality_score=normalized_score,
        feedback_history=feedback_history,
        messages=state.get("messages", []) + messages + [AIMessage(content=evaluation_text)]
    ) # type: ignore

def should_continue(state: GraphState) -> Literal['continue', 'finish']:
    iteration = state.get('iteration', 0)
    max_iterations = state.get('max_iterations', 3)
    quality_score = state.get('quality_score', 0.0)
    target_score = state.get('target_score', 0.8)

    if iteration >= max_iterations:
        logger.info(f"   ⏹️  최대 반복 도달 → 종료")
        return 'finish'
    
    if quality_score >= target_score:
        logger.info(f"   ✅ 목표 달성! → 종료")
        return 'finish'
    
    logger.info(f"   🔄 품질 부족 → 재생성 (남은 시도: {max_iterations - iteration}회)")
    return 'continue'



def generate_final_report(state: GraphState) -> GraphState:
    logger.info(f"\n{'='*70}")
    logger.info("📋 최종 리포트 생성")
    logger.info(f"{'='*70}")

    success = state.get('quality_score', 0) >= state.get('target_score', 0.8)
    status = "✅ 목표 달성" if success else "⚠️  부분 완료"

    report = f"""
{'='*70}
최종 리포트
{'='*70}

[작업]
{state.get('task')}

[상태]
{status}

[통계]
- 반복 횟수: {state.get('iteration')}회
- 최종 품질: {state.get('quality_score'):.2f} / {state.get('target_score'):.2f}
- 개선 횟수: {len(state.get('feedback_history', []))}회

[최종 응답]
{state.get('current_response')}

[최종 평가]
{state.get('evaluation')}

{'='*70}
"""
    
    logger.info(report)

    return GraphState(
        messages=state.get("messages", []) + [AIMessage(content=report)]
    ) # type: ignore

def create_loop_graph() -> CompiledStateGraph:
    builder = StateGraph(GraphState)

    builder.add_node('generate', generate_response)
    builder.add_node('evaluate', evaluate_response)
    builder.add_node('report', generate_final_report)

    builder.add_edge(START, 'generate')
    builder.add_edge('generate', 'evaluate')

    builder.add_conditional_edges(
        'evaluate',
        should_continue,
        {
            "continue": 'generate',
            "finish": "report"
        }
    )

    builder.add_edge("report", END)

    return builder.compile()

def loop_graph_example():
    graph = create_loop_graph()

    task = "양자 컴퓨팅을 코드로 구현하는 방법을 자세하게 알려주세요."

    state = GraphState(
        task=task,
        iteration=0,
        max_interations=3,
        current_response="",
        evaluation="",
        quality_score=0.0,
        target_score=0.75,
        feedback_history=[],
        messages=[] # type: ignore
    )

    result = graph.invoke(state)

    for chunk in graph.stream(state):
        node_name = list(chunk.keys())[0]
        node_output = chunk[node_name]
        
        if node_name == "generate":
            logger.info(f"   응답 길이: {len(node_output.get('current_response', ''))}자")
        elif node_name == "evaluate":
            logger.info(f"   품질 점수: {node_output.get('quality_score', 0):.2f}")
        elif node_name == "report":
            logger.info(f"   최종 완료!")
    
    # 최종 결과는 마지막 chunk
    final_state = node_output
    
    for i, feedback in enumerate(final_state.get('feedback_history', []), 1):
        score_line = [line for line in feedback.split('\n') if '점수:' in line]
        if score_line:
            logger.info(f"   반복 {i}: {score_line[0]}")


if __name__ == "__main__":
    loop_graph_example()