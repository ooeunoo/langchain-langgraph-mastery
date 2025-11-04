"""
기본 서브 그래프
"""

from typing import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.state import CompiledStateGraph

from utils.logger import setup_logger

logger = setup_logger("basic_subgraph")

class MainGraphState(TypedDict):
    task: str
    sub_result: str
    final_output: str

class SubGraphState(TypedDict):
    input: str
    result: str


def create_simple_subgraph()-> CompiledStateGraph:
    """
        단순 서브 그래프
        Step1: 입력 처리
        Step2: 결과 최종화
    """
    def step1(state: SubGraphState) -> SubGraphState:
        processed = f"Processed -> {state['input']}"
        return SubGraphState(result=processed) # type: ignore
    
    def step2(state: SubGraphState) -> SubGraphState:
        finalized = f"Finalized -> {state['result']}"
        return SubGraphState(result=finalized) # type: ignore
    
    builder = StateGraph(SubGraphState)

    builder.add_node('step1', step1)
    builder.add_node('step2', step2)

    builder.add_edge(START, 'step1')
    builder.add_edge('step1', 'step2')
    builder.add_edge('step2', END)

    return builder.compile() 

def basic_subgraph_example_1():

    subgraph = create_simple_subgraph()

    def start_task(state: MainGraphState) -> MainGraphState:
        return SubGraphState() # type: ignore
     
    def call_subgraph(state: MainGraphState) -> MainGraphState:
        sub_result = subgraph.invoke(SubGraphState(input=state["task"])) # type: ignore
        return MainGraphState(sub_result=sub_result) # type: ignore
    
    def finalize(state: MainGraphState) -> MainGraphState:
        final = f"Final: {state['sub_result']}"
        return MainGraphState(final_output=final) # type: ignore

    builder = StateGraph(MainGraphState)

    builder.add_node('start', start_task)
    builder.add_node('process_step', call_subgraph)
    builder.add_node('finalize', finalize)

    builder.add_edge(START, 'start')
    builder.add_edge('start', 'process_step')
    builder.add_edge('process_step', 'finalize')
    builder.add_edge('finalize', END)

    graph = builder.compile()

    result = graph.invoke(MainGraphState(task='Make Financial Report')) # type: ignore
    # 결과 출력
    logger.info("\n" + "="*60)
    logger.info(f"Task: {result['task']}")
    logger.info(f"Sub Result: {result['sub_result']}")
    logger.info(f"Final Output: {result['final_output']}")


if __name__ == "__main__":
    basic_subgraph_example_1()