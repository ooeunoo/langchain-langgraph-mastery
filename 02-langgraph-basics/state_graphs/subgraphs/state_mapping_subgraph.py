from typing import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.state import CompiledStateGraph
from langchain_core.output_parsers import CommaSeparatedListOutputParser
from langchain_core.prompts import ChatPromptTemplate

from common.llm import get_llm


class DocumentState(TypedDict):
    document: str
    summary: str
    keywords: list[str]

class ProcessingSubState(TypedDict):
    text: str
    analysis: str
    keywords: list[str]


def create_document_processor_subgraph() -> CompiledStateGraph:
    def analysize_text(state: ProcessingSubState) -> ProcessingSubState:
        llm = get_llm()
        
        text = state["text"]

        prompt = f"Summarize in one sentence: {text}"

        response = llm.invoke(prompt)

        return ProcessingSubState(analysis=response) # type: ignore
    
    def extract_keywords(state: ProcessingSubState) -> ProcessingSubState:
        llm = get_llm()

        analysis = state["analysis"]

        parser = CommaSeparatedListOutputParser()

        prompt = ChatPromptTemplate.from_template(
            "Extract keywords from sentence"
            "{format_instructions}"
            "sentence:\n{analysis}"
        )

        chain = prompt | llm | parser

        response = chain.invoke({"analysis": analysis, "format_instructions": parser.get_format_instructions()})

        return ProcessingSubState(keywords=response) # type: ignore


    builder = StateGraph(ProcessingSubState)

    builder.add_node('analysis', analysize_text)
    builder.add_node('keywords', extract_keywords)

    builder.add_edge(START, 'analysis')
    builder.add_edge('analysis', 'keywords')
    builder.add_edge('keywords', END)

    return builder.compile()

def state_mapping_subgraph_example_1():
    subgraph = create_document_processor_subgraph()

    def prepare_document(state: DocumentState) -> DocumentState:
        return DocumentState() # type: ignore
    
    def process_with_subgraph(state: DocumentState) -> DocumentState:
        subgraph_state =  ProcessingSubState(
            text=state['document'],
            analysis=""
        )   # type: ignore

        subgraph_result = subgraph.invoke(subgraph_state)

        return DocumentState(summary=subgraph_result['analysis'], keywords=subgraph_result['keywords']) # type: ignore
    
    def save_result(state: DocumentState) -> DocumentState:
        return DocumentState() # type: ignore
    
    builder = StateGraph(DocumentState)

    builder.add_node('prepare', prepare_document)
    builder.add_node('process', process_with_subgraph)
    builder.add_node('save', save_result)

    builder.add_edge(START, 'prepare')
    builder.add_edge('prepare', 'process')
    builder.add_edge('process', 'save')
    builder.add_edge('save', END)

    graph = builder.compile()

    document = """
    Artificial Intelligence is transforming the technology industry.
    Machine learning models are becoming more sophisticated every day.
    LangGraph provides a framework for building stateful AI applications.
    """
    
    result = graph.invoke(DocumentState(document=document,summary= "",keywords=[]))

    print(f"Summary: {result['summary']}")
    print(f"Keywords: {result['keywords']}")
    print(f"Document length: {len(result['document'])} chars")

if __name__ == "__main__":
    state_mapping_subgraph_example_1()

