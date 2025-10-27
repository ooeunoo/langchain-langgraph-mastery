from typing import Any, Sequence, TypedDict
from langchain_core.messages.base import BaseMessage
from langchain_core.prompt_values import PromptValue
from langchain_core.runnables.base import RunnableSerializable
from langgraph.graph import StateGraph, START, END
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import PydanticOutputParser, CommaSeparatedListOutputParser, StrOutputParser
from langgraph.graph.state import CompiledStateGraph
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.prompts import ChatPromptTemplate
from datetime import datetime


from common.llm import get_llm
from config import settings
from utils.logger import setup_logger

logger = setup_logger("multiple_subgraph")

class ResearchState(TypedDict):
    topic: str
    web_data: str
    analysis: str
    report: str

class SearchState(TypedDict):
    query: str
    results: str

class AnalysisState(TypedDict):
    data: str
    insights: str


def create_web_search_subgraph() -> CompiledStateGraph:

    def search(state: SearchState) -> SearchState:
        query = state["query"]

        search_tool = TavilySearchResults(
            tavily_api_key=settings.TAVILY_API_KEY,
            max_results=5,
            include_answer=True,
            include_raw_content=False,
            include_images=False
        )

        search_results = search_tool.invoke({"query": query})

        results = f"Search results for '{query}':\n\n"

        for i, result in enumerate(search_results, 1):
            title = result.get('title', 'No title')
            url = result.get('url', 'No url')
            content = result.get('content', 'No content')

            results += f"{i}. {title}\n"
            results += f"    URL: {url}\n"
            results += f"    {content[:200]}...\n\n"

        return SearchState(results=results) # type: ignore  
    
    def filter_results(state: SearchState) -> SearchState:
        results = state['results']

        # 단순 필터링 작업이라 생각
        filtered = f"Filtered and Ranked Results:\n{'='*50}\n\n{results}"

        return SearchState(results=results) # type: ignore


    builder = StateGraph(SearchState)

    builder.add_node('search', search)
    builder.add_node('filter', filter_results)

    builder.add_edge(START, 'search')
    builder.add_edge('search', 'filter')
    builder.add_edge('filter', END)

    return builder.compile()


def create_analysis_subgraph() -> CompiledStateGraph:
    def analyze(state: AnalysisState) -> AnalysisState:
        llm = get_llm()

        data = state["data"]

        prompt = ChatPromptTemplate.from_template(
            "다음 검색 결과를 분석하여 핵심 인사이트를 추출하세요."
            "다음 형식으로 분석해주세요:"
            "1. 주요 발견사항 (3-5개)"
            "2. 트렌드 분석"
            "3. 실용적 권장사항"

            "검색 결과:\n"
            "{data}"
        )

        chain = prompt | llm | StrOutputParser()
        response = chain.invoke({"data": data}) # type: ignore

        return AnalysisState(insights=response) # type: ignore

        
    def validate(state: AnalysisState) -> AnalysisState:
        insights = state['insights']

        # 간단한 validation
        validated = f" Validated Insights\n{'='*50}\n\n{insights}"
        return AnalysisState(insights=validated)  # type: ignore
    
    builder = StateGraph(AnalysisState)

    builder.add_node('analyze', analyze)
    builder.add_node('validate', validate)

    builder.add_edge(START, 'analyze')
    builder.add_edge('analyze', 'validate')
    builder.add_edge('validate', END)

    return builder.compile()

def multiple_subgraph_example_1():

    search_subgraph = create_web_search_subgraph()
    analyze_subgraph = create_analysis_subgraph()

    def initlize(state: ResearchState) -> ResearchState:
        return ResearchState() # type: ignore

    def search_web(state: ResearchState) -> ResearchState:
        result = search_subgraph.invoke(SearchState(query=state['topic'], results=""))
        return ResearchState(web_data=result['results']) # type: ignore
    
    def analyze_data(state: ResearchState) -> ResearchState:
        result = analyze_subgraph.invoke(AnalysisState(data=state['web_data'], insights=""))
        return ResearchState(analysis=result['insights']) # type: ignore
    
    def generate_report(state: ResearchState) -> ResearchState:
        current_date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        report = f"""
{'='*70}
📄 RESEARCH REPORT
{'='*70}

주제: {state['topic']}
생성일시: {current_date}

{'='*70}
🔍 수집된 데이터
{'='*70}

{state['web_data']}

{'='*70}
📊 분석 결과
{'='*70}

{state['analysis']}

{'='*70}
"""
        return ResearchState(report=report.strip()) # type: ignore
    
    builder = StateGraph(ResearchState)

    builder.add_node('init', initlize)
    builder.add_node('search', search_web)
    builder.add_node('analyze', analyze_data)
    builder.add_node('report', generate_report)

    builder.add_edge(START, 'init')
    builder.add_edge('init', 'search')
    builder.add_edge('search', 'analyze')
    builder.add_edge('analyze', 'report')
    builder.add_edge('report', END)

    graph = builder.compile()

    result = graph.invoke(
        ResearchState(
            topic="Langgraph 최신 기능과 활용 사례",
            web_data="",
            analysis="",
            report=""
        )
    )

    logger.info("="*70)
    logger.info("\n📄 최종 리포트:")
    logger.info(result["report"])
    


if __name__ == "__main__":
    multiple_subgraph_example_1()