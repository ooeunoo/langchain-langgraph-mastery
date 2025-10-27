"""
병렬 노드
"""
from typing import Union, Optional
from pydantic import BaseModel
from typing_extensions import Annotated, TypedDict, Literal
from langgraph.graph import StateGraph, START, END
from langgraph.types import Send  
from langgraph.graph.message import add_messages
from langchain_core.messages import HumanMessage, AIMessage
from operator import add
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser, CommaSeparatedListOutputParser
from IPython.display import Image, display


from common.llm import get_llm
from utils.logger import setup_logger

logger = setup_logger(__name__)


AnalysisType = Literal['sentiment', 'keywords', 'summary']
TranslationLang = Literal['en', 'ja', 'zh', 'es']
Datasource = Literal['api', 'database', 'file']

SentimentType = Literal['positive', 'negative', 'neutral']

class TextAnalysisState(TypedDict):
    messages: Annotated[list, add_messages]
    text: str
    sentiment: Optional[SentimentType]
    keywords: list[str]
    summary: str
    analysis_complete: bool

class TranslationState(TypedDict):
    messages: Annotated[list, add_messages]
    original_text: str
    translations: dict[str, str]
    languages: list[str]

class DataAggregationState(TypedDict):
    messages: Annotated[list, add_messages]
    query: str
    results: Annotated[list[dict], add]
    sources: list[str]
    total_count: int


def sentiment_analysis_node(state: TextAnalysisState) -> TextAnalysisState:
    llm = get_llm()

    text = state['text']

    class SentimentOutput(BaseModel):
        sentiment: SentimentType

    parser = PydanticOutputParser(pydantic_object=SentimentOutput)

    prompt = ChatPromptTemplate.from_template(
        "다음 텍스트의 감정을 분석하세요."
        "{format_instructions}"
        "텍스트:\n{text}"
    )

    chain = prompt | llm | parser
    result = chain.invoke({
        "text": text,
        "format_instructions": parser.get_format_instructions()
    })

    return TextAnalysisState(sentiment=result.sentiment) # type: ignore
    

def keyword_extraction_node(state: TextAnalysisState) -> TextAnalysisState:
    llm = get_llm()

    text = state["text"]


    parser = CommaSeparatedListOutputParser()

    prompt = ChatPromptTemplate.from_template(
        "다음 텍스트에서 핵심 키워드 5개를 추출하세요."
        "{format_instructions}"
        "텍스트:\n{text}"
    )

    chain = prompt | llm | parser

    result = chain.invoke({
        "text": text,
        "format_instructions": parser.get_format_instructions()
    })

    print(result)

    return TextAnalysisState(keywords=result) # type: ignore


def summarization_node(state: TextAnalysisState) -> TextAnalysisState:
    llm = get_llm()

    text = state['text']

    class SummarizeOutput(BaseModel):
        summary: str

    parser = PydanticOutputParser(pydantic_object=SummarizeOutput)

    prompt = ChatPromptTemplate.from_template(
        "다음 텍스트를 한 문장으로 요약하세요"
        "{format_instructions}"
        "텍스트:\n{text}"
    )

    chain = prompt | llm | parser

    result = chain.invoke({
        "text": text,
        "format_instructions": parser.get_format_instructions() 
    })

    return TextAnalysisState(summary=result.summary) # type: ignore

def aggregator_node(state: TextAnalysisState) -> TextAnalysisState:
    sentiment = state.get('sentiment', '')
    keywords = state.get('keywords', [])
    summary = state.get('summary', '')

    report = AIMessage(
        content=f"""
        분석 결과 보고서

        요약: {summary}
        감정: {sentiment}
        키워드: {', '.join(keywords)}

        """
    )

    return TextAnalysisState(messages=[report], analysis_complete=True) # type: ignore


def parallel_execution_example_1():
    """
    Send API를 사용한 Fan-out/Fan-in 패턴
    """
    builder = StateGraph(TextAnalysisState)

    # 노드 추가
    builder.add_node('sentiment', sentiment_analysis_node)
    builder.add_node('summary', summarization_node)
    builder.add_node('keyword', keyword_extraction_node)
    builder.add_node('aggregator', aggregator_node)

    def route_parallel_tasks(state: TextAnalysisState) -> list[Send]:
        return [
            Send('sentiment', state),
            Send('summary', state),
            Send('keyword', state)
        ]
    
    builder.add_conditional_edges(
        START,
        route_parallel_tasks,
        ['sentiment', 'keyword', 'summary']
    )

    # 병렬작업이 완료되면 aggregator로 fan-in
    builder.add_edge('sentiment', 'aggregator')
    builder.add_edge('summary', 'aggregator')
    builder.add_edge('keyword', 'aggregator')

    builder.add_edge('aggregator', END)

    graph = builder.compile()

    state = TextAnalysisState(
        messages=[HumanMessage(content="텍스트 분석 시작")],
        text="인공지능 기술의 발전으로 우리의 삶이 크게 변화하고 있습니다. "
             "특히 자연어 처리 기술은 놀라운 성과를 보여주고 있으며, "
             "앞으로 더 많은 혁신이 기대됩니다.",
        sentiment=None,
        keywords=[],
        summary="",
        analysis_complete=False
    )

    result = graph.invoke(state)

    logger.info(f"\n최종 결과:")
    logger.info(f"감정: {result['sentiment']}")
    logger.info(f"키워드: {result['keywords']}")
    logger.info(f"요약: {result['summary']}")
    logger.info(f"분석 완료: {result['analysis_complete']}")


    

if __name__ == "__main__":
    parallel_execution_example_1()