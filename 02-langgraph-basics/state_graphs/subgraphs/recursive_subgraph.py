
from typing import TypedDict, Annotated, Literal
from operator import add
from pydantic import BaseModel, Field

from langgraph.graph import StateGraph, START, END
from langgraph.types import Send, Command
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langgraph.graph.state import CompiledStateGraph

from common.llm import get_llm
from utils.logger import setup_logger

logger = setup_logger('recursive_subgraph')


Complexity = Literal['simple', 'moderate', 'complex']


class SubTask(BaseModel):
    title: str
    description: str
    estimated_hours: float

class TaskAnalysis(BaseModel):
    estimated_hours: float
    complexity: Complexity
    needs_breakdown: bool
    reason: str
    subtasks: list[SubTask] = Field(default_factory=list)


class TaskState(TypedDict):
    root_task: dict
    task_id: str
    title: str
    description: str
    depth: int
    all_tasks: Annotated[list[dict], add]


def process_task(state: TaskState) -> Command[Literal['process_task', '__end__']]:
    llm = get_llm()

    task_id = state['task_id']
    title = state['title']
    description = state['description']
    depth = state['depth']

    if depth >= 3:
        return Command(
            goto=END,
            update={
                "all_tasks": [{
                    "id": task_id,
                    "title": title,
                    "description": description,
                    "depth": depth,
                    "estimated_hours": 20.0,
                    "complexity": "moderate"
                }]
            }
        ) # type: ignore
    
    parser = PydanticOutputParser(pydantic_object=TaskAnalysis)

    prompt = ChatPromptTemplate.from_template(
"""작업을 분석하고 필요시 분해하세요.

작업: {title}
설명: {description}
현재 깊이: {depth} (최대 3)

{format_instructions}

분해 규칙 (중요!):
1. estimated_hours
   - 단순 작업: 4-8시간
   - 중간 작업: 16-40시간
   - 큰 작업: 80-200시간

2. needs_breakdown
   - 20시간 이하 → false (바로 실행 가능)
   - 20시간 초과 → true (분해 필요)
   
3. subtasks (분해 시)
   - 2-4개로만 분해 (너무 잘게 쪼개지 말 것!)
   - 각 서브태스크는 10-30시간 정도의 의미있는 단위
   - 관련 기능끼리 묶어서 큰 단위로

좋은 분해 예시:
"전체 백엔드 개발" (120시간) →
  - "인증 시스템 전체" (30시간) - 회원가입, 로그인, OAuth, 권한 등
  - "상품 시스템 전체" (40시간) - 상품 CRUD, 검색, 필터, 카테고리 등
  - "주문/결제 시스템" (30시간) - 장바구니, 결제, 주문관리 등
  - "배포 및 인프라" (20시간) - CI/CD, 모니터링, 로깅 등

나쁜 분해 예시:
"백엔드 개발" →
  - "파일 생성" (1시간) ❌ 너무 작음
  - "API 하나 만들기" (2시간) ❌ 너무 작음
  - "테스트 코드" (3시간) ❌ 너무 작음
"""
    )
    
    chain = prompt | llm | parser

    analysis = chain.invoke({
        "title": title,
        "description": description,
        "depth": depth,
        "format_instructions": parser.get_format_instructions()
    })

    if analysis.needs_breakdown and analysis.subtasks:
        return Command(
            goto=[
                Send(
                    "process_task",
                    {
                        "task_id": f"{task_id}.{i+1}",
                        "title": st.title,
                        "description": st.description,
                        "depth": depth + 1
                    }
                )
                for i, st in enumerate(analysis.subtasks)
            ]
        )
    
    else:
        return Command(
            goto=END,
            update={
                "all_tasks": [{
                    "id": task_id,
                    "title": title,
                    "description": description,
                    "depth": depth,
                    "estimated_hours": analysis.estimated_hours,
                    "complexity": analysis.complexity
                }]
            }
        ) # type: ignore
    

def start_decomposition(state: TaskState) -> Command[Literal['process_task']]:
    root = state['root_task']

    return Command(
        goto=[
            Send(
                "process_task",
                {
                    "task_id": "1",
                    "title": root["title"],
                    "description": root['description'],
                    "depth": 0
                }
            )
        ]
    )


def create_graph() -> CompiledStateGraph:
    builder = StateGraph(TaskState)

    builder.add_node('start', start_decomposition)
    builder.add_node('process_task', process_task)

    builder.add_edge(START, 'start')

    return builder.compile()

def recursive_subgraph_example_1():
    graph = create_graph()

    task = {
        "title": "개인 블로그 웹사이트 만들기",
        "description": """
React와 Node.js를 사용한 풀스택 개인 블로그 웹사이트
- 프론트엔드: React, TailwindCss, 반응형 디자인
- 백엔드: Node.js, Express, MongoDB
- 기능: 글 작성/수정/삭제, 마크다운 에디터, 댓글, 태그
- 배포: Vercel (프론트), Railway (백엔드)
"""
    }

    result = graph.invoke({
        "root_task": task,
        "task_id": "",
        "title": "",
        "description": "",
        "depth": 0,
        "all_tasks": []
    })

    tasks = result["all_tasks"]
    total_hours = sum(t.get('estimated_hours', 0) for t in tasks)

    logger.info(f"총 작업 수: {len(tasks)}")
    logger.info(f"총 예상 시간: {total_hours:.1f}시간 ({total_hours/8:.1f}일)")
    logger.info(f"최대 깊이: {max(t['depth'] for t in tasks)}단계")
    
    logger.info("\n🌲 작업 트리:\n")
    for task in sorted(tasks, key=lambda t: t['id']):
        indent = "  " * task['depth']
        hours = task.get('estimated_hours', 0)
        complexity = task.get('complexity', 'unknown')
        logger.info(f"{indent}✅ [{task['id']}] {task['title']}")
        logger.info(f"{indent}    ⏱️  {hours}시간 | 📈 {complexity}")
    

if __name__ == "__main__":
    recursive_subgraph_example_1()