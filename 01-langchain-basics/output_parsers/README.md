# Output Parsers

## 학습 목표

- LLM 응답을 다양한 형식으로 파싱하는 방법 학습
- 구조화된 데이터 추출 기술 습득
- 프로덕션 환경에서의 Output Parser 활용

### 1. String Parser

**가장 기본적인 파서**

- StrOutputParser 사용법
- 스트리밍 출력
- Batch 처리

### 2. JSON Parser (`json_parser.py`)

**JSON 형식의 구조화된 데이터**

- JsonOutputParser 사용
- 중첩 JSON 처리
- 리스트 형태 JSON
- 에러 핸들링

### 3. Pydantic Parser (`pydantic_parser.py`)

**타입 안전한 파싱**

- PydanticOutputParser
- 자동 유효성 검사
- Optional 필드
- IDE 자동완성

### 4. List Parser (`list_parser.py`)

**간단한 리스트 추출**

- CommaSeparatedListOutputParser
- 키워드 추출
- 카테고리 분류

## 주요 개념

### Parser 선택 가이드

```python
# 1. 간단한 텍스트 응답
parser = StrOutputParser()

# 2. JSON 필요 (유연성)
parser = JsonOutputParser()

# 3. 타입 안전 필요 (프로덕션)
parser = PydanticOutputParser(pydantic_object=MyModel)

# 4. 간단한 리스트
parser = CommaSeparatedListOutputParser()

```

## 핵심 포인트

### 1. Format Instructions

```python
# Parser마다 format_instructions 제공
instructions = parser.get_format_instructions()

# Prompt에 포함 필수!
prompt = ChatPromptTemplate.from_template(
    "Your task...\n\n{format_instructions}"
)
```

## 참고 자료

- [LangChain Output Parsers](https://python.langchain.com/docs/modules/model_io/output_parsers/)
- [Pydantic Documentation](https://docs.pydantic.dev/)
