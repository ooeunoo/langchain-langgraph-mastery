# LangChain 프롬프트의 기초

프롬프트는 LLM과 효과적으로 소통하기 위한 핵심 도구입니다. 이 섹션에서는 LangChain의 다양한 프롬프트 기법을 학습합니다.

### 핵심 개념

- Prompt Template의 구조와 활용
- Few-Shot Learning을 통한 성능 향상
- 동적 예시 선택과 최적화

---

## 파일 목록 및 학습 순서

### 1. `prompt_templates.py`

**프롬프트 템플릿 기초**

- PromptTemplate 기본 사용법
- 변수 삽입 및 포맷팅
- ChatPromptTemplate 활용
- 템플릿 재사용 패턴

**핵심 내용:**

```python
# 기본 템플릿
template = PromptTemplate.from_template(
    "{topic}에 대해 설명해주세요."
)

# Chat 템플릿
chat_template = ChatPromptTemplate.from_messages([
    ("system", "당신은 {role}입니다."),
    ("human", "{question}")
])
```

### 2. `few_shot_prompts.py`

**Few-Shot Learning**

- 예시 기반 프롬프팅
- FewShotPromptTemplate 사용
- 고정 예시 vs 동적 예시
- 패턴 학습을 통한 정확도 향상

**핵심 내용:**

```python
# Few-Shot 예시
examples = [
    {"input": "행복하다", "output": "긍정적"},
    {"input": "슬프다", "output": "부정적"}
]

few_shot_prompt = FewShotPromptTemplate(
    examples=examples,
    example_prompt=example_prompt,
    suffix="입력: {input}\n감정:"
)
```

### 3. `partial_prompts.py` 

**부분 프롬프트**

- 변수 미리 채우기
- 동적 기본값 설정
- 템플릿 재사용 최적화
- 함수를 통한 동적 값 생성

**핵심 내용:**

```python
# 일부 변수 미리 채우기
partial_template = template.partial(role="AI 전문가")

# 함수로 동적 값 생성
def get_current_date():
    return datetime.now().strftime("%Y-%m-%d")

template = template.partial(date=get_current_date)
```

---

## 주요 개념

### 1. Prompt Template

프롬프트 템플릿은 **재사용 가능한 프롬프트 구조**를 정의합니다.

```
Template: "{role}로서 {topic}을 설명하세요."
Variables: role, topic
```

**장점:**

- 일관된 프롬프트 관리
- 변수를 통한 동적 구성
- 재사용성 향상

### 2. Few-Shot Learning

**예시를 통해 LLM이 패턴을 학습**하도록 합니다.

```
예시 1: 입력 → 출력
예시 2: 입력 → 출력
예시 3: 입력 → 출력
---
새 입력: ? → LLM이 패턴을 따라 출력
```

**효과:**

- Zero-shot보다 높은 정확도
- 특정 포맷 학습
- 복잡한 작업 수행

### 3. Chat Prompts

**역할 기반 대화 구조**를 통해 명확한 컨텍스트 제공:

```
System: 역할 및 규칙 정의
Human: 사용자 입력
AI: 어시스턴트 응답
```

### 4. Example Selection

**상황에 맞는 최적의 예시 선택**:

- **Semantic Similarity**: 의미적으로 유사한 예시
- **Length-Based**: 프롬프트 길이 제한
- **MMR**: 유사성 + 다양성 균형

---

## 💡 프롬프트 엔지니어링 팁

### 1. 명확하고 구체적으로

❌ 나쁜 예:

```python
"AI에 대해 설명해줘"
```

✅ 좋은 예:

```python
"""당신은 기술 전문가입니다.
초보자를 위해 인공지능의 개념을 3가지 핵심 요소로 나누어
각각 예시와 함께 설명해주세요."""
```

### 2. 역할과 컨텍스트 제공

```python
ChatPromptTemplate.from_messages([
    ("system", "당신은 10년 경력의 Python 전문가입니다."),
    ("human", "{question}")
])
```

### 3. Few-Shot 예시 활용

```python
examples = [
    {"input": "happy", "output": "positive"},
    {"input": "sad", "output": "negative"},
]
# 패턴을 학습하여 새로운 입력 처리
```

### 4. 출력 포맷 명시

```python
"""다음 형식으로 답변하세요:
1. 개념 설명
2. 실전 예시
3. 주의사항"""
```

---

## 프롬프트 설계 프로세스

```
1. 목표 정의
   ↓
2. 기본 템플릿 작성
   ↓
3. 변수 식별
   ↓
4. Few-Shot 예시 추가 (필요시)
   ↓
5. 테스트 및 개선
   ↓
6. 프로덕션 적용
```

---

## 성능 비교

| 방식                | 정확도 | 복잡도 | 사용 사례      |
| ------------------- | ------ | ------ | -------------- |
| Zero-Shot           | 낮음   | 낮음   | 간단한 질문    |
| Few-Shot            | 중간   | 중간   | 패턴 학습 필요 |
| Few-Shot + Selector | 높음   | 높음   | 복잡한 도메인  |

---

##  참고 자료

### 공식 문서

- [Prompt Templates](https://python.langchain.com/docs/modules/model_io/prompts/prompt_templates/)
- [Few-Shot Prompting](https://python.langchain.com/docs/modules/model_io/prompts/few_shot_examples/)
- [Example Selectors](https://python.langchain.com/docs/modules/model_io/prompts/example_selectors/)

### 추천 읽을거리

- [Prompt Engineering Guide](https://www.promptingguide.ai/)
- [OpenAI Best Practices](https://platform.openai.com/docs/guides/prompt-engineering)

---
