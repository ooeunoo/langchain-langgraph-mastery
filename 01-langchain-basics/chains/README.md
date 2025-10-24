# LangChain 체인의 기초

### 핵심 개념

- Chain의 기본 구조와 동작 원리
- 다양한 체인 유형과 사용 사례
- 체인 조합 및 커스터마이징
- 실전 예제를 통한 체인 활용

---

## 파일 목록 및 학습 순서

### 1. `simple_chain.py`

**가장 기본적인 체인**

- LLM + Prompt Template의 결합
- Chain의 핵심 개념 이해
- 단일 입력 → 단일 출력

### 2. `sequential_chain.py`

**순차적 체인**

- 여러 체인을 순서대로 연결
- 이전 체인의 출력이 다음 체인의 입력
- 복잡한 워크플로우 구성

### 3. `router_chain.py`

**라우터 체인**

- 입력에 따라 다른 체인으로 라우팅
- 조건부 로직 구현
- 다양한 시나리오 처리

### 4. `transform_chain.py`

**변환 체인**

- 데이터 전처리 및 후처리
- 커스텀 변환 로직
- 파이프라인 구성

### 5. `custom_chain.py`

**커스텀 체인**

- 완전한 커스텀 체인 구현
- 복잡한 비즈니스 로직
- 프로덕션 레벨 체인

---

## 주요 개념

### Chain이란?

Chain은 LangChain의 기본 구성 요소로, **여러 컴포넌트를 연결하여 복잡한 작업을 수행**하는 파이프라인입니다.

```
Input → [Component 1] → [Component 2] → Output
```

### Chain의 구성 요소

1. **Prompt Template**: 입력을 형식화
2. **LLM**: 언어 모델 추론
3. **Output Parser**: 출력을 구조화
4. **Memory** (선택): 대화 기록 저장

### LCEL (LangChain Expression Language)

LangChain 0.1.0부터 도입된 새로운 체인 구성 방식:

```python
chain = prompt | llm | output_parser
```

---

## 참고 자료

- [LangChain Chains 공식 문서](https://python.langchain.com/docs/modules/chains/)
- [LCEL 가이드](https://python.langchain.com/docs/expression_language/)
- [Chain 예제 모음](https://python.langchain.com/docs/modules/chains/how_to/)
