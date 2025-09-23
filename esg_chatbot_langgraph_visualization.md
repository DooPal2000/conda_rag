

# ESG 챗봇 LangGraph 아키텍처 시각화

## 1. 환경 설정 및 라이브러리 임포트

```python
# 기본 라이브러리
import os
import json
from typing import Dict, List, Any, Optional, Literal
from datetime import datetime
import logging
from dotenv import load_dotenv

# LangChain & LangGraph
from langchain_openai import ChatOpenAI
from langchain.tools import tool
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from typing_extensions import TypedDict
from typing import Annotated

# 시각화
from IPython.display import Image, display

# 환경 변수 로드
load_dotenv()

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
```



## 2. ESG 챗봇 State 정의

```python
class ESGAgentState(TypedDict):
    """ESG 챗봇의 상태를 정의하는 TypedDict"""
    messages: Annotated[List, add_messages]
    query: str
    intent: str # data_query, report_generation, analysis_request, general_query
    cmp_num: Optional[str] # company_id -> cmp_num으로 변경
    company_context: Dict[str, Any]
    esg_data_summary: Dict[str, Any]
    tool_results: Dict[str, Any]
    response_content: str
    needs_data_collection: bool
    report_generated: bool
    session_id: str
    iteration_count: int
    ui_context: Dict[str, Any]  # UI 컨텍스트 추가

print("✅ ESGAgentState 정의 완료")
```



## 3. 워크플로우 노드 함수들 정의

```python
# 간단한 더미 노드 함수들 (실제 구현 대신 시각화용)
def analyze_intent(state: ESGAgentState) -> ESGAgentState:
    """사용자 의도 분석"""
    print("🔍 사용자 의도 분석 중...")
    return state

def load_company_context(state: ESGAgentState) -> ESGAgentState:
    """회사 컨텍스트 로드"""
    print("🏢 회사 정보 로드 중...")
    return state

def check_data_availability(state: ESGAgentState) -> ESGAgentState:
    """ESG 데이터 가용성 확인"""
    print("📊 ESG 데이터 확인 중...")
    return state

def execute_esg_tools(state: ESGAgentState) -> ESGAgentState:
    """ESG 도구 실행"""
    print("🛠️ ESG 도구 실행 중...")
    return state

def generate_response(state: ESGAgentState) -> ESGAgentState:
    """응답 생성"""
    print("💬 응답 생성 중...")
    return state

def save_conversation(state: ESGAgentState) -> ESGAgentState:
    """대화 저장"""
    print("💾 대화 저장 중...")
    return state

def handle_no_data(state: ESGAgentState) -> ESGAgentState:
    """데이터 없음 처리"""
    print("❌ 데이터 없음 처리...")
    return state

def handle_no_company(state: ESGAgentState) -> ESGAgentState:
    """회사 미선택 처리"""
    print("🚫 회사 미선택 처리...")
    return state

# 조건부 분기 함수들
def decide_company_check(state: ESGAgentState) -> Literal["has_company", "no_company"]:
    """회사 선택 여부 확인"""
    return "has_company" if state.get("cmp_num") else "no_company"

def decide_data_availability(state: ESGAgentState) -> Literal["has_data", "no_data"]:
    """데이터 가용성에 따른 분기"""
    return "no_data" if state.get("needs_data_collection", True) else "has_data"

print("✅ 모든 노드 함수 정의 완료")
```



## 4. ESG 챗봇 LangGraph 아키텍처 생성 및 시각화

```python
# 그래프 생성을 위한 StateGraph 객체를 정의
builder = StateGraph(ESGAgentState)

# 각 노드를 초기화
builder.add_node("analyze_intent", analyze_intent)                    # 의도 분석 노드
builder.add_node("load_company_context", load_company_context)        # 회사 컨텍스트 로드 노드
builder.add_node("check_data_availability", check_data_availability)  # 데이터 확인 노드
builder.add_node("execute_esg_tools", execute_esg_tools)              # ESG 도구 실행 노드
builder.add_node("generate_response", generate_response)              # 응답 생성 노드
builder.add_node("save_conversation", save_conversation)              # 대화 저장 노드
builder.add_node("handle_no_data", handle_no_data)                    # 데이터 없음 처리 노드
builder.add_node("handle_no_company", handle_no_company)              # 회사 미선택 처리 노드

# 그래프 로직 정의 (엣지 연결)
builder.add_edge(START, "analyze_intent")

# 조건부 분기: 회사 선택 여부 확인
builder.add_conditional_edges(
    "analyze_intent",
    decide_company_check,
    {
        "has_company": "load_company_context",
        "no_company": "handle_no_company"
    }
)

builder.add_edge("handle_no_company", END)
builder.add_edge("load_company_context", "check_data_availability")

# 조건부 분기: 데이터 가용성 확인
builder.add_conditional_edges(
    "check_data_availability",
    decide_data_availability,
    {
        "has_data": "execute_esg_tools",
        "no_data": "handle_no_data"
    }
)

builder.add_edge("handle_no_data", "save_conversation")
builder.add_edge("execute_esg_tools", "generate_response")
builder.add_edge("generate_response", "save_conversation")
builder.add_edge("save_conversation", END)

# 그래프 컴파일
esg_chatbot_graph = builder.compile()

print("✅ ESG 챗봇 그래프 생성 완료!")
print("📊 노드 수:", len(builder._nodes))
print("🔗 엣지 수:", len(builder._edges))
```



## 5. 그래프 아키텍처 시각화

```python
try:
    # 그래프 시각화 (Mermaid 형식의 PNG로)
    print("🎨 ESG 챗봇 LangGraph 아키텍처 시각화")
    display(Image(esg_chatbot_graph.get_graph().draw_mermaid_png()))
    print("✅ 그래프 시각화 완료!")
except Exception as e:
    print(f"❌ 시각화 오류: {e}")
    print("💡 대신 텍스트 형태로 그래프 구조를 출력합니다:")

    # 그래프 구조를 텍스트로 출력
    print("\n📋 ESG 챗봇 워크플로우 구조:")
    print("START")
    print("  ↓")
    print("analyze_intent (의도 분석)")
    print("  ↓")
    print("조건부 분기: 회사 선택 여부")
    print("  ├─ has_company → load_company_context (회사 컨텍스트 로드)")
    print("  └─ no_company → handle_no_company (회사 미선택 처리) → END")
    print("                    ↓")
    print("              check_data_availability (데이터 확인)")
    print("                    ↓")
    print("              조건부 분기: 데이터 가용성")
    print("                ├─ has_data → execute_esg_tools (ESG 도구 실행)")
    print("                └─ no_data → handle_no_data (데이터 없음 처리)")
    print("                              ↓                    ↓")
    print("                    generate_response         save_conversation")
    print("                         (응답 생성)              (대화 저장)")
    print("                              ↓                    ↓")
    print("                         save_conversation        END")
    print("                            (대화 저장)")
    print("                              ↓")
    print("                             END")
```



## 6. 그래프 구조 상세 정보

```python
# 노드 정보
print("📋 ESG 챗봇 노드 목록:")
for i, (node_name, _) in enumerate(builder._nodes.items(), 1):
    print(f"  {i}. {node_name}")

print("\n🔗 엣지 연결 정보:")
for i, edge in enumerate(builder._edges, 1):
    if hasattr(edge, 'source') and hasattr(edge, 'target'):
        print(f"  {i}. {edge.source} → {edge.target}")

print("\n🎯 조건부 분기점:")
print("  1. analyze_intent: 회사 선택 여부에 따라 분기")
print("  2. check_data_availability: ESG 데이터 가용성에 따라 분기")

print("\n📊 워크플로우 특징:")
print("  - 총 8개 노드로 구성")
print("  - 2개의 조건부 분기점")
print("  - ESG 데이터 처리에 특화된 구조")
print("  - 회사 선택 및 데이터 검증 단계 포함")
print("  - 대화 저장 및 세션 관리 기능")
```



## 7. 간단한 테스트 실행

```python
# 테스트용 상태 생성
test_state = {
    "messages": [HumanMessage(content="ESG 보고서를 생성해주세요")],
    "query": "ESG 보고서를 생성해주세요",
    "intent": "",
    "cmp_num": "6182618882",  # 더미 회사 코드
    "company_context": {},
    "esg_data_summary": {},
    "tool_results": {},
    "response_content": "",
    "needs_data_collection": False,
    "report_generated": False,
    "session_id": "test_session_001",
    "iteration_count": 0,
    "ui_context": {"selected_category": "all", "selected_period": "current_year"}
}

print("🧪 테스트 상태 생성 완료")
print("📝 테스트 쿼리:", test_state["query"])
print("🏢 회사 코드:", test_state["cmp_num"])
print("🎯 UI 컨텍스트:", test_state["ui_context"])

# 실제 실행하려면 아래 주석을 해제하세요 (DB 연결 필요)
# result = esg_chatbot_graph.invoke(test_state)
# print("✅ 테스트 실행 완료:", result)

print("\n💡 실제 실행을 위해서는 다음이 필요합니다:")
print("  - OpenAI API 키 설정")
print("  - 데이터베이스 연결")
print("  - ESG 데이터 처리 모듈")
print("  - 보고서 생성 템플릿")
```


## 8. 추가 정보

### ESG 챗봇 주요 기능:
- **의도 분석**: 사용자 질문의 의도를 파악 (데이터 조회, 보고서 생성, 분석 요청 등)
- **회사 컨텍스트**: 선택된 회사의 기본 정보 로드
- **데이터 검증**: ESG 데이터 가용성 확인
- **도구 실행**: ESG 전용 도구들 (데이터 조회, 트렌드 분석, 보고서 생성)
- **응답 생성**: LLM 기반 전문적인 ESG 분석 응답
- **세션 관리**: 대화 히스토리 저장 및 관리

### 아키텍처 특징:
- **모듈화된 구조**: 각 기능별로 독립적인 노드로 구성
- **조건부 분기**: 상황에 따른 유연한 워크플로우 제어
- **오류 처리**: 데이터 부족, 회사 미선택 등 예외 상황 처리
- **UI 통합**: 웹 인터페이스와의 연동을 위한 컨텍스트 지원

이 구조를 통해 중소기업의 ESG 보고서 작성과 데이터 분석을 효율적으로 지원할 수 있습니다.
