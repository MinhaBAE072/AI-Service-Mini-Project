# AI 스타트업 비교 평가 시스템 (Startup Evaluator v1.4)

본 프로젝트는 AI 스타트업에 대한 심층적이고 객관적인 비교 분석 정보를 제공하여, 투자자 및 이해관계자의 의사결정을 지원하는 "Startup Evaluator v1.4" 에이전트를 설계하고 구현한 실습 프로젝트입니다.

## Overview

-   **Objective**: 사용자가 지정한 복수의 AI 스타트업에 대해 웹 정보 수집, RAG를 통한 문서 참조, 평가 항목별 전문 에이전트 분석을 수행하여, 개별 분석 결과, 비교 분석 결과, 그리고 종합 보고서를 생성함으로써 투자 의사결정을 지원합니다.
-   **Methods**: LangGraph 기반 멀티 에이전트 시스템 (Supervisor 패턴 적용), RAG (Retrieval Augmented Generation), 웹 스크래핑 (Tavily API 활용), LLM 기반 상세 항목 분석 및 보고서 생성.
-   **Tools**: Python, LangGraph, LangChain, OpenAI GPT-4o-mini, Tavily Search API, FAISS (벡터 스토어), PyPDFLoader.

## Features

-   **다중 스타트업 동시 비교 분석**: 최소 2개 이상의 스타트업을 입력받아 투자 매력도를 병렬적으로 심층 비교 분석합니다.
-   **자동화된 정보 수집 및 상세 분석**: 웹 및 제공된 PDF 문서(`Artificial General Intelligence, Intelligent Agents, Voice Intelligence.pdf`)에서 정보를 자동 수집하고, 기술, 시장, 팀, 사업 모델 등 다각적인 기준으로 상세 평가합니다.
-   **신뢰도 기반 리포팅 및 맞춤형 보고서 생성**: 각 분석 항목에 대한 AI의 분석 신뢰도 점수를 제공하며, 최종적으로 사용자 정의된 프롬프트 템플릿에 따라 종합적인 Markdown 보고서를 생성합니다.
-   **동적 평가 기준 적용 가능성**: 사용자가 평가 기준의 가중치를 조절할 수 있는 아키텍처적 유연성을 고려합니다. (구현됨: `main.py`의 `user_defined_criteria_weights`를 통해 `DEFAULT_CRITERIA_WEIGHTS` 오버라이드 가능)

## 주요 아키텍처 개선 및 고도화 포인트

본 시스템은 기존 단일 스타트업 순차 평가 방식에서 발전하여 다음과 같은 주요 고도화 및 아키텍처 개선 사항을 포함합니다:

1.  **다중 스타트업 비교 분석 기능 확장**:
    * 여러 스타트업 정보를 동시에 입력받아, 각 스타트업에 대한 개별 분석 후 그 결과를 심층적으로 비교하는 아키텍처를 채택했습니다.
    * `Supervisor` 역할을 하는 로직(StateGraph의 흐름 제어)을 통해 개별 스타트업 분석 루프를 관리하고, 모든 분석이 완료되면 비교 분석 단계로 전환합니다.

2.  **투자 평가 기준의 상세화 및 동적 가중치 적용**:
    * 기존 평가 기준(창업자, 시장성, 제품/기술력, 경쟁 우위, 실적, 투자조건)을 구체적인 하위 항목으로 상세화하여 각 전문 분석 에이전트가 평가하도록 설계했습니다.
    * 사용자가 평가 기준의 가중치를 동적으로 조절할 수 있는 기능을 상태(`AdvancedStartupEvaluationState`의 `user_defined_criteria_weights`)에 반영하여, `main.py`의 `calculate_weighted_score` 함수에서 사용자 정의 가중치 또는 기본 가중치를 사용해 종합 점수를 계산합니다.

3.  **보고서의 질적 향상 및 분석 신뢰도 명시**:
    * 최종 보고서에 스타트업 간 구체적인 비교 분석 내용, 각 분석 근거 데이터의 출처(`SourceInfo` 활용)를 포함합니다.
    * 각 개별 분석 항목(기술, 시장, 팀, 사업모델)의 결과(`TechEvaluationOutput` 등)에는 `evaluation_confidence_score`를 명시하여 AI 분석의 신뢰도를 제공합니다. 이는 보고서 생성 시 활용됩니다.

4.  **Supervisor 패턴을 통한 워크플로우 조정**:
    * LangGraph의 StateGraph를 활용하여 전체 워크플로우를 효과적으로 조정하고, `InfoScraperAgent`부터 `ReportGeneratorAgent`까지 각 전문 에이전트의 역할과 데이터 흐름을 명확히 했습니다.
    * 조건부 엣지(`should_continue_processing`)를 통해 개별 스타트업 분석 완료 여부를 판단하고 다음 단계로의 전환을 자동화합니다.

5.  **모듈화된 에이전트 및 도구 사용**:
    * 정보 수집, 기술 분석, 시장 분석 등 각 기능을 독립적인 에이전트 모듈로 분리하여 개발 및 유지보수 용이성을 높였습니다.
    * 웹 검색(Tavily), RAG(FAISS, PyPDF) 등 외부 도구를 효과적으로 통합하여 분석의 깊이와 범위를 확장했습니다.

## Tech Stack

| Category        | Details                                                                 |
| :-------------- | :---------------------------------------------------------------------- |
| Framework       | LangGraph, LangChain                                                    |
| Language        | Python                                                                  |
| LLM             | OpenAI GPT-4o-mini                                                      |
| Retrieval       | FAISS (Vector Store for RAG), PyPDFLoader                               |
| Web Search      | Tavily Search API                                                       |
| Core Libraries  | Pydantic (State/Output definition), python-dotenv (Environment management) |

## Agents

-   **`SupervisorAgent` (Conceptual, implemented in `main.py`)**: 전체 워크플로우 조정, 태스크 분배, 결과 취합 및 다음 단계 결정 로직 수행.
-   **`InfoScraperAgent` (`agents/info_scraper_agent.py`)**: 웹 검색(Tavily) 및 RAG(PDF 문서)를 통해 각 스타트업의 기본 정보를 수집합니다.
-   **`TechEvaluationAgent` (`agents/tech_eval_agent.py`)**: 수집된 정보를 바탕으로 스타트업의 기술 및 제품(독창성, 구현 가능성, 특허 등)을 분석합니다.
-   **`MarketEvaluationAgent` (`agents/market_eval_agent.py`)**: 목표 시장의 규모, 성장성, 경쟁 환경 등을 분석합니다.
-   **`TeamEvaluationAgent` (`agents/team_eval_agent.py`)**: 창업자 및 핵심 팀원의 전문성, 경험, 실행력 등을 분석합니다.
-   **`BizModelEvaluationAgent` (`agents/biz_model_eval_agent.py`)**: 수익 모델, 고객 확보 전략, 실적, 투자 조건 등을 분석합니다.
-   **`ComparativeAnalysisAgent` (`agents/comparative_analysis_agent.py`)**: 다수 스타트업의 개별 분석 결과를 종합하여 항목별 비교 분석 및 요약 결과를 생성합니다.
-   **`ReportGeneratorAgent` (`agents/report_generator_agent.py`)**: 모든 분석 결과를 바탕으로 정의된 템플릿(`prompts/ai_mini_design_20_3반_배민하_prompt.md`)에 맞춰 최종 비교 평가 보고서를 생성합니다.

## State (`states/ai_mini_design_20_3반_배민하_state_py.py`)

-   **`startup_names_to_compare`** (`List[str]`): 사용자가 입력한 비교 분석 대상 스타트업 이름 목록.
-   **`user_defined_criteria_weights`** (`Optional[Dict[str, float]]`): (선택적) 사용자가 정의한 평가 기준별 가중치.
-   **`initial_scraped_data_all_startups`** (`List[Dict[str, Any]]`): `InfoScraperAgent`가 각 스타트업별로 수집한 원시 정보 (출처 URL, 스니펫 등).
-   **`individual_detailed_analyses`** (`List[SingleStartupDetailedAnalysis]`): 각 전문 분석 에이전트가 개별 스타트업에 대해 수행한 상세 분석 결과 (기술, 시장, 팀, 사업모델 평가, SWOT, 가중 점수 등 포함) 리스트.
-   **`comparative_analysis_output`** (`Dict[str, Any]`): `ComparativeAnalysisAgent`가 생성한 스타트업 간 비교 분석 결과 (종합 요약, 항목별 비교 데이터, 투자 제언 등).
-   **`final_comparison_report_text`** (`str`): `ReportGeneratorAgent`가 생성한 최종 보고서의 Markdown 텍스트.
-   **`current_startup_index`** (`Optional[int]`): 현재 처리 중인 스타트업의 인덱스 (개별 분석 루프 관리용).
-   **`current_startup_name`** (`Optional[str]`): 현재 처리 중인 스타트업의 이름.
-   **`error_log`** (`List[str]`): 워크플로우 실행 중 발생한 오류 기록.

## Architecture

(시스템 아키텍처 다이어그램은 `1일차/ai_mini_design_20_3반_배민하_graph.png` 이미지 파일 또는 `1일차/ai_mini_design_20_3반_배민하_graph.gv` DOT 코드를 통해 확인할 수 있습니다. LangGraph의 `app.get_graph().print_ascii()`를 통해서도 텍스트 기반 시각화가 가능합니다.)

## Directory Structure
AI-Service-Mini-Project/
├── 1일차/                     # 초기 설계 산출물
│   ├── ai_mini_design_20_3반_배민하_graph.gv
│   ├── ai_mini_design_20_3반_배민하_prompt.md
│   ├── ai_mini_design_20_3반_배민하_state_py.py
│   └── ai_mini_design_20_3반_배민하_template.md
├── 2일차/
│   ├── agents/               # 평가 기준별 Agent 모듈
│   ├── data/                 # RAG용 PDF 문서 (Startup Info)
│   ├── outputs/              # 최종 평가 보고서 저장
│   ├── prompts/              # ReportGeneratorAgent용 프롬프트 템플릿
│   ├── states/               # LangGraph 상태 정의
│   ├── tools/                # 웹 검색, RAG 도구 모듈
│   ├── main.py               # 메인 실행 스크립트 (LangGraph 워크플로우 정의)
│   └── requirements.txt      # Python 패키지 의존성
├── README.md                 # 프로젝트 안내 문서 (본 파일)
└── ai_mini_3반_배민하_보고서(pdf).pdf # 프로젝트 최종 보고서 (PDF 버전)

