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

본 README 섹션에서는 스타트업 평가 시스템의 주요 개선 사항을 기존 방식(Before)과 개선된 방식(After)으로 나누어 설명합니다.

## Before: 기존 시스템

* **평가 방식:**
    * 한 번에 하나의 스타트업 정보만 입력받아 분석 및 평가를 진행했습니다.
    * 여러 스타트업을 비교하기 위해서는 각 스타트업에 대한 평가를 개별적으로 완료한 후, 수동으로 결과를 취합하여 비교해야 했습니다.
* **평가 기준:**
    * 창업자, 시장성, 제품/기술력, 경쟁 우위, 실적, 투자조건 등과 같은 개괄적인 평가 기준을 사용했습니다.
    * 각 평가 기준의 하위 항목이 구체적으로 정의되지 않았거나, 평가 가중치가 고정되어 있어 사용자의 관점을 유연하게 반영하기 어려웠습니다.
* **보고서 및 신뢰도:**
    * 생성되는 보고서는 주로 개별 스타트업에 대한 분석 결과를 요약하는 수준이었습니다.
    * 분석 근거 데이터의 출처가 명확히 제시되지 않거나, AI 분석 결과의 신뢰도를 구체적인 점수로 제공하지 않았습니다.
* **워크플로우 및 아키텍처:**
    * 전체 흐름을 체계적으로 관리하는 제어 로직이 부족했습니다.
    * 외부 정보 수집 도구(웹 검색, 문서 분석 등)의 통합 및 활용이 제한적이었습니다.

## After: 개선된 시스템

새로운 시스템은 다중 스타트업 비교 분석, 평가 기준의 상세화 및 동적 가중치 적용, 보고서 품질 향상, 그리고 모듈화된 아키텍처를 통해 기존 시스템의 한계를 극복하고 사용자에게 더욱 심층적이고 유연한 평가 경험을 제공합니다.

### 주요 고도화 및 아키텍처 개선 사항:

1.  **다중 스타트업 비교 분석 기능 확장:**
    * **동시 입력 및 비교:** 여러 스타트업 정보를 동시에 입력받아, 각 스타트업에 대한 개별 분석 후 그 결과를 시스템 내에서 심층적으로 비교하는 아키텍처를 채택했습니다.
    * **체계적 흐름 관리:** `Supervisor` 역할을 하는 로직(LangGraph의 `StateGraph` 흐름 제어)을 통해 개별 스타트업 분석 루프를 효율적으로 관리하고, 모든 분석이 완료되면 자동으로 비교 분석 단계로 전환합니다.

2.  **투자 평가 기준의 상세화 및 동적 가중치 적용:**
    * **상세 하위 항목:** 기존 평가 기준(창업자, 시장성, 제품/기술력, 경쟁 우위, 실적, 투자조건)을 구체적인 하위 항목으로 상세화하여, 각 전문 분석 에이전트가 세분화된 평가를 수행하도록 설계했습니다.
    * **사용자 정의 가중치:** 사용자가 평가 기준의 중요도에 따라 가중치를 동적으로 조절할 수 있는 기능을 상태 객체(`AdvancedStartupEvaluationState`의 `user_defined_criteria_weights`)에 반영했습니다. 이를 통해 `main.py`의 `calculate_weighted_score` 함수에서 사용자 정의 가중치 또는 기본 가중치를 사용하여 종합 점수를 계산합니다.

3.  **보고서의 질적 향상 및 분석 신뢰도 명시:**
    * **심층 비교 보고서:** 최종 보고서에 스타트업 간 구체적인 비교 분석 내용을 포함하여, 사용자가 각 스타트업의 강점과 약점을 명확히 파악할 수 있도록 합니다.
    * **데이터 출처 투명성:** 각 분석 근거 데이터의 출처를 (`SourceInfo` 객체 활용) 명시하여 보고서 내용의 신뢰성을 높였습니다.
    * **AI 분석 신뢰도 제공:** 각 개별 분석 항목(기술, 시장, 팀, 사업모델 등)의 결과(`TechEvaluationOutput` 등)에 `evaluation_confidence_score`를 명시하여 AI 분석 자체의 신뢰도를 제공하고, 이는 보고서 생성 시 활용됩니다.

4.  **Supervisor 패턴을 통한 워크플로우 조정:**
    * **효과적인 워크플로우 관리:** LangGraph의 `StateGraph`를 활용하여 `InfoScraperAgent`부터 `ReportGeneratorAgent`까지 각 전문 에이전트의 역할과 데이터 흐름을 명확히 하고, 전체 워크플로우를 효과적으로 조정합니다.
    * **자동화된 단계 전환:** 조건부 엣지(`should_continue_processing` 로직)를 통해 개별 스타트업 분석 완료 여부를 판단하고, 다음 단계(추가 분석 또는 비교 분석)로의 전환을 자동화하여 프로세스 효율성을 증대시켰습니다.

5.  **모듈화된 에이전트 및 도구 사용:**
    * **개발 및 유지보수 용이성:** 정보 수집, 기술 분석, 시장 분석 등 각 기능을 독립적인 에이전트 모듈로 분리하여 개발 및 유지보수의 용이성을 크게 향상시켰습니다.
    * **분석 깊이 및 범위 확장:** 웹 검색(Tavily), RAG(Retrieval Augmented Generation - FAISS, PyPDF 활용) 등 외부 도구를 효과적으로 통합하여 분석의 깊이와 범위를 확장하고, 더욱 풍부한 정보를 바탕으로 평가를 수행합니다.

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

## State 

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

![workflow_manual](https://github.com/user-attachments/assets/7605b046-1976-4ede-994a-09c0f42132e6)

## Directory Structure

```
AI-Service-Mini-Project/
├── README.md
├── requirements.txt
├── main.py
├── ai_mini_design_20_3반_배민하_graph.gv
├── ai_mini_design_20_3반_배민하_template.md
├── final_ai_startup_comparison_report.md
├── data/
│   └── Artificial General Intelligence, Intelligent Agents, Voice Intelligence.pdf
├── prompts/
│   └── ai_mini_design_20_3반_배민하_prompt.md
├── states/
│   └── ai_mini_design_20_3반_배민하_state_py.py
├── tools/
│   ├── rag_tool.py
│   └── web_search_tool.py
└── agents/
    ├── biz_model_eval_agent.py
    ├── comparative_analysis_agent.py
    ├── info_scraper_agent.py
    ├── market_eval_agent.py
    ├── report_generator_agent.py
    ├── team_eval_agent.py
    └── tech_eval_agent.py
```
