# AI-Service-Mini-Project-main/agents/market_eval_agent.py
import sys
import os
import json
from typing import Dict, List, Any, Optional

# pydantic v1 또는 v2에 따라 import 방식이 다를 수 있습니다.
# 프로젝트의 다른 파일들과 일관성을 맞추기 위해 langchain_core.pydantic_v1을 사용할 수 있으나,
# 다른 에이전트에서 pydantic을 직접 사용했다면 동일하게 맞춰주는 것이 좋습니다.
# 여기서는 Pydantic v2를 직접 사용하는 것으로 가정합니다. (필요시 v1으로 변경: from langchain_core.pydantic_v1 import BaseModel, Field)
from pydantic import BaseModel, Field


PROJECT_ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if PROJECT_ROOT_DIR not in sys.path:
    sys.path.append(PROJECT_ROOT_DIR)

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
# from langchain_core.pydantic_v1 import BaseModel, Field # Pydantic v1 for LangChain if preferred

from states.ai_mini_design_20_3반_배민하_state_py import MarketEvaluationOutput, SourceInfo
from tools.rag_tool import RAGTool
from tools.web_search_tool import search_web_for_startup # 추가 웹 검색을 위해

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.2)
rag_tool_instance = RAGTool()

# LLM이 반환할 구조 정의
class LLMMarketEvaluationResponse(BaseModel):
    market_size_score: Optional[float] = Field(description="목표 시장 크기 점수 (1-5점 척도)")
    market_size_comment: str = Field(description="목표 시장 크기 및 매력도에 대한 상세 코멘트 및 근거")
    growth_potential_score: Optional[float] = Field(description="시장 성장 가능성 점수 (1-5점 척도)")
    growth_potential_comment: str = Field(description="시장 성장 가능성에 대한 상세 코멘트 및 근거")
    competitive_landscape_summary: str = Field(description="주요 경쟁사 및 경쟁 환경 요약")
    entry_barriers_comment: Optional[str] = Field(description="시장 진입 장벽에 대한 분석 코멘트 (높음/중간/낮음 및 이유)")
    network_effects_comment: Optional[str] = Field(description="네트워크 효과 존재 여부 및 영향력 분석 코멘트")
    evaluation_confidence_score: Optional[float] = Field(description="이 시장 평가 전반에 대한 AI의 분석 신뢰도 점수 (0.0 ~ 1.0 사이).")


MARKET_EVAL_PROMPT_TEMPLATE = """
당신은 AI 스타트업의 시장성 및 경쟁 환경을 심층 분석하는 시장 분석 전문가입니다.
제공된 '{startup_name}' 스타트업에 대한 다음 정보를 바탕으로 시장성을 평가해주십시오.

[제공된 정보]
{information_context}

[평가 항목]
1.  **목표 시장 크기 및 매력도**: 이 스타트업이 타겟하는 시장의 현재 규모와 전반적인 매력도를 분석하고 1점에서 5점 사이로 점수를 매겨주세요. 근거를 상세히 설명해주세요. (예: TAM, SAM, SOM 언급 가능)
2.  **시장 성장 가능성**: 해당 시장의 미래 성장 잠재력은 어느 정도인지, 주요 성장 동인은 무엇인지 평가하고 1점에서 5점 사이로 점수를 매겨주세요. 근거를 상세히 설명해주세요.
3.  **경쟁 환경**: 주요 경쟁사들은 누구이며, 그들의 강점과 약점은 무엇입니까? 시장의 경쟁 강도는 어느 정도입니까?
4.  **진입 장벽**: 이 시장의 주요 진입 장벽은 무엇이며, {startup_name}이 이를 어떻게 극복할 수 있을지 분석해주세요. (예: 기술, 자본, 규제, 브랜드 인지도 등)
5.  **네트워크 효과**: 해당 시장 또는 {startup_name}의 사업 모델에서 네트워크 효과가 존재한다면, 그 영향력은 어느 정도일지 분석해주세요.
6.  **분석 신뢰도**: 위 시장 평가 항목들에 대한 당신의 전반적인 분석 신뢰도를 0.0에서 1.0 사이의 점수로 매겨주십시오. (예: 정보가 충분하고 명확하면 0.9, 정보가 부족하거나 모호하면 0.5)

[출력 형식 지침]
다음 JSON 형식을 반드시 준수하여 답변해주십시오:
{{
    "market_size_score": (float, 예: 4.0 또는 null),
    "market_size_comment": "(string)",
    "growth_potential_score": (float, 예: 4.5 또는 null),
    "growth_potential_comment": "(string)",
    "competitive_landscape_summary": "(string)",
    "entry_barriers_comment": "(string, 예: '기술적 진입 장벽 높음. 하지만 XXX로 극복 가능' 또는 null)",
    "network_effects_comment": "(string, 예: '강력한 네트워크 효과 존재. 사용자 증가가 가치 증대로 이어짐' 또는 null)",
    "evaluation_confidence_score": (float, 예: 0.85)
}}

스타트업 이름: {startup_name}
"""

def run_market_evaluator(startup_name: str, scraped_info: List[SourceInfo], use_rag: bool = True, use_web_search: bool = True) -> MarketEvaluationOutput:
    print(f"--- MarketEvaluationAgent 실행 시작: {startup_name} ---")

    information_snippets = []
    all_sources_for_this_evaluation: List[SourceInfo] = []

    # 1. InfoScraper가 제공한 웹 스크랩 정보 컨텍스트화
    for src_info in scraped_info:
        updated_src_info = src_info.copy()
        if 'ai_confidence_score' not in updated_src_info :
             updated_src_info['ai_confidence_score'] = None
        information_snippets.append(f"기존 수집 정보 ({updated_src_info['source_url_or_doc']}, 신뢰도: {updated_src_info.get('ai_confidence_score', 'N/A')}):\n{updated_src_info['retrieved_content_snippet']}\n---")
        all_sources_for_this_evaluation.append(updated_src_info)

    # 2. (선택적) 추가 웹 검색으로 시장 정보 보강
    if use_web_search:
        market_queries = [
            f"{startup_name} 시장 규모 및 전망",
            f"{startup_name} 타겟 고객",
            f"{startup_name} 경쟁사 분석",
            f"{startup_name} 산업 동향"
        ]
        for query in market_queries:
            web_results = search_web_for_startup(query)
            for res in web_results:
                source_info_entry = SourceInfo(
                    source_url_or_doc=res['source_url_or_doc'],
                    retrieved_content_snippet=res['retrieved_content_snippet'],
                    ai_confidence_score=None # 웹 검색 결과에 대한 신뢰도는 별도 평가 필요
                )
                information_snippets.append(f"웹 검색 ({res['source_url_or_doc']}, 신뢰도: N/A):\n{res['retrieved_content_snippet']}\n---")
                all_sources_for_this_evaluation.append(source_info_entry)


    # 3. (선택적) RAG를 통한 PDF 문서 정보 보강
    if use_rag:
        rag_query = f"{startup_name}의 시장성, 목표 시장, 경쟁 환경, 성장 잠재력에 대한 정보"
        pdf_results = rag_tool_instance.search_documents(rag_query, k=2)
        for res in pdf_results:
            rag_source_info: SourceInfo = {
                "source_url_or_doc": f"{res['source_document']} (Page: {res['source_page']})",
                "retrieved_content_snippet": res["content"],
                "ai_confidence_score": None # RAG 결과에 대한 신뢰도도 추후 평가 가능
            }
            information_snippets.append(f"문서 출처 ({rag_source_info['source_url_or_doc']}, 신뢰도: N/A):\n{res['content']}\n---")
            all_sources_for_this_evaluation.append(rag_source_info)

    if not information_snippets:
        print(f"{startup_name}에 대한 시장 분석 정보가 부족합니다.")
        return MarketEvaluationOutput(
            market_size_score=None, market_size_comment="분석 정보 부족",
            growth_potential_score=None, growth_potential_comment="분석 정보 부족",
            competitive_landscape_summary="분석 정보 부족",
            entry_barriers_comment="분석 정보 부족",
            network_effects_comment="분석 정보 부족",
            evaluation_confidence_score=0.0, # 정보 부족 시 신뢰도 0
            sources=[]
        )

    information_context = "\n".join(information_snippets)
    # print(f"LLM에 전달될 시장 컨텍스트 (일부): {information_context[:500]}...")

    prompt = ChatPromptTemplate.from_template(MARKET_EVAL_PROMPT_TEMPLATE)
    chain = prompt | llm

    try:
        response_str = chain.invoke({
            "startup_name": startup_name,
            "information_context": information_context
        }).content
        print(f"LLM 응답 (문자열): {response_str}")

        # 마크다운 코드 블록 제거 로직
        if response_str.strip().startswith("```json"):
            response_str = response_str.strip()[7:]
            if response_str.strip().endswith("```"):
                response_str = response_str.strip()[:-3]
        elif response_str.strip().startswith("```"):
             response_str = response_str.strip()[3:]
             if response_str.strip().endswith("```"):
                response_str = response_str.strip()[:-3]
        
        response_str = response_str.strip() # 추가: 앞뒤 공백 제거

        try:
            llm_response_data = json.loads(response_str)
            market_eval_output = MarketEvaluationOutput(
                market_size_score=llm_response_data.get("market_size_score"),
                market_size_comment=llm_response_data.get("market_size_comment", "파싱 오류 또는 값 없음"),
                growth_potential_score=llm_response_data.get("growth_potential_score"),
                growth_potential_comment=llm_response_data.get("growth_potential_comment", "파싱 오류 또는 값 없음"),
                competitive_landscape_summary=llm_response_data.get("competitive_landscape_summary", "파싱 오류 또는 값 없음"),
                entry_barriers_comment=llm_response_data.get("entry_barriers_comment"),
                network_effects_comment=llm_response_data.get("network_effects_comment"),
                evaluation_confidence_score=llm_response_data.get("evaluation_confidence_score"), # 신뢰도 점수 추가
                sources=all_sources_for_this_evaluation
            )
        except json.JSONDecodeError as e:
            print(f"LLM 응답 JSON 파싱 오류: {e}")
            print(f"오류 발생한 응답: {response_str}")
            market_eval_output = MarketEvaluationOutput(
                market_size_score=None, market_size_comment=f"LLM 응답 파싱 오류: {response_str}",
                growth_potential_score=None, growth_potential_comment=f"LLM 응답 파싱 오류: {response_str}",
                competitive_landscape_summary=f"LLM 응답 파싱 오류: {response_str}",
                entry_barriers_comment="파싱 오류", network_effects_comment="파싱 오류",
                evaluation_confidence_score=0.1, # 파싱 오류 시 낮은 신뢰도
                sources=all_sources_for_this_evaluation
            )
    except Exception as e:
        print(f"LLM 호출 중 오류 발생: {e}")
        market_eval_output = MarketEvaluationOutput(
            market_size_score=None, market_size_comment=f"LLM 호출 오류: {e}",
            growth_potential_score=None, growth_potential_comment=f"LLM 호출 오류: {e}",
            competitive_landscape_summary=f"LLM 호출 오류: {e}",
            entry_barriers_comment="LLM 호출 오류", network_effects_comment="LLM 호출 오류",
            evaluation_confidence_score=0.1, # LLM 호출 오류 시 낮은 신뢰도
            sources=all_sources_for_this_evaluation
        )

    print(f"--- MarketEvaluationAgent 실행 완료: {startup_name} ---")
    print(f"분석 결과: {market_eval_output}")
    return market_eval_output

if __name__ == '__main__':
    # MarketEvaluationAgent 직접 실행 테스트
    sample_scraped_data_mago = [
        SourceInfo(source_url_or_doc="https://www.holamago.com/", retrieved_content_snippet="MAGO는 음성 AI 빌더 '오디온'과 음성 AI 기반 정신건강 관리 플랫폼 '카세트'를 개발한 기업입니다. 주요 시장은 디지털 헬스케어 및 음성 AI 솔루션 시장입니다.", ai_confidence_score=0.9),
        SourceInfo(source_url_or_doc="MAGO_market_report.com", retrieved_content_snippet="디지털 정신건강 시장은 연평균 20% 이상 성장하고 있으며, 특히 AI 기반 솔루션의 수요가 증가하고 있습니다. 경쟁사로는 A, B 등이 있습니다.", ai_confidence_score=0.8)
    ]
    sample_scraped_data_plang = [
        SourceInfo(source_url_or_doc="https://theplang.com/", retrieved_content_snippet="The Plan G의 '오딩가 잉글리시'는 초등학생 대상 에듀테크 시장을 공략합니다. 국내외 유사 AI 영어 학습 앱들이 경쟁자로 존재합니다.", ai_confidence_score=0.85),
        SourceInfo(source_url_or_doc="Edutech_market_news.com", retrieved_content_snippet="글로벌 에듀테크 시장은 코로나19 이후 비대면 학습 수요 증가로 빠르게 성장 중입니다. 개인 맞춤형 학습 콘텐츠가 중요해지고 있습니다.", ai_confidence_score=0.75)
    ]

    print("\n--- MAGO 시장성 평가 테스트 ---")
    mago_market_eval = run_market_evaluator(startup_name="MAGO", scraped_info=sample_scraped_data_mago, use_rag=True, use_web_search=True)
    # 상세 결과 출력 (신뢰도 점수 확인)
    # import json
    # print(json.dumps(mago_market_eval, indent=2, ensure_ascii=False))


    print("\n--- The Plan G 시장성 평가 테스트 ---")
    plang_market_eval = run_market_evaluator(startup_name="The Plan G", scraped_info=sample_scraped_data_plang, use_rag=False, use_web_search=True)
    # 상세 결과 출력 (신뢰도 점수 확인)
    # print(json.dumps(plang_market_eval, indent=2, ensure_ascii=False))