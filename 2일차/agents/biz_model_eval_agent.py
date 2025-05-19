# AI-Service-Mini-Project-main/agents/biz_model_eval_agent.py
import sys
import os
import json
from typing import Dict, List, Any, Optional

# Pydantic v1 또는 v2를 사용합니다. 프로젝트 일관성을 위해 다른 에이전트와 동일하게 설정합니다.
# 여기서는 Pydantic v2를 직접 사용하는 것으로 가정합니다.
from pydantic import BaseModel, Field


# 프로젝트 루트 디렉토리를 sys.path에 추가
PROJECT_ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if PROJECT_ROOT_DIR not in sys.path:
    sys.path.append(PROJECT_ROOT_DIR)

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
# from langchain_core.pydantic_v1 import BaseModel, Field # LangChain과 함께 Pydantic v1을 사용하려면 이 줄을 사용

from states.ai_mini_design_20_3반_배민하_state_py import BizModelEvaluationOutput, SourceInfo
from tools.rag_tool import RAGTool
from tools.web_search_tool import search_web_for_startup

# LLM 초기화
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.2)

# RAG 도구 인스턴스화
rag_tool_instance = RAGTool()

# LLM이 반환할 구조 정의
class LLMBizModelEvaluationResponse(BaseModel):
    track_record_summary: str = Field(description="주요 실적(매출, 계약, 사용자 수 등) 요약 (발견된 정보 기반)")
    track_record_score: Optional[float] = Field(description="실적에 대한 점수 (1-5점 척도, 정보 부족 시 null)")
    deal_terms_comment: Optional[str] = Field(description="투자 조건(Valuation, 지분율 등)에 대한 코멘트 (발견 시, 정보 부족 시 '정보 부족')")
    deal_terms_score: Optional[float] = Field(description="투자 조건 매력도 점수 (1-5점 척도, 정보 부족 시 null)")
    revenue_model_summary: str = Field(description="주요 수익 모델 요약")
    g2m_strategy_summary: Optional[str] = Field(description="주요 고객 확보(Go-to-Market) 전략 요약")
    evaluation_confidence_score: Optional[float] = Field(description="이 사업 모델 평가 전반에 대한 AI의 분석 신뢰도 점수 (0.0 ~ 1.0 사이).")


# 프롬프트 템플릿 정의
BIZ_MODEL_EVAL_PROMPT_TEMPLATE = """
당신은 AI 스타트업의 비즈니스 모델, 실적, 투자 조건 등을 심층 분석하는 투자 분석가입니다.
제공된 '{startup_name}' 스타트업에 대한 다음 정보를 바탕으로 사업 모델 및 재무적 측면을 평가해주십시오.

[제공된 정보]
{information_context}

[평가 항목]
1.  **주요 실적 (Track Record)**: 현재까지 파악된 주요 실적(예: 매출액, 사용자 수, 주요 계약 건수, 파트너십 등)을 요약하고, 이를 바탕으로 실적에 대한 점수를 1점에서 5점 사이로 매겨주세요. (정보가 부족하면 '정보 부족'으로 명시하고 점수는 null)
2.  **투자 조건 (Deal Terms)**: 현재 투자 라운드, 예상 기업 가치(Valuation), 주요 투자자, 투자 유치 이력 등 공개된 투자 조건 관련 정보가 있다면 요약하고, 투자 조건의 매력도에 대한 점수를 1점에서 5점 사이로 매겨주세요. (정보가 매우 제한적일 가능성이 높으므로, 부족하면 '정보 부족'으로 명시하고 점수는 null)
3.  **수익 모델 (Revenue Model)**: 이 스타트업의 주요 수익원은 무엇이며, 수익 모델의 지속 가능성과 확장성은 어떻게 평가하십니까?
4.  **고객 확보 전략 (GTM Strategy)**: 주요 목표 고객은 누구이며, 이들에게 도달하고 고객으로 전환하기 위한 핵심 시장 진출(Go-to-Market) 전략은 무엇입니까? (정보가 부족하면 '정보 부족'으로 명시)
5.  **분석 신뢰도**: 위 사업 모델 및 재무 관련 평가 항목들에 대한 당신의 전반적인 분석 신뢰도를 0.0에서 1.0 사이의 점수로 매겨주십시오. (예: 구체적인 실적 및 투자 정보가 풍부하면 0.9, 대부분 추론에 의존하면 0.5)

[출력 형식 지침]
다음 JSON 형식을 반드시 준수하여 답변해주십시오:
{{
    "track_record_summary": "(string)",
    "track_record_score": (float 또는 null),
    "deal_terms_comment": "(string 또는 null)",
    "deal_terms_score": (float 또는 null),
    "revenue_model_summary": "(string)",
    "g2m_strategy_summary": "(string 또는 null)",
    "evaluation_confidence_score": (float, 예: 0.70)
}}

스타트업 이름: {startup_name}
"""

def run_biz_model_evaluator(startup_name: str, scraped_info: List[SourceInfo], use_rag: bool = True, use_web_search: bool = True) -> BizModelEvaluationOutput:
    print(f"--- BizModelEvaluationAgent 실행 시작: {startup_name} ---")

    information_snippets = []
    all_sources_for_this_evaluation: List[SourceInfo] = []

    # 1. InfoScraper가 제공한 웹 스크랩 정보 컨텍스트화
    for src_info in scraped_info:
        updated_src_info = src_info.copy()
        if 'ai_confidence_score' not in updated_src_info:
            updated_src_info['ai_confidence_score'] = None
        information_snippets.append(f"기존 수집 정보 ({updated_src_info['source_url_or_doc']}, 신뢰도: {updated_src_info.get('ai_confidence_score', 'N/A')}):\n{updated_src_info['retrieved_content_snippet']}\n---")
        all_sources_for_this_evaluation.append(updated_src_info)

    # 2. (선택적) 추가 웹 검색으로 사업 모델/실적 정보 보강
    if use_web_search:
        biz_queries = [
            f"{startup_name} 비즈니스 모델",
            f"{startup_name} 수익 구조",
            f"{startup_name} 고객 유치 전략",
            f"{startup_name} 파트너십",
            f"{startup_name} 매출",
            f"{startup_name} 투자 유치 현황",
            f"{startup_name} 사용자 수"
        ]
        for query in biz_queries:
            web_results = search_web_for_startup(query)
            for res in web_results:
                source_info_entry = SourceInfo(
                    source_url_or_doc=res['source_url_or_doc'],
                    retrieved_content_snippet=res['retrieved_content_snippet'],
                    ai_confidence_score=None # 웹 검색 결과 신뢰도
                )
                information_snippets.append(f"웹 검색 ({res['source_url_or_doc']}, 신뢰도: N/A):\n{res['retrieved_content_snippet']}\n---")
                all_sources_for_this_evaluation.append(source_info_entry)

    # 3. (선택적) RAG를 통한 PDF 문서 정보 보강
    if use_rag:
        rag_query = f"{startup_name}의 사업 모델, 수익, 투자, 고객, 파트너십에 대한 정보"
        pdf_results = rag_tool_instance.search_documents(rag_query, k=2)
        for res in pdf_results:
            rag_source_info: SourceInfo = {
                "source_url_or_doc": f"{res['source_document']} (Page: {res['source_page']})",
                "retrieved_content_snippet": res["content"],
                "ai_confidence_score": None # RAG 결과 신뢰도
            }
            information_snippets.append(f"문서 출처 ({rag_source_info['source_url_or_doc']}, 신뢰도: N/A):\n{res['content']}\n---")
            all_sources_for_this_evaluation.append(rag_source_info)

    if not information_snippets:
        print(f"{startup_name}에 대한 사업 모델 분석 정보가 부족합니다.")
        return BizModelEvaluationOutput(
            track_record_summary="분석 정보 부족", track_record_score=None,
            deal_terms_comment="분석 정보 부족", deal_terms_score=None,
            revenue_model_summary="분석 정보 부족",
            g2m_strategy_summary="분석 정보 부족",
            evaluation_confidence_score=0.0, # 정보 부족 시 신뢰도 0
            sources=[]
        )

    information_context = "\n".join(information_snippets)
    # print(f"LLM에 전달될 사업 모델 컨텍스트 (일부): {information_context[:500]}...")

    prompt = ChatPromptTemplate.from_template(BIZ_MODEL_EVAL_PROMPT_TEMPLATE)
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
            biz_model_eval_output = BizModelEvaluationOutput(
                track_record_summary=llm_response_data.get("track_record_summary", "파싱 오류 또는 값 없음"),
                track_record_score=llm_response_data.get("track_record_score"),
                deal_terms_comment=llm_response_data.get("deal_terms_comment"),
                deal_terms_score=llm_response_data.get("deal_terms_score"),
                revenue_model_summary=llm_response_data.get("revenue_model_summary", "파싱 오류 또는 값 없음"),
                g2m_strategy_summary=llm_response_data.get("g2m_strategy_summary"),
                evaluation_confidence_score=llm_response_data.get("evaluation_confidence_score"), # 신뢰도 점수 추가
                sources=all_sources_for_this_evaluation
            )
        except json.JSONDecodeError as e:
            print(f"LLM 응답 JSON 파싱 오류: {e}")
            print(f"오류 발생한 응답: {response_str}")
            biz_model_eval_output = BizModelEvaluationOutput(
                track_record_summary=f"LLM 응답 파싱 오류: {response_str}", track_record_score=None,
                deal_terms_comment=f"LLM 응답 파싱 오류: {response_str}", deal_terms_score=None,
                revenue_model_summary=f"LLM 응답 파싱 오류: {response_str}",
                g2m_strategy_summary=f"LLM 응답 파싱 오류: {response_str}",
                evaluation_confidence_score=0.1, # 파싱 오류 시 낮은 신뢰도
                sources=all_sources_for_this_evaluation
            )
    except Exception as e:
        print(f"LLM 호출 중 오류 발생: {e}")
        biz_model_eval_output = BizModelEvaluationOutput(
            track_record_summary=f"LLM 호출 오류: {e}", track_record_score=None,
            deal_terms_comment=f"LLM 호출 오류: {e}", deal_terms_score=None,
            revenue_model_summary=f"LLM 호출 오류: {e}",
            g2m_strategy_summary=f"LLM 호출 오류: {e}",
            evaluation_confidence_score=0.1, # LLM 호출 오류 시 낮은 신뢰도
            sources=all_sources_for_this_evaluation
        )

    print(f"--- BizModelEvaluationAgent 실행 완료: {startup_name} ---")
    print(f"분석 결과: {biz_model_eval_output}")
    return biz_model_eval_output

if __name__ == '__main__':
    # BizModelEvaluationAgent 직접 실행 테스트
    sample_scraped_data_mago = [
        SourceInfo(source_url_or_doc="MAGO_news_article_1.com", retrieved_content_snippet="마고, 최근 시리즈 A 투자 유치 성공. 주요 투자자는 KB Investment. AI 음성 분석 기술 고도화 및 시장 확대 계획.", ai_confidence_score=0.85),
        SourceInfo(source_url_or_doc="https://www.holamago.com/pricing", retrieved_content_snippet="MAGO의 '오디온' 서비스는 사용량 기반 구독 모델 및 기업용 맞춤형 솔루션 제공. '카세트'는 B2B2C 모델로 정신건강 서비스 제공자와 협력.", ai_confidence_score=0.9),
        SourceInfo(source_url_or_doc="MAGO_MOU_Healthcare.com", retrieved_content_snippet="마고, 국내 대형 병원과 정신건강 관리 솔루션 도입을 위한 MOU 체결.", ai_confidence_score=0.75)
    ]
    sample_scraped_data_plang = [
        SourceInfo(source_url_or_doc="https://theplang.com/business", retrieved_content_snippet="The Plan G의 '오딩가 잉글리시'는 앱 내 부분 유료화(Freemium) 모델과 교육기관 대상 B2B 라이선스 판매를 주요 수익원으로 합니다.", ai_confidence_score=0.9),
        SourceInfo(source_url_or_doc="ThePlanG_downloads_report.com", retrieved_content_snippet="오딩가 잉글리시, 출시 1년 만에 누적 다운로드 15만 건 달성. 월간 활성 사용자(MAU) 3만 명.", ai_confidence_score=0.8), # PDF source 29 (page 5)
        SourceInfo(source_url_or_doc="ThePlanG_seed_funding.com", retrieved_content_snippet="더플랜지, 2016년 설립 후 초기 시드 투자 2억원 유치 성공.", ai_confidence_score=0.85) # PDF source 22 (page 4)
    ]

    print("\n--- MAGO 사업모델 평가 테스트 ---")
    mago_biz_eval = run_biz_model_evaluator(startup_name="MAGO", scraped_info=sample_scraped_data_mago, use_rag=True, use_web_search=True)
    # import json
    # print(json.dumps(mago_biz_eval, indent=2, ensure_ascii=False))

    print("\n--- The Plan G 사업모델 평가 테스트 ---")
    plang_biz_eval = run_biz_model_evaluator(startup_name="The Plan G", scraped_info=sample_scraped_data_plang, use_rag=True, use_web_search=True)
    # import json
    # print(json.dumps(plang_biz_eval, indent=2, ensure_ascii=False))