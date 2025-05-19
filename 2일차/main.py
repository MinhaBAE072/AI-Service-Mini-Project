# AI-Service-Mini-Project-main/main.py
import os
import sys
import json
from typing import List, Dict, Any, Optional # TypedDict 제거, AdvancedStartupEvaluationState로 대체
from langgraph.graph import StateGraph, END
from dotenv import load_dotenv

# 프로젝트 루트를 sys.path에 추가
PROJECT_ROOT_DIR = os.path.abspath(os.path.dirname(__file__))
if PROJECT_ROOT_DIR not in sys.path:
    sys.path.append(PROJECT_ROOT_DIR)

# 상태 정의
from states.ai_mini_design_20_3반_배민하_state_py import (
    AdvancedStartupEvaluationState,
    SingleStartupDetailedAnalysis,
    TechEvaluationOutput, MarketEvaluationOutput, TeamEvaluationOutput, BizModelEvaluationOutput,
    SourceInfo
)

# 에이전트 함수들
from agents.info_scraper_agent import run_info_scraper
from agents.tech_eval_agent import run_tech_evaluator
from agents.market_eval_agent import run_market_evaluator
from agents.team_eval_agent import run_team_evaluator
from agents.biz_model_eval_agent import run_biz_model_evaluator
from agents.comparative_analysis_agent import run_comparative_analyzer
from agents.report_generator_agent import run_report_generator

# LLM 및 프롬프트 관련
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

# .env 파일 로드
load_dotenv()

# --- LLM 인스턴스 (전역 또는 필요시 각 함수 내에서 정의) ---
llm_general = ChatOpenAI(model="gpt-4o-mini", temperature=0.2)
llm_creative_summary = ChatOpenAI(model="gpt-4o-mini", temperature=0.4)

# --- 기본 가중치 정의 (Scorecard Method 기반, 합계 1.0으로 조정) ---
# 각 평가 항목의 어떤 '점수' 필드를 사용할지 명확히 해야 합니다.
# 예: Tech (originality + feasibility)/2, Market (market_size + growth_potential)/2 등
# 여기서는 각 EvaluationOutput의 대표 점수를 가정하거나, 세부 점수를 합산/평균하여 사용합니다.
# 편의상, 각 EvaluationOutput의 주요 점수 하나씩을 대표로 사용한다고 가정하고 가중치를 할당합니다.
# 실제로는 각 EvaluationOutput 내 여러 점수를 종합하는 방식이 더 정교할 수 있습니다.

DEFAULT_CRITERIA_WEIGHTS: Dict[str, float] = {
    # 키 이름은 각 EvaluationOutput의 주요 점수 필드명과 일치시키거나,
    # calculate_weighted_score 함수에서 매핑할 수 있도록 명확하게 정의합니다.
    # 여기서는 각 평가 영역에 대한 가중치로 정의하고, 세부 점수는 함수 내에서 처리합니다.
    "tech_score": 0.15,       # 예: (originality_score + feasibility_score) / 2
    "market_score": 0.25,     # 예: (market_size_score + growth_potential_score) / 2
    "team_score": 0.30,       # 예: (founder_expertise_score + founder_execution_score) / 2
    "biz_model_score": 0.30    # 예: (track_record_score + revenue_model 관련 점수 + deal_terms_score) / 3 (세부항목 조정 필요)
                                # 기존 BizModel은 실적 10%, 투자조건 10% 등 세분화 되어 있었음
                                # 가중치 합이 1.0 이 되도록 조정 (0.15 + 0.25 + 0.30 + 0.30 = 1.0)
}

# --- 가중 점수 계산 유틸리티 함수 ---
def calculate_weighted_score(
    tech_eval: TechEvaluationOutput,
    market_eval: MarketEvaluationOutput,
    team_eval: TeamEvaluationOutput,
    biz_model_eval: BizModelEvaluationOutput,
    weights: Dict[str, float]
) -> Optional[float]:
    
    print(f"--- 가중 점수 계산 시작 (가중치: {weights}) ---")
    total_weighted_score = 0.0
    # 각 평가 항목에서 점수를 가져옵니다. 점수가 None일 경우 0점으로 처리하거나, 가중치 계산에서 제외할 수 있습니다.
    # 여기서는 None일 경우 해당 항목 기여도를 0으로 처리합니다.

    # 기술 점수 (예: 독창성 + 구현 가능성 평균)
    tech_scores_present = []
    if tech_eval.get("originality_score") is not None:
        tech_scores_present.append(tech_eval["originality_score"])
    if tech_eval.get("feasibility_score") is not None:
        tech_scores_present.append(tech_eval["feasibility_score"])
    
    avg_tech_score = sum(tech_scores_present) / len(tech_scores_present) if tech_scores_present else 0
    total_weighted_score += avg_tech_score * weights.get("tech_score", 0)
    print(f"기술 점수: {avg_tech_score:.2f}, 가중 기여도: {(avg_tech_score * weights.get('tech_score', 0)):.2f}")

    # 시장 점수 (예: 시장 크기 + 성장 잠재력 평균)
    market_scores_present = []
    if market_eval.get("market_size_score") is not None:
        market_scores_present.append(market_eval["market_size_score"])
    if market_eval.get("growth_potential_score") is not None:
        market_scores_present.append(market_eval["growth_potential_score"])

    avg_market_score = sum(market_scores_present) / len(market_scores_present) if market_scores_present else 0
    total_weighted_score += avg_market_score * weights.get("market_score", 0)
    print(f"시장 점수: {avg_market_score:.2f}, 가중 기여도: {(avg_market_score * weights.get('market_score', 0)):.2f}")

    # 팀 점수 (예: 창업자 전문성 + 실행력 평균)
    team_scores_present = []
    if team_eval.get("founder_expertise_score") is not None:
        team_scores_present.append(team_eval["founder_expertise_score"])
    if team_eval.get("founder_execution_score") is not None: # 실행력 점수
        team_scores_present.append(team_eval["founder_execution_score"])
    
    avg_team_score = sum(team_scores_present) / len(team_scores_present) if team_scores_present else 0
    total_weighted_score += avg_team_score * weights.get("team_score", 0)
    print(f"팀 점수: {avg_team_score:.2f}, 가중 기여도: {(avg_team_score * weights.get('team_score', 0)):.2f}")
    
    # 사업 모델 점수 (예: 실적 점수. 더 많은 세부 항목을 포함하도록 확장 가능)
    # BizModelEvaluationOutput에는 track_record_score, deal_terms_score 등이 있음.
    # 여기서는 track_record_score만 사용. 필요시 가중치 세분화 및 deal_terms_score 등 추가.
    biz_model_scores_present = []
    if biz_model_eval.get("track_record_score") is not None:
        biz_model_scores_present.append(biz_model_eval["track_record_score"])
    # if biz_model_eval.get("deal_terms_score") is not None: # 투자 조건 점수도 반영한다면
    #     biz_model_scores_present.append(biz_model_eval["deal_terms_score"])

    avg_biz_model_score = sum(biz_model_scores_present) / len(biz_model_scores_present) if biz_model_scores_present else 0
    total_weighted_score += avg_biz_model_score * weights.get("biz_model_score", 0) # 'biz_model_score' 가중치 사용
    print(f"사업모델 점수: {avg_biz_model_score:.2f}, 가중 기여도: {(avg_biz_model_score * weights.get('biz_model_score', 0)):.2f}")

    # 최종 점수는 5점 만점으로 스케일링 될 수 있도록, 또는 가중치 합이 1이므로 그대로 사용.
    # 모든 점수가 1-5점 척도라고 가정하면, 최종 점수도 유사한 범위를 가질 것.
    # 만약 모든 점수가 0일 경우, None을 반환하거나 0.0을 반환.
    if not tech_scores_present and not market_scores_present and not team_scores_present and not biz_model_scores_present:
        print("가중 점수 계산 불가: 모든 평가 항목 점수 없음.")
        return None
        
    print(f"--- 최종 가중 점수: {total_weighted_score:.2f} ---")
    return round(total_weighted_score, 2)


# --- 유틸리티 함수: 회사 개요 및 SWOT 분석 생성 (기존 코드 유지) ---
def generate_company_overview_llm(startup_name: str,
                                  retrieved_sources: List[SourceInfo],
                                  tech_eval: TechEvaluationOutput,
                                  market_eval: MarketEvaluationOutput,
                                  team_eval: TeamEvaluationOutput,
                                  biz_model_eval: BizModelEvaluationOutput) -> str:
    print(f"--- {startup_name} 회사 개요 LLM 생성 시작 ---")
    context_parts = [f"스타트업 '{startup_name}'에 대한 정보입니다."]

    if retrieved_sources:
        context_parts.append("\n[수집된 웹 정보 요약]")
        for i, src in enumerate(retrieved_sources[:2]): # 처음 2개 웹 정보 스니펫 활용
            context_parts.append(f"  - 출처 {i+1} ({src.get('source_url_or_doc', 'N/A')}, 신뢰도: {src.get('ai_confidence_score', 'N/A')}): {src.get('retrieved_content_snippet', 'N/A')[:150]}...")

    context_parts.append("\n[전문가 분석 요약]")
    context_parts.append(f"  - 기술/제품 (신뢰도: {tech_eval.get('evaluation_confidence_score', 'N/A')}): {tech_eval.get('originality_comment', '코멘트 없음')[:100]}...")
    context_parts.append(f"  - 시장성 (신뢰도: {market_eval.get('evaluation_confidence_score', 'N/A')}): {market_eval.get('market_size_comment', '코멘트 없음')[:100]}...")
    context_parts.append(f"  - 팀 역량 (신뢰도: {team_eval.get('evaluation_confidence_score', 'N/A')}): {team_eval.get('founder_expertise_comment', '코멘트 없음')[:100]}...")
    context_parts.append(f"  - 사업 모델 (신뢰도: {biz_model_eval.get('evaluation_confidence_score', 'N/A')}): {biz_model_eval.get('revenue_model_summary', '코멘트 없음')[:100]}...")
    
    information_context = "\n".join(context_parts)

    prompt_template = ChatPromptTemplate.from_template(
        """당신은 스타트업 분석가입니다. 제공된 정보를 바탕으로 '{startup_name}' 회사에 대한 핵심 내용을 담은 간결한 회사 개요를 2~3문장으로 작성해주세요. 
        이 회사가 무엇을 하는 곳인지, 주요 제품이나 서비스는 무엇인지, 어떤 시장을 목표로 하는지가 명확히 드러나도록 합니다. 각 정보의 신뢰도도 참고하십시오.

        [제공된 정보]
        {information_context}

        [작성할 회사 개요]
        """
    )
    chain = prompt_template | llm_creative_summary 
    try:
        response = chain.invoke({"startup_name": startup_name, "information_context": information_context})
        overview = response.content.strip()
        print(f"--- {startup_name} 회사 개요 생성 완료 ---")
        return overview
    except Exception as e:
        print(f"회사 개요 생성 중 오류 발생: {e}")
        return f"{startup_name}의 회사 개요를 생성하는 중 오류가 발생했습니다. 수집된 정보를 참고해주십시오."

def generate_swot_analysis_llm(startup_name: str,
                               tech_eval: TechEvaluationOutput,
                               market_eval: MarketEvaluationOutput,
                               team_eval: TeamEvaluationOutput,
                               biz_model_eval: BizModelEvaluationOutput) -> Dict[str, str]:
    print(f"--- {startup_name} SWOT 분석 LLM 생성 시작 ---")
    
    context_parts = [f"스타트업 '{startup_name}'의 강점(Strength), 약점(Weakness), 기회(Opportunity), 위협(Threat) 요인을 분석하기 위한 정보입니다. 각 분석 결과의 신뢰도도 참고하여 판단해주십시오."]
    context_parts.append("\n[기술 및 제품 분석 (신뢰도: {tech_eval.get('evaluation_confidence_score', 'N/A')})]")
    context_parts.append(f"  - 독창성: {tech_eval.get('originality_score')}점, {tech_eval.get('originality_comment')}")
    context_parts.append(f"  - 구현가능성: {tech_eval.get('feasibility_score')}점, {tech_eval.get('feasibility_comment')}")
    context_parts.append(f"  - 기술스택: {tech_eval.get('tech_stack_summary', '정보 부족')}")

    context_parts.append("\n[시장 분석 (신뢰도: {market_eval.get('evaluation_confidence_score', 'N/A')})]")
    context_parts.append(f"  - 시장크기: {market_eval.get('market_size_score')}점, {market_eval.get('market_size_comment')}")
    context_parts.append(f"  - 성장잠재력: {market_eval.get('growth_potential_score')}점, {market_eval.get('growth_potential_comment')}")
    context_parts.append(f"  - 경쟁환경: {market_eval.get('competitive_landscape_summary')}")
    context_parts.append(f"  - 진입장벽: {market_eval.get('entry_barriers_comment', '정보 부족')}")

    context_parts.append("\n[팀 역량 분석 (신뢰도: {team_eval.get('evaluation_confidence_score', 'N/A')})]")
    context_parts.append(f"  - 창업자 전문성: {team_eval.get('founder_expertise_score')}점, {team_eval.get('founder_expertise_comment')}")
    context_parts.append(f"  - 창업자 실행력(추론): {team_eval.get('founder_execution_score')}점, {team_eval.get('founder_execution_comment')}")
    context_parts.append(f"  - 핵심팀원: {team_eval.get('key_team_members_summary', '정보 부족')}")

    context_parts.append("\n[사업 모델 및 실적 분석 (신뢰도: {biz_model_eval.get('evaluation_confidence_score', 'N/A')})]")
    context_parts.append(f"  - 수익모델: {biz_model_eval.get('revenue_model_summary')}")
    context_parts.append(f"  - 주요실적: {biz_model_eval.get('track_record_summary')} (점수: {biz_model_eval.get('track_record_score')})")
    context_parts.append(f"  - GTM전략: {biz_model_eval.get('g2m_strategy_summary', '정보 부족')}")
    
    information_context = "\n".join(context_parts)

    prompt_template = ChatPromptTemplate.from_template(
        """당신은 경험 많은 스타트업 분석가입니다. 제공된 '{startup_name}'에 대한 상세 분석 정보를 바탕으로, 이 스타트업의 강점(Strength), 약점(Weakness), 기회(Opportunity), 위협(Threat) 요인을 각각 1~2개의 핵심 문장으로 구체적으로 기술해주십시오. 
        각 요인은 제공된 분석 내용에 근거해야 합니다. 각 분석 결과의 신뢰도를 참고하여 판단의 강도를 조절할 수 있습니다.

        [제공된 분석 정보]
        {information_context}

        [출력할 JSON 형식]
        {{
            "strength_summary": "강점1. 강점2.",
            "weakness_summary": "약점1. 약점2.",
            "opportunity_summary": "기회1. 기회2.",
            "threat_summary": "위협1. 위협2."
        }}
        """
    )
    chain = prompt_template | llm_creative_summary 
    default_swot = {"strength_summary": "분석 정보 부족", "weakness_summary": "분석 정보 부족", "opportunity_summary": "분석 정보 부족", "threat_summary": "분석 정보 부족"}
    try:
        response_content = chain.invoke({"startup_name": startup_name, "information_context": information_context}).content
        if response_content.strip().startswith("```json"):
            response_content = response_content.strip()[7:]
            if response_content.strip().endswith("```"):
                response_content = response_content.strip()[:-3]
        elif response_content.strip().startswith("```"):
            response_content = response_content.strip()[3:]
            if response_content.strip().endswith("```"):
                response_content = response_content.strip()[:-3]
        response_content = response_content.strip()
        
        swot_data = json.loads(response_content)
        print(f"--- {startup_name} SWOT 분석 생성 완료 ---")
        return {
            "strength_summary": swot_data.get("strength_summary", "SWOT 강점 생성 실패"),
            "weakness_summary": swot_data.get("weakness_summary", "SWOT 약점 생성 실패"),
            "opportunity_summary": swot_data.get("opportunity_summary", "SWOT 기회 생성 실패"),
            "threat_summary": swot_data.get("threat_summary", "SWOT 위협 생성 실패")
        }
    except Exception as e:
        print(f"SWOT 분석 생성 중 오류 발생: {e}")
        return default_swot

# --- Supervisor 및 그래프 노드 함수들 ---
def supervisor_start(state: AdvancedStartupEvaluationState) -> AdvancedStartupEvaluationState:
    print("--- Supervisor: 전체 워크플로우 시작 ---")
    state["individual_detailed_analyses"] = []
    state["current_startup_index"] = 0
    startup_names_list = state.get("startup_names_to_compare", [])
    state["current_startup_name"] = startup_names_list[0] if startup_names_list else None
    print(f"DEBUG supervisor_start: current_startup_index = {state.get('current_startup_index')}, current_startup_name = '{state.get('current_startup_name')}'")
    return state

def state_check_node_after_scraper(state: AdvancedStartupEvaluationState) -> AdvancedStartupEvaluationState:
    print(f"DEBUG state_check_node_after_scraper entry: current_idx = {state.get('current_startup_index')}, current_name = '{state.get('current_startup_name')}'")
    return state

def process_single_startup(state: AdvancedStartupEvaluationState) -> AdvancedStartupEvaluationState:
    current_idx = state.get("current_startup_index")
    startup_names_list = state.get("startup_names_to_compare", [])
    print(f"DEBUG process_single_startup entry: current_idx = {current_idx}, state's current_startup_name = '{state.get('current_startup_name')}'")

    if current_idx is None or not startup_names_list or current_idx >= len(startup_names_list):
        error_msg = f"유효한 스타트업 인덱스({current_idx}) 또는 목록({startup_names_list})이 없습니다."
        print(f"오류: {error_msg}")
        state["error_log"] = state.get("error_log", []) + [f"Supervisor: {error_msg}"]
        return state

    current_startup_name = startup_names_list[current_idx]
    state["current_startup_name"] = current_startup_name
    print(f"\n--- {current_startup_name} 상세 분석 시작 (인덱스: {current_idx}) ---")

    scraped_data_for_current_startup = [] 
    initial_scraped_data = state.get("initial_scraped_data_all_startups", [])
    for data in initial_scraped_data:
        if data.get("startup_name") == current_startup_name:
            scraped_data_for_current_startup = data.get("retrieved_sources", [])
            break
    
    if not scraped_data_for_current_startup:
        print(f"경고: {current_startup_name}에 대한 스크랩된 정보(웹)가 없습니다. RAG 결과 및 PDF 정보에 의존합니다.")

    print(f"DEBUG {current_startup_name}: Tech Evaluator 호출")
    tech_eval = run_tech_evaluator(current_startup_name, scraped_data_for_current_startup)
    print(f"DEBUG {current_startup_name}: Market Evaluator 호출")
    market_eval = run_market_evaluator(current_startup_name, scraped_data_for_current_startup)
    print(f"DEBUG {current_startup_name}: Team Evaluator 호출")
    team_eval = run_team_evaluator(current_startup_name, scraped_data_for_current_startup)
    print(f"DEBUG {current_startup_name}: Biz Model Evaluator 호출")
    biz_model_eval = run_biz_model_evaluator(current_startup_name, scraped_data_for_current_startup)

    company_overview_text = generate_company_overview_llm(
        current_startup_name,
        scraped_data_for_current_startup,
        tech_eval, market_eval, team_eval, biz_model_eval
    )

    swot_results = generate_swot_analysis_llm(
        current_startup_name,
        tech_eval, market_eval, team_eval, biz_model_eval
    )

    # 가중치 적용 및 최종 점수 계산
    active_weights = state.get("user_defined_criteria_weights") or DEFAULT_CRITERIA_WEIGHTS
    final_weighted_score = calculate_weighted_score(
        tech_eval, market_eval, team_eval, biz_model_eval, active_weights
    )

    analysis_entry = SingleStartupDetailedAnalysis(
        startup_name=current_startup_name,
        company_overview=company_overview_text, 
        tech_evaluation=tech_eval,
        market_evaluation=market_eval,
        team_evaluation=team_eval,
        biz_model_evaluation=biz_model_eval,
        weighted_overall_score=final_weighted_score, # 계산된 가중 점수 적용
        strength_summary=swot_results.get("strength_summary", "SWOT 분석 정보 부족"),
        weakness_summary=swot_results.get("weakness_summary", "SWOT 분석 정보 부족"),
        opportunity_summary=swot_results.get("opportunity_summary", "SWOT 분석 정보 부족"),
        threat_summary=swot_results.get("threat_summary", "SWOT 분석 정보 부족"),
        scraped_urls=[src['source_url_or_doc'] for src in (tech_eval.get('sources', []) + market_eval.get('sources', []) + team_eval.get('sources', []) + biz_model_eval.get('sources', [])) if src.get('source_url_or_doc', '').startswith("http")],
        key_documents_retrieved=[src['source_url_or_doc'] for src in (tech_eval.get('sources', []) + market_eval.get('sources', []) + team_eval.get('sources', []) + biz_model_eval.get('sources', [])) if "pdf" in src.get('source_url_or_doc', '').lower()]
    )
    
    current_analyses = state.get("individual_detailed_analyses", [])
    current_analyses.append(analysis_entry)
    state["individual_detailed_analyses"] = current_analyses
    
    if current_idx is not None:
        state["_temp_next_index"] = current_idx + 1 
        print(f"DEBUG process_single_startup_node SETTING _temp_next_index to: {state['_temp_next_index']}")
    
    print(f"--- {current_startup_name} 상세 분석 완료 (가중 점수: {final_weighted_score}) ---")
    print(f"DEBUG process_single_startup exit: current_idx = {current_idx}, current_startup_name = '{current_startup_name}', _temp_next_index_in_state = {state.get('_temp_next_index')}")
    return state

def update_index_node_func(state: AdvancedStartupEvaluationState) -> AdvancedStartupEvaluationState:
    print(f"DEBUG update_index_node_func ENTRY - _temp_next_index: {state.get('_temp_next_index')}")
    next_index = state.get("_temp_next_index")
    if next_index is not None:
        state["current_startup_index"] = next_index
        startup_names_list = state.get("startup_names_to_compare", [])
        if next_index < len(startup_names_list):
            state["current_startup_name"] = startup_names_list[next_index]
            print(f"DEBUG update_index_node_func SUCCESS - Updated index: {state['current_startup_index']}, name: {state['current_startup_name']}")
        else:
            print(f"DEBUG update_index_node_func: 모든 스타트업 처리 완료됨 (인덱스 {next_index}가 범위 초과).")
        
        if "_temp_next_index" in state:
             del state["_temp_next_index"]
    else:
        print("ERROR in update_index_node_func: _temp_next_index is None.")
        state["error_log"] = state.get("error_log", []) + ["Error: _temp_next_index was None in update_index_node_func"]
    return state

def should_continue_processing(state: AdvancedStartupEvaluationState) -> str:
    potential_next_index = state.get("_temp_next_index")
    total_startups = len(state.get("startup_names_to_compare", []))
    last_processed_name = state.get('current_startup_name')

    print(f"DEBUG should_continue_processing entry: potential_next_index_from_state = {potential_next_index}, total_startups = {total_startups}, last_processed_name = '{last_processed_name}'")

    if potential_next_index is not None and potential_next_index < total_startups:
        print(f"--- Supervisor: 다음 스타트업 처리를 위해 update_index_node로 라우팅 (다음 인덱스: {potential_next_index}) ---")
        return "go_to_update_index_node"
    else:
        print("--- Supervisor: 모든 스타트업 개별 분석 완료. 비교 분석 단계로 라우팅. ---")
        return "prepare_for_comparison"

# --- LangGraph 워크플로우 정의 (기존 코드 유지) ---
workflow = StateGraph(AdvancedStartupEvaluationState)

workflow.add_node("supervisor_start_node", supervisor_start)
workflow.add_node("run_info_scraper_node", run_info_scraper)
workflow.add_node("state_check_node_after_scraper", state_check_node_after_scraper)
workflow.add_node("process_single_startup_node", process_single_startup)
workflow.add_node("update_index_node", update_index_node_func)
workflow.add_node("comparative_analysis_node", run_comparative_analyzer)
workflow.add_node("report_generator_node", run_report_generator)

workflow.set_entry_point("supervisor_start_node")
workflow.add_edge("supervisor_start_node", "run_info_scraper_node")
workflow.add_edge("run_info_scraper_node", "state_check_node_after_scraper")
workflow.add_edge("state_check_node_after_scraper", "process_single_startup_node")
workflow.add_edge("update_index_node", "process_single_startup_node")

workflow.add_conditional_edges(
    "process_single_startup_node",
    should_continue_processing,
    {
        "go_to_update_index_node": "update_index_node",
        "prepare_for_comparison": "comparative_analysis_node"
    }
)
workflow.add_edge("comparative_analysis_node", "report_generator_node")
workflow.add_edge("report_generator_node", END)

app = workflow.compile()

# --- 워크플로우 실행 (예시) ---
if __name__ == "__main__":
    startup_list_to_compare = ["MAGO", "The Plan G"]

    # 사용자 정의 가중치 예시 (선택적)
    # user_weights = {
    #     "tech_score": 0.20,
    #     "market_score": 0.30,
    #     "team_score": 0.25,
    #     "biz_model_score": 0.25 
    # }

    initial_state_dict: AdvancedStartupEvaluationState = {
        "startup_names_to_compare": startup_list_to_compare,
        "user_defined_criteria_weights": None, # 또는 user_weights,
        "initial_scraped_data_all_startups": [],
        "individual_detailed_analyses": [],
        "comparative_analysis_output": {},
        "final_comparison_report_text": "",
        "messages": [], # type: ignore
        "error_log": [],
    }

    print(f"워크플로우 실행: {startup_list_to_compare}에 대한 분석 시작")
    final_state = app.invoke(initial_state_dict, {"recursion_limit": 150})

    print("\n\n--- 최종 워크플로우 실행 완료 ---")
    print("\n--- 개별 스타트업 분석 결과 (요약) ---")
    if final_state and final_state.get("individual_detailed_analyses"):
        for analysis in final_state["individual_detailed_analyses"]:
            print(f"\n스타트업: {analysis.get('startup_name')}")
            print(f"  회사 개요: {analysis.get('company_overview', 'N/A')}")
            print(f"  기술 평가 (독창성 코멘트): {analysis.get('tech_evaluation', {}).get('originality_comment', 'N/A')} (신뢰도: {analysis.get('tech_evaluation', {}).get('evaluation_confidence_score', 'N/A')})")
            print(f"  시장 평가 (시장 크기 코멘트): {analysis.get('market_evaluation', {}).get('market_size_comment', 'N/A')} (신뢰도: {analysis.get('market_evaluation', {}).get('evaluation_confidence_score', 'N/A')})")
            print(f"  팀 평가 (창업자 전문성 코멘트): {analysis.get('team_evaluation', {}).get('founder_expertise_comment', 'N/A')} (신뢰도: {analysis.get('team_evaluation', {}).get('evaluation_confidence_score', 'N/A')})")
            print(f"  사업모델 평가 (수익 모델): {analysis.get('biz_model_evaluation', {}).get('revenue_model_summary', 'N/A')} (신뢰도: {analysis.get('biz_model_evaluation', {}).get('evaluation_confidence_score', 'N/A')})")
            print(f"  SWOT - 강점: {analysis.get('strength_summary', 'N/A')}")
            print(f"  SWOT - 약점: {analysis.get('weakness_summary', 'N/A')}")
            print(f"  최종 가중 점수: {analysis.get('weighted_overall_score', '계산 안됨')}")
    else:
        print("개별 분석 결과가 없습니다.")

    print("\n--- 비교 분석 결과 (요약) ---")
    if final_state and final_state.get("comparative_analysis_output"):
        comp_output = final_state["comparative_analysis_output"]
        if "error" in comp_output:
            print(f"비교 분석 중 오류: {comp_output['error']}")
        else:
            print(f"  종합 비교 요약: {comp_output.get('comparison_summary_text', 'N/A')[:300]}...")
            # 추가: 상세 비교표 데이터 일부 출력 (필요시)
            # side_by_side_data = comp_output.get('side_by_side_comparison_data', {})
            # if side_by_side_data:
            #     print("  항목별 비교 (예시):")
            #     for key, value_dict in list(side_by_side_data.items())[:2]: # 처음 2개 항목만 예시로 출력
            #         print(f"    {key}: {value_dict}")
    else:
        print("비교 분석 결과가 없습니다.")

    print("\n--- 최종 생성 보고서 (final_comparison_report_text 일부) ---")
    if final_state and final_state.get("final_comparison_report_text"):
        report_text = final_state["final_comparison_report_text"]
        if "실패" not in report_text and report_text:
            print(report_text[:2000] + "...")
            try:
                with open("final_ai_startup_comparison_report_v3.md", "w", encoding="utf-8") as f: # 파일명 변경 (v3)
                    f.write(report_text)
                print("\n보고서가 'final_ai_startup_comparison_report_v3.md' 파일로 저장되었습니다.")
            except Exception as e:
                print(f"\n보고서 파일 저장 중 오류 발생: {e}")
        else:
            print(f"보고서 생성 실패 또는 내용 없음: {report_text}")
    else:
        print("생성된 보고서가 없습니다.")

    if final_state and final_state.get("error_log"):
        print("\n--- 최종 오류 로그 ---")
        for error in final_state["error_log"]:
            print(error)