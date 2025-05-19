# AI-Service-Mini-Project-main/agents/comparative_analysis_agent.py
import sys
import os
import json
from typing import Dict, List, Any, Optional
from dotenv import load_dotenv

# 프로젝트 루트 디렉토리를 sys.path에 추가
PROJECT_ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if PROJECT_ROOT_DIR not in sys.path:
    sys.path.append(PROJECT_ROOT_DIR)

# .env 파일 로드 (파일 상단에서 한 번만 실행)
env_path = os.path.join(PROJECT_ROOT_DIR, '.env')
if os.path.exists(env_path):
    load_dotenv(dotenv_path=env_path)
else:
    print(f"경고: .env 파일을 찾을 수 없습니다. ({env_path}) OpenAI API 키가 환경 변수에 설정되어 있어야 합니다.")

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field

# 상태 정의에서 필요한 TypedDict 가져오기
from states.ai_mini_design_20_3반_배민하_state_py import (
    AdvancedStartupEvaluationState,
    SingleStartupDetailedAnalysis,
    SourceInfo,
    TechEvaluationOutput,
    MarketEvaluationOutput,
    TeamEvaluationOutput,
    BizModelEvaluationOutput
)

# LLM 초기화
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.3)

# LLM이 반환할 비교 분석 결과의 구조 정의
class LLMComparativeAnalysisResponse(BaseModel):
    comparison_summary_text: str = Field(description="여러 스타트업 간의 주요 비교 분석 결과를 종합적으로 요약한 텍스트입니다.")
    side_by_side_comparison_data: Dict[str, Dict[str, Any]] = Field(
        description="주요 평가 항목별로 스타트업들의 점수 또는 핵심 내용을 표 형태로 비교하는 데이터입니다. 각 항목명을 키로, 그 안에 스타트업 이름을 키로 하여 값(점수 또는 요약)을 갖는 딕셔너리 구조입니다."
    )
    overall_recommendation: str = Field(description="종합적인 투자 매력도 비교 및 어떤 유형의 투자자에게 어떤 스타트업이 더 적합한지에 대한 구체적인 제언입니다.")

# 프롬프트 템플릿 정의 수정
COMPARATIVE_ANALYSIS_PROMPT_TEMPLATE = """
당신은 다수의 AI 스타트업에 대한 심층 분석 자료를 바탕으로, 이들을 객관적이고 날카롭게 비교 분석하여 투자자에게 명확한 인사이트를 제공하는 최고 수준의 투자 분석가입니다.

다음은 개별적으로 상세 분석된 AI 스타트업들의 요약 정보입니다 (비교 대상 스타트업: {startup_names_to_compare_str}):
--------------------------------------
{individual_analyses_formatted_summary}
--------------------------------------

위 정보를 바탕으로 다음 스타트업들({startup_names_to_compare_str})을 종합적으로 비교 분석해주십시오.

[요청 분석 항목 및 출력 구조]

1.  **종합 비교 요약 (comparison_summary_text)**:
    * 각 스타트업의 핵심 강점 및 약점을 명확히 비교해주십시오.
    * 시장 내에서의 상대적 위치 및 주요 차별화 포인트를 분석해주십시오.
    * 종합적인 관점에서 어떤 스타트업이 더 매력적인 투자 대상인지, 혹은 어떤 측면에서 우위를 보이는지 명확한 근거와 함께 요약해주십시오.

2.  **주요 평가 항목별 직접 비교 데이터 (side_by_side_comparison_data)**:
    * 다음 주요 평가 항목들에 대해 각 스타트업의 핵심 내용이나 점수를 비교할 수 있도록 데이터를 구성해주십시오. 제공된 개별 분석 결과를 최대한 활용하여 객관적인 비교가 가능하도록 해주십시오.
    * **필수 포함 항목**: '기술_독창성_점수', '기술_구현가능성_점수', '시장_크기_매력도_점수', '시장_성장잠재력_점수', '팀_창업자전문성_점수', '팀_실행력_점수_추론', '사업모델_수익성_지속성', '사업모델_주요실적_요약'
    * 위 항목 외에도 비교에 중요하다고 판단되는 항목이 있다면 자유롭게 추가하되, 일관된 형식으로 비교해주십시오.
    * 데이터는 각 평가 항목명을 최상위 키로 하고, 그 값으로 각 스타트업 이름을 키로 하여 해당 스타트업의 평가 내용(점수 또는 간결한 요약 텍스트)을 값으로 갖는 딕셔너리 형태로 만들어 주십시오. (예: "기술_독창성_점수": {{ "스타트업A": 4.5, "스타트업B": "혁신적" }})

3.  **종합 투자 제언 (overall_recommendation)**:
    * 분석된 스타트업들의 전반적인 투자 매력도를 비교하고, 최종적인 투자 우선순위를 제시해주십시오.
    * 어떤 유형의 투자자(예: 초기 단계 기술 중심 투자자, 성장 단계 시장 확장 지원 투자자, 특정 산업 전문 투자자 등)에게 어떤 스타트업이 각각 더 매력적일 수 있는지, 그 이유와 함께 구체적으로 설명해주십시오.

[출력 형식 지침]
반드시 다음 JSON 형식을 준수하여 답변해주십시오:
{{
    "comparison_summary_text": "(string - 상세하고 논리적인 종합 비교 요약)",
    "side_by_side_comparison_data": {{
        "기술_독창성_점수": {{ "스타트업A 이름": (float 또는 string), "스타트업B 이름": (float 또는 string) /* ...기타 스타트업 */ }},
        "시장_성장잠재력_점수": {{ "스타트업A 이름": (float 또는 string), "스타트업B 이름": (float 또는 string) /* ... */ }}
        // ... 위에서 요청된 모든 '필수 포함 항목' 및 추가 항목 포함. 스타트업 이름은 {startup_names_to_compare_str} 에 명시된 실제 스타트업 이름을 사용하십시오.
    }},
    "overall_recommendation": "(string - 구체적이고 실행 가능한 종합 투자 제언)"
}}
"""

def format_analysis_for_llm(analysis: SingleStartupDetailedAnalysis) -> str:
    tech_eval = analysis.get('tech_evaluation', {})
    market_eval = analysis.get('market_evaluation', {})
    team_eval = analysis.get('team_evaluation', {})
    biz_model_eval = analysis.get('biz_model_evaluation', {})

    summary_parts = [
        f"스타트업 명: {analysis.get('startup_name', 'N/A')}",
        f"  회사 개요: {analysis.get('company_overview', 'N/A')[:200]}...",
        f"  기술 평가: 독창성 {tech_eval.get('originality_score', 'N/A')}점 ({tech_eval.get('originality_comment', 'N/A')[:100]}...), 구현가능성 {tech_eval.get('feasibility_score', 'N/A')}점 ({tech_eval.get('feasibility_comment', 'N/A')[:100]}...)",
        f"  시장 평가: 시장크기 {market_eval.get('market_size_score', 'N/A')}점 ({market_eval.get('market_size_comment', 'N/A')[:100]}...), 성장잠재력 {market_eval.get('growth_potential_score', 'N/A')}점 ({market_eval.get('growth_potential_comment', 'N/A')[:100]}...)",
        f"  팀 평가: 창업자전문성 {team_eval.get('founder_expertise_score', 'N/A')}점 ({team_eval.get('founder_expertise_comment', 'N/A')[:100]}...)",
        f"  사업모델 평가: 실적 {biz_model_eval.get('track_record_score', 'N/A')}점 ({biz_model_eval.get('track_record_summary', 'N/A')[:100]}...), 수익모델: {biz_model_eval.get('revenue_model_summary', 'N/A')[:100]}..."
    ]
    return "\n".join(summary_parts)

def run_comparative_analyzer(state: AdvancedStartupEvaluationState) -> Dict[str, Any]:
    print(f"--- ComparativeAnalysisAgent 실행 시작 ---")
    updates_to_state: Dict[str, Any] = {}

    individual_analyses: List[SingleStartupDetailedAnalysis] = state.get("individual_detailed_analyses", [])

    if not individual_analyses or len(individual_analyses) < 1:
        print("비교 분석을 위한 스타트업 정보가 부족합니다 (최소 1개 이상 필요).")
        updates_to_state["comparative_analysis_output"] = {"error": "비교 분석을 위한 스타트업 정보 부족 (최소 1개 또는 2개 필요)"}
        current_error_log = state.get("error_log", [])
        updates_to_state["error_log"] = current_error_log + ["ComparativeAnalysisAgent: 비교 분석 정보 부족"]
        return updates_to_state

    startup_names_list = [analysis['startup_name'] for analysis in individual_analyses]
    startup_names_to_compare_str = ", ".join(startup_names_list)

    formatted_summaries = [format_analysis_for_llm(analysis) for analysis in individual_analyses]
    individual_analyses_formatted_summary = "\n\n---\n\n".join(formatted_summaries)
    
    prompt_input = {
        "individual_analyses_formatted_summary": individual_analyses_formatted_summary,
        "startup_names_to_compare_str": startup_names_to_compare_str
    }
    
    prompt = ChatPromptTemplate.from_template(COMPARATIVE_ANALYSIS_PROMPT_TEMPLATE)
    chain = prompt | llm

    try:
        response_content = chain.invoke(prompt_input).content
        print(f"LLM 응답 (문자열): {response_content}")

        # 마크다운 코드 블록 제거
        if response_content.strip().startswith("```json"):
            response_content = response_content.strip()[7:] # ```json\n 제거
            if response_content.strip().endswith("```"):
                response_content = response_content.strip()[:-3] # \n``` 제거
        elif response_content.strip().startswith("```"): # json 명시 없이 ``` ... ``` 만 있을 경우
             response_content = response_content.strip()[3:]
             if response_content.strip().endswith("```"):
                response_content = response_content.strip()[:-3]
        response_content = response_content.strip()
        
        try:
            parsed_response = json.loads(response_content)
            updates_to_state["comparative_analysis_output"] = parsed_response
        except json.JSONDecodeError as e:
            print(f"LLM 응답 JSON 파싱 오류: {e}")
            print(f"오류 발생한 응답: {response_content}")
            updates_to_state["comparative_analysis_output"] = {"error": f"LLM 응답 파싱 오류: {response_content}"}
            current_error_log = state.get("error_log", [])
            updates_to_state["error_log"] = current_error_log + [f"ComparativeAnalysisAgent: JSON 파싱 오류 - {e}"]
    except Exception as e:
        print(f"LLM 호출 중 오류 발생: {e}")
        updates_to_state["comparative_analysis_output"] = {"error": f"LLM 호출 오류: {e}"}
        current_error_log = state.get("error_log", [])
        updates_to_state["error_log"] = current_error_log + [f"ComparativeAnalysisAgent: LLM 호출 오류 - {e}"]

    print(f"--- ComparativeAnalysisAgent 실행 완료 ---")
    if "error" not in updates_to_state.get("comparative_analysis_output", {}):
        print(f"비교 분석 요약: {updates_to_state.get('comparative_analysis_output', {}).get('comparison_summary_text', '요약 없음')[:200]}...")
    
    return updates_to_state

if __name__ == '__main__':
    # ComparativeAnalysisAgent 직접 실행 테스트
    if not os.getenv("OPENAI_API_KEY"):
        print("OPENAI_API_KEY 환경 변수가 설정되지 않았습니다. .env 파일을 확인하거나 직접 설정해주세요.")
    else:
        sample_mago_analysis = SingleStartupDetailedAnalysis(
            startup_name="MAGO",
            company_overview="MAGO는 음성 AI 및 정신건강 관리 플랫폼 개발 스타트업입니다. 주요 제품은 '오디온'과 '카세트'입니다.",
            tech_evaluation=TechEvaluationOutput(originality_score=4.0, originality_comment="음성 AI와 정신건강 결합 독창적이나 기반 기술은 기존 활용.", feasibility_score=4.5, feasibility_comment="기술 구현 가능성 높음. 데이터 민감성 고려 필요.", patent_info="정보 부족", tech_stack_summary="음성인식, NLP, 감정분석 (추정)", sources=[]),
            market_evaluation=MarketEvaluationOutput(market_size_score=4.2, market_size_comment="디지털 정신건강 시장 매력적", growth_potential_score=4.5, growth_potential_comment="AI 기반 솔루션 수요 증가로 잠재력 큼", competitive_landscape_summary="경쟁 강도 높으나 차별화 가능", entry_barriers_comment="기술 장벽 존재하나 전문성으로 극복 가능성.", network_effects_comment="네트워크 효과 존재", sources=[]),
            team_evaluation=TeamEvaluationOutput(founder_expertise_score=4.3, founder_expertise_comment="CEO(Hyunwoong Ko) 음성 AI 분야 박사 및 다수 연구 경력.", founder_communication_score=4.0, founder_communication_comment="비전 공유 및 외부 활동으로 역량 추정.", founder_execution_score=4.1, founder_execution_comment="초기 제품 개발 및 프로그램 선정 등 성과 있음.", key_team_members_summary="Sukbong Kwon (음성인식 전문가) 등 포함.", sources=[]),
            biz_model_evaluation=BizModelEvaluationOutput(track_record_summary="2022년 5월 설립, 시드 투자 유치 준비 중. Naver 협력 PoC 진행.", track_record_score=3.0, deal_terms_comment="시드 단계로 구체적 조건 미확인.", deal_terms_score=None, revenue_model_summary="B2B (기업용 솔루션), B2B2C (서비스 제공자 협력), 구독 모델 예상.", g2m_strategy_summary="주요 기업(Naver) 협력 통한 시장 진입 및 서비스 검증.", sources=[]),
            weighted_overall_score=None, strength_summary="", weakness_summary="", opportunity_summary="", threat_summary="", scraped_urls=[], key_documents_retrieved=[]
        )
        sample_plang_analysis = SingleStartupDetailedAnalysis(
            startup_name="The Plan G",
            company_overview="The Plan G는 AI 기반 초등 영어 학습 앱 '오딩가 잉글리시' 개발사. '가르치며 배우는' 방식 적용.",
            tech_evaluation=TechEvaluationOutput(originality_score=3.8, originality_comment="Teach-to-learn 방식과 AI 캐릭터 접목은 신선. 기반 기술(GenAI, Google API)은 기존 활용.", feasibility_score=4.0, feasibility_comment="구글 API 등 검증된 기술 활용으로 구현 용이. AI 캐릭터 고도화는 과제.", patent_info="정보 부족", tech_stack_summary="Generative AI, Google API", sources=[]),
            market_evaluation=MarketEvaluationOutput(market_size_score=4.0, market_size_comment="에듀테크 시장, 특히 초등 영어 시장 규모 크고 성장 중.", growth_potential_score=4.3, growth_potential_comment="개인화 학습 및 비대면 교육 수요 증가로 성장 잠재력 높음.", competitive_landscape_summary="유사 AI 영어 학습 앱 다수. 콘텐츠 및 학습 효과 차별화 중요.", entry_barriers_comment="기술 개발, 양질의 콘텐츠 확보, 마케팅 비용이 진입 장벽.", network_effects_comment="사용자 데이터 축적 시 AI 모델 개선 및 추천 정확도 향상 기대.", sources=[]),
            team_evaluation=TeamEvaluationOutput(founder_expertise_score=4.2, founder_expertise_comment="CEO(Kyunga Lee) 교육 콘텐츠 개발 15년 경력. 에듀테크 이해도 높음.", founder_communication_score=3.9, founder_communication_comment="인도 시장 진출 등 대외 활동으로 추정. 추가 정보 필요.", founder_execution_score=4.2, founder_execution_comment="앱 출시 및 인도 공교육 도입 등 구체적 성과. 추가 투자 유치 필요.", key_team_members_summary="정보 부족.", sources=[]),
            biz_model_evaluation=BizModelEvaluationOutput(track_record_summary="2016년 설립. 누적 다운로드 15만, MAU 3만. 시드 2억원 유치. 인도 공교육 커리큘럼 채택.", track_record_score=3.8, deal_terms_comment="초기 시드 투자 완료. 후속 투자 정보 필요.", deal_terms_score=3.5, revenue_model_summary="앱 내 부분 유료화 (Freemium), 교육기관 대상 B2B 라이선스.", g2m_strategy_summary="온라인 마케팅, 학교 파트너십, 콘텐츠 차별화 통한 사용자 확보.", sources=[]),
            weighted_overall_score=None, strength_summary="", weakness_summary="", opportunity_summary="", threat_summary="", scraped_urls=[], key_documents_retrieved=[]
        )

        test_state_for_comp_analysis = AdvancedStartupEvaluationState(
            startup_names_to_compare=["MAGO", "The Plan G"],
            individual_detailed_analyses=[sample_mago_analysis, sample_plang_analysis],
            messages=[],
            error_log=[]
        )

        comparative_output_dict_update = run_comparative_analyzer(test_state_for_comp_analysis)
        
        print("\n--- 최종 비교 분석 결과 (comparative_analysis_output) ---")
        if "error" not in comparative_output_dict_update.get("comparative_analysis_output", {}):
            print(json.dumps(comparative_output_dict_update.get("comparative_analysis_output"), indent=2, ensure_ascii=False))
        else:
            print(comparative_output_dict_update.get("comparative_analysis_output"))